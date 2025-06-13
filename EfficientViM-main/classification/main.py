import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter


from data import build_loader
from utils.config import get_config
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from utils.losses import DistillationLoss

from timm.utils import ModelEma as ModelEma
from torch.utils.tensorboard import SummaryWriter

from models import *

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('EfficientViM training and evaluation script', add_help=False)
    # easy config modification
    parser.add_argument('--name', type=str, default="EfficientViM_M1", help="Model Name")
    
    parser.add_argument('--epochs', type=int, default=300, help="epochs")
    parser.add_argument('--warmup-epochs', type=int, default=20, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    
    parser.add_argument('--batch-size', type=int, default=256, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<MODEL.NAME>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9995, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--ddp', type=str, default='torch', help='distributed data parallel')
    parser.add_argument('--disable_mesa', action='store_true', help='Disable MESA')
    
    # Distillation parametersdistillation_type
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help="")

    
    
    # Personal configs
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    if config.MODEL.NAME=="EfficientViM_M4":
        config.defrost()
        config.DATA.IMG_SIZE = 256
        config.freeze()
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = create_model(config.MODEL.NAME,
        num_classes=config.MODEL.NUM_CLASSES,
        distillation=(args.distillation_type != 'none'),
        pretrained=args.eval,
    )

    if dist.get_rank() == 0:
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params (M): {n_parameters / 1e6}")
        flops = model.flops(shape=(3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        logger.info(f"number of MFLOPs: {flops / 1e6} with (3, {config.DATA.IMG_SIZE}, {config.DATA.IMG_SIZE})")

    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        
    optimizer = build_optimizer(config, model, logger, mute_repeat=args.mute_repeat)
    if args.ddp == 'torch':
        model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    else:
        raise ValueError(f"Unknown ddp type {args.ddp}")

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.TRAIN.DISTIILATION.DISTILLATION_TYPE != 'none':
        assert config.TRAIN.DISTIILATION.TEACHER_PATH, 'need to specify teacher-path when using distillation'
        teacher_model = create_model(
            config.TRAIN.DISTIILATION.TEACHER_MODEL,
            pretrained=False,
            num_classes= config.MODEL.NUM_CLASSES,
            global_pool='avg',
        )
        if config.TRAIN.DISTIILATION.TEACHER_PATH.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.TRAIN.DISTIILATION.TEACHER_PATH, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.TRAIN.DISTIILATION.TEACHER_PATH, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.cuda()
        teacher_model.eval()
        
        criterion = DistillationLoss(
            criterion, teacher_model, config.TRAIN.DISTIILATION.DISTILLATION_TYPE, config.TRAIN.DISTIILATION.DISTILLATION_ALPHA, config.TRAIN.DISTIILATION.DISTILLATION_TAU
        )

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    steps = 0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, max_accuracy_ema, steps = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler,
                                                                    loss_scaler, logger, model_ema)
        if steps + 1 == len(data_loader_train):
            config.defrost()
            config.TRAIN.START_EPOCH += 1
            config.freeze()
            steps = 0
        if config.EVAL_MODE:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if model_ema is not None:
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
                logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    writer = SummaryWriter((os.path.join(config.OUTPUT, 'tensorboard')))

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS+config.TRAIN.COOLDOWN_EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, model_ema, steps=steps,
                        mesa=1.0 if (epoch >= int(0.25 * config.TRAIN.EPOCHS) and not args.disable_mesa and args.distillation_type == "none") else -1.0)
        steps = 0
        if dist.get_rank() == 0:
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                loss_scaler, logger, model_ema, max_accuracy_ema, steps=0, ckpt_name='latest_ckpt')
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # Log the accuracy to TensorBoard
        if dist.get_rank() == 0:
            writer.add_scalar('Accuracy/val', acc1, epoch)

        # Check if current accuracy is higher than the max accuracy
        if acc1 > max_accuracy:
            max_accuracy = acc1
            logger.info(f'New max accuracy: {max_accuracy:.2f}%')
            # Save the model if this is the best accuracy so far
            if dist.get_rank() == 0:
                save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                    loss_scaler, logger, model_ema, max_accuracy_ema, steps=0, ckpt_name='best_ckpt')
        
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

            # Check if current EMA accuracy is higher than the max EMA accuracy
            # Log the EMA accuracy to TensorBoard
            if dist.get_rank() == 0:
                writer.add_scalar('Accuracy_ema/val', acc1_ema, epoch)

            if acc1_ema > max_accuracy_ema:
                max_accuracy_ema = acc1_ema
                logger.info(f'New max accuracy ema: {max_accuracy_ema:.2f}%')
                # Save the model if this is the best EMA accuracy so far
                if dist.get_rank() == 0:
                    save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                        loss_scaler, logger, model_ema, max_accuracy_ema, steps=0,
                                        ckpt_name='best_ckpt_ema')
                    
    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn,
                    lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, steps=0, mesa=-1):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader) + steps
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        idx += steps

        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            if mesa > 0.0:
                with torch.inference_mode():
                    ema_output = model_ema.ema(samples).detach()
                ema_output = torch.clone(ema_output)
                ema_output = ema_output.softmax(dim=-1).detach()
                ema_loss = criterion(outputs, ema_output) * mesa

        if config.TRAIN.DISTIILATION.DISTILLATION_TYPE != "none":
            loss = criterion(samples, outputs, targets) 
        elif mesa > 0.0:
            loss = criterion(outputs, targets) + ema_loss
        else:
            loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    try:
        ddp = config.MODEL.DDP
    except:
        ddp = 'torch'
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1, ddp=ddp)
        acc5 = reduce_tensor(acc5, ddp=ddp)
        loss = reduce_tensor(loss, ddp=ddp)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

if __name__ == '__main__':

    args, config = parse_option()

    stime = time.time()
    if args.ddp == 'torch':
        if torch.multiprocessing.get_start_method() != "spawn":
            torch.multiprocessing.set_start_method("spawn", force=True)
        dist = torch_dist
    else:
        raise ValueError(f"Unknown ddp type {args.ddp}")
 

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]

    resume_file = auto_resume_helper(config.OUTPUT)
    if resume_file:
        args.mute_repeat = True
    else:
        args.mute_repeat = False

    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    if not args.mute_repeat:
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

    main(config, args)
