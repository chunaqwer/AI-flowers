import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
from tqdm import tqdm
import csv
import random
import numpy as np

# 选择模型
MODEL_LIST = [
    'resnet18',
    'alexnet',
    'mobilenet_v3_small',
    'vit_b_16',
    'vgg16',
    'deit_base_patch16_ls',
    'efficientvim_m3',
    'convnext_small',
    'custom_fused',
    'resnet18_mobilenetv2_fused',
    'resnet18_mobilenetv3_fused',
]

def get_model(name, num_classes=102, **kwargs):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == 'vgg16':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == 'deit_base_patch16_ls':
        import sys
        sys.path.append('./deit-main')
        import importlib
        models_v2 = importlib.import_module('models_v2')
        model = models_v2.deit_base_patch16_LS(pretrained=False, img_size=224)
        if hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    elif name == 'efficientvim_m3':
        import sys
        sys.path.append('./EfficientViM-main/classification')
        import importlib
        models_mod = importlib.import_module('models.EfficientViM')
        model = models_mod.EfficientViM_M3(num_classes=num_classes)
    elif name == 'convnext_small':
        from ConvNeXt import ConvNeXt
        variant = kwargs.get('variant', 'small') # Get variant from kwargs, default to 'small'
        model = ConvNeXt(num_classes=num_classes, variant=variant)
    elif name == 'custom_fused':
        import importlib.util
        import os
        import sys
        sys.path.append(os.path.abspath('./EfficientViM-main/classification/models'))
        customer_path = os.path.abspath('./EfficientViM-main/classification/models/customer.py')
        spec = importlib.util.spec_from_file_location('customer', customer_path)
        customer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(customer_mod)
        model = customer_mod.FusedModel(num_classes=num_classes)
    elif name == 'resnet18_mobilenetv2_fused':
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # 去掉resnet18的fc和mobilenetv2的classifier
        resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        mobilenet_features = mobilenet.features
        class ResNet18MobileNetV2Fused(nn.Module):
            def __init__(self, resnet_features, mobilenet_features, num_classes):
                super().__init__()
                self.resnet_features = resnet_features
                self.mobilenet_features = mobilenet_features
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(512 + 1280, num_classes)
            def forward(self, x):
                r = self.resnet_features(x)  # (B, 512, 7, 7)
                m = self.mobilenet_features(x)  # (B, 1280, 7, 7)
                r = self.gap(r).flatten(1)  # (B, 512)
                m = self.gap(m).flatten(1)  # (B, 1280)
                out = torch.cat([r, m], dim=1)
                return self.classifier(out)
        model = ResNet18MobileNetV2Fused(resnet_features, mobilenet_features, num_classes)
    elif name == 'resnet18_mobilenetv3_fused':
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # 去掉resnet18的fc和mobilenetv3的classifier
        resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        mobilenet_features = mobilenet.features
        class ResNet18MobileNetV3Fused(nn.Module):
            def __init__(self, resnet_features, mobilenet_features, num_classes):
                super().__init__()
                self.resnet_features = resnet_features
                self.mobilenet_features = mobilenet_features
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(512 + 576, num_classes)
            def forward(self, x):
                r = self.resnet_features(x)  # (B, 512, 7, 7)
                m = self.mobilenet_features(x)  # (B, 576, 7, 7)
                r = self.gap(r).flatten(1)  # (B, 512)
                m = self.gap(m).flatten(1)  # (B, 576)
                out = torch.cat([r, m], dim=1)
                return self.classifier(out)
        model = ResNet18MobileNetV3Fused(resnet_features, mobilenet_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(model_name, train_loader, val_loader, test_loader, device, epochs=10, lr=1e-3, save_dir='weights'):
    model = get_model(model_name, num_classes=102).to(device)
    # 自动适配DeiT损失函数
    if model_name == 'deit_base_patch16_ls':
        import sys
        sys.path.append('./deit-main')
        import importlib
        losses = importlib.import_module('losses')
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    os.makedirs(save_dir, exist_ok=True)
    # 新增：保存每个epoch的训练日志
    csv_path = os.path.join(save_dir, f'{model_name}_train_log.csv')
    total_train_time = 0
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'time'])
        for epoch in range(epochs):
            model.train()
            start_time = time.time()
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            elapsed = time.time() - start_time
            total_train_time += elapsed
            val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f} | Time: {elapsed:.2f}s")
            writer.writerow([epoch+1, val_acc, val_precision, val_recall, val_f1, elapsed])
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pth'))
    print(f"{model_name} 总训练时间: {total_train_time:.2f}s")
    # 测试集评估
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{model_name}_best.pth')))
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    print(f"参数数量: {count_params(model)}")
    return {
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'params': count_params(model),
        'csv': os.path.abspath(csv_path),
        'train_time': total_train_time
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    import datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("可用设备:", device)
    results_dict = {}
    for model_name in MODEL_LIST:
        print(f"\n===== 正在训练模型: {model_name} =====")
        results = train_and_eval(
            model_name,
            datasets.train_loader,
            datasets.val_loader,
            datasets.test_loader,
            device,
            epochs=15,
            lr=1e-3
        )
        results_dict[model_name] = results
        print(f"{model_name} 结果: {results}")
        print(f"{model_name} 总训练时间: {results['train_time']:.2f}s")
    print("\n所有模型训练与评估完毕。汇总如下：")
    for k, v in results_dict.items():
        print(f"{k}: {v}")
