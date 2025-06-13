import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from classification.datasets import train_dataset, val_dataset, test_dataset
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

class ConvNeXt(nn.Module):
    """ConvNeXt模型（基于torchvision实现）"""
    def __init__(self, num_classes=102, variant='small'):
        super(ConvNeXt, self).__init__()
        # 可选变体：tiny, small, base, large
        variants = {
            'tiny': models.convnext_tiny,
            'small': models.convnext_small,
            'base': models.convnext_base,
            'large': models.convnext_large
        }
        weights = {
            'tiny': models.ConvNeXt_Tiny_Weights.DEFAULT,
            'small': models.ConvNeXt_Small_Weights.DEFAULT,
            'base': models.ConvNeXt_Base_Weights.DEFAULT,
            'large': models.ConvNeXt_Large_Weights.DEFAULT
        }
        assert variant in variants, f"不支持的变体: {variant}"
        self.backbone = variants[variant](weights=weights[variant])
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{val_loss/len(val_loader):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 参数设置
    variant = 'small'  # 可选：tiny, small, base, large
    num_epochs = 20
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # 加载数据（直接用自定义的dataset）
    print("加载数据...")
    # train_dataset, val_dataset, test_dataset 已从 datasets.py 导入
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"创建ConvNeXt-{variant}模型...")
    model = ConvNeXt(num_classes=102, variant=variant).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练历史记录
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    results = []
    csv_path = '1111_train_results.csv'
    total_train_time = 0
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        start_time = time.time()
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_train_time = time.time() - start_time
        total_train_time += epoch_train_time
        # 记录每轮结果
        results.append([
            epoch+1, val_acc/100, 0, 0, 0, train_loss, epoch_train_time  # precision/recall/f1可后续补充
        ])
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, f'checkpoints/best_model_{variant}.pth')
            print(f'保存最佳模型 (Val Acc: {best_val_acc:.2f}%)')
    
    training_time = time.time() - start_time
    print(f'\n训练完成! 总训练时间: {training_time/3600:.2f}小时')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 保存csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'acc', 'precision', 'recall', 'f1', 'train_loss', 'train_time'])
        writer.writerows(results)
    
    # 测试最佳模型
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(f'checkpoints/best_model_{variant}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    test_acc = correct / total if total > 0 else 0
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'测试集准确率: {test_acc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1: {test_f1:.4f}')

    # 统计测试集指标后：
    result_dict = {
        'acc': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'params': count_params(model),
        'csv': os.path.abspath(csv_path),
        'train_time': total_train_time
    }
    print(result_dict)

if __name__ == '__main__':
    main()