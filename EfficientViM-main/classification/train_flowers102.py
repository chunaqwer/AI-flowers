import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, optim
from classification.models.EfficientViM import EfficientViM_M3
from tqdm import tqdm
from datasets import train_loader, val_loader
import time
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

NUM_CLASSES = 102
EPOCHS = 20  # 可根据需要调整
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EfficientViM_M3(num_classes=NUM_CLASSES).to(DEVICE)
# 自动拼接权重文件路径，确保无论在哪运行都能找到
pretrained_path = os.path.join(os.path.dirname(__file__), 'EfficientViM_M3_e450.pth')
if not os.path.exists(pretrained_path):
    # 兼容 classification/ 目录下
    pretrained_path = os.path.join(os.path.dirname(__file__), 'classification', 'EfficientViM_M3_e450.pth')
if os.path.exists(pretrained_path):
    # 兼容 PyTorch 2.6+ 权重安全加载
    try:
        state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        # 兼容旧版 torch.load
        state_dict = torch.load(pretrained_path, map_location=DEVICE)
    except Exception as e:
        print(f"权重文件加载失败: {e}\n尝试使用 weights_only=False 再次加载（仅信任官方权重时可用）...")
        state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    # 只加载backbone参数，跳过分类头
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('heads')}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded pretrained backbone from {pretrained_path}. Missing: {missing}, Unexpected: {unexpected}")
else:
    print(f"Warning: Pretrained weights not found at {pretrained_path}, training from scratch!")

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

results = []
csv_path = 'efficientvim_flowers102_results.csv'
total_train_time = 0
best_val_acc = 0.0
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    for imgs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_train_time = time.time() - start_time
    total_train_time += epoch_train_time
    print(f'Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}')

    # 验证
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Val'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'Epoch {epoch+1} Val Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
    results.append([epoch+1, acc, precision, recall, f1, total_loss/len(train_loader), epoch_train_time])
    # 保存最佳权重
    if acc > best_val_acc:
        best_val_acc = acc
        best_state_dict = model.state_dict()

# 保存csv
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'acc', 'precision', 'recall', 'f1', 'train_loss', 'train_time'])
    writer.writerows(results)

# 保存最佳模型
if best_state_dict is not None:
    torch.save(best_state_dict, 'efficientvim_flowers102_best.pth')
    print('已保存验证集最佳模型权重为 efficientvim_flowers102_best.pth')

# 统计测试集指标
from datasets import test_loader
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc='Test'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
test_acc = correct / total
test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'Test Acc: {test_acc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1: {test_f1:.4f}')

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
# 可选：保存最终模型
# torch.save(model.state_dict(), 'efficientvim_flowers102.pth')
