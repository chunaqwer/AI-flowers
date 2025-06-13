import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, optim
# 修改导入的模型
from classification.models.HybridEfficientViMResNet import HybridEfficientViMResNet 
from tqdm import tqdm
from datasets import train_loader, val_loader # 假设datasets.py中定义了适用于flowers102的加载器
import time
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

NUM_CLASSES = 102
EPOCHS = 20  # 可以根据需要调整
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化新的混合模型 (穿插式)
# 参数根据新的 HybridEfficientViMResNet.__init__ 进行调整

# 假设输入图像大小为 224x224
INPUT_IMAGE_SIZE = 224

# 配置将插入的EfficientViM stage
# 示例：在ResNet的layer1, layer2之后插入EfficientViM stage
resnet_model_name_for_config = 'resnet18' # 与下面模型实例化中的resnet_model_name一致
config_efficientvim_dims = [128, 256]  # 每个穿插的EfficientViM stage的维度
config_efficientvim_depths = [1, 1]     # 每个穿插的EfficientViM stage的深度
config_insert_after_layer4 = False    # 是否在layer4后也插入 (如果为True, 上面两个list应有相应数量的元素)

# 根据ResNet类型和插入位置动态计算efficientvim_state_dims
# ResNet18/34/50+ 在不同层输出的特征图大小不同
# stem (conv1, maxpool) -> /4
# layer1 -> /4 (resnet18/34), /4 (resnet50+)
# layer2 -> /8 (resnet18/34), /8 (resnet50+)
# layer3 -> /16 (resnet18/34), /16 (resnet50+)
# layer4 -> /32 (resnet18/34), /32 (resnet50+)

# state_dims 计算逻辑 (L = H*W)
# 假设 EfficientViM stage 不改变空间维度 (downsample=None)
calculated_state_dims = []
if resnet_model_name_for_config.startswith('resnet18') or resnet_model_name_for_config.startswith('resnet34'):
    # After layer1 (input to EViM_stage_0): INPUT_IMAGE_SIZE / 4
    if len(config_efficientvim_dims) > 0: calculated_state_dims.append((INPUT_IMAGE_SIZE // 4)**2)
    # After layer2 (input to EViM_stage_1): INPUT_IMAGE_SIZE / 8
    if len(config_efficientvim_dims) > 1: calculated_state_dims.append((INPUT_IMAGE_SIZE // 8)**2)
    # After layer3 (input to EViM_stage_2): INPUT_IMAGE_SIZE / 16
    if len(config_efficientvim_dims) > 2: calculated_state_dims.append((INPUT_IMAGE_SIZE // 16)**2)
    # After layer4 (input to EViM_stage_3, if config_insert_after_layer4 is True): INPUT_IMAGE_SIZE / 32
    if len(config_efficientvim_dims) > 3 and config_insert_after_layer4: 
        calculated_state_dims.append((INPUT_IMAGE_SIZE // 32)**2)
elif resnet_model_name_for_config.startswith('resnet50') or resnet_model_name_for_config.startswith('resnet101') or resnet_model_name_for_config.startswith('resnet152'):
    # After layer1: INPUT_IMAGE_SIZE / 4
    if len(config_efficientvim_dims) > 0: calculated_state_dims.append((INPUT_IMAGE_SIZE // 4)**2)
    # After layer2: INPUT_IMAGE_SIZE / 8
    if len(config_efficientvim_dims) > 1: calculated_state_dims.append((INPUT_IMAGE_SIZE // 8)**2)
    # After layer3: INPUT_IMAGE_SIZE / 16
    if len(config_efficientvim_dims) > 2: calculated_state_dims.append((INPUT_IMAGE_SIZE // 16)**2)
    # After layer4: INPUT_IMAGE_SIZE / 32
    if len(config_efficientvim_dims) > 3 and config_insert_after_layer4: 
        calculated_state_dims.append((INPUT_IMAGE_SIZE // 32)**2)
else:
    # 默认或未知ResNet，使用通用但可能不准确的state_dims
    print(f"Warning: Unknown ResNet variant '{resnet_model_name_for_config}' for state_dim calculation. Using placeholder values.")
    if len(config_efficientvim_dims) > 0: calculated_state_dims.append(56*56) # Placeholder for /4
    if len(config_efficientvim_dims) > 1: calculated_state_dims.append(28*28) # Placeholder for /8
    if len(config_efficientvim_dims) > 2: calculated_state_dims.append(14*14) # Placeholder for /16
    if len(config_efficientvim_dims) > 3 and config_insert_after_layer4: calculated_state_dims.append(7*7) # Placeholder for /32

# 确保 calculated_state_dims 的长度与 config_efficientvim_dims 一致
final_efficientvim_state_dims = calculated_state_dims[:len(config_efficientvim_dims)]

model = HybridEfficientViMResNet(
    num_classes=NUM_CLASSES,
    resnet_model_name=resnet_model_name_for_config, # e.g., 'resnet18', 'resnet50'
    resnet_pretrained=True,
    efficientvim_dims=config_efficientvim_dims, 
    efficientvim_depths=config_efficientvim_depths,
    efficientvim_mlp_ratio=4.0, # Standard MLP ratio for transformers
    efficientvim_ssd_expand=1.0, # Expansion ratio for SSD block
    efficientvim_state_dims=final_efficientvim_state_dims,
    insert_after_layer4=config_insert_after_layer4
).to(DEVICE)

print(f"Using model: HybridEfficientViMResNet (Interspersed) with {resnet_model_name_for_config}")
print(f"EViM Dims: {config_efficientvim_dims}, Depths: {config_efficientvim_depths}, State Dims: {final_efficientvim_state_dims}, Insert after L4: {config_insert_after_layer4}")

print(f"Using model: HybridEfficientViMResNet")

# 混合模型的预训练权重加载逻辑可能需要调整
# 例如，可以分别加载ResNet部分的预训练权重，EfficientViM部分可以从头训练或加载其自身的预训练（如果适用）
# 这里简化为不加载特定于整个混合模型的预训练权重，依赖于ResNet部分的预训练

optimizer = optim.AdamW(model.parameters(), lr=1e-2) # 学习率可能需要调整
criterion = nn.CrossEntropyLoss()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {count_params(model)}")

results = []
csv_path = 'hybrid_efficientvim_resnet_flowers102_results.csv'
total_train_time = 0
best_val_acc = 0.0
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    for imgs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{EPOCHS}'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0] # 确保取到主要的分类输出
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_train_time = time.time() - start_time
    total_train_time += epoch_train_time
    print(f'Epoch {epoch+1}/{EPOCHS} Train Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_train_time:.2f}s')

    # 验证
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{EPOCHS}'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = correct / total if total > 0 else 0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'Epoch {epoch+1}/{EPOCHS} Val Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
    results.append([epoch+1, acc, precision, recall, f1, total_loss/len(train_loader), epoch_train_time])
    
    if acc > best_val_acc:
        best_val_acc = acc
        best_state_dict = model.state_dict()
        print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
        torch.save(best_state_dict, 'hybrid_efficientvim_resnet_flowers102_best.pth')

# 保存CSV结果
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'train_loss', 'epoch_train_time'])
    writer.writerows(results)
print(f"Results saved to {csv_path}")

# 加载最佳模型进行测试
if best_state_dict is not None:
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('hybrid_efficientvim_resnet_flowers102_best.pth'))
else:
    print("No best model found, testing with the last model state.")

# 统计测试集指标 (假设test_loader也已在datasets.py中定义)
from datasets import test_loader 
model.eval()
correct = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc='Testing'):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

test_acc = correct / total if total > 0 else 0
test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'Test Acc: {test_acc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1: {test_f1:.4f}')

result_summary = {
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'params': count_params(model),
    'csv_results_path': os.path.abspath(csv_path),
    'total_train_time_seconds': total_train_time,
    'best_model_path': os.path.abspath('hybrid_efficientvim_resnet_flowers102_best.pth') if best_state_dict else None
}
print("Training and testing finished.")
print("Summary:", result_summary)

# 可选：保存最终模型状态
# torch.save(model.state_dict(), 'hybrid_efficientvim_resnet_flowers102_final.pth')