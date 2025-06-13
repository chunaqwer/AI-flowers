import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
import random
from torchvision.transforms import functional as F
from collections import Counter
import os

# 数据增强
class AddGaussianNoise(object):  # 添加高斯噪声
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载全部数据集（train+val+test）
all_train = datasets.Flowers102(
    root='./data',
    split='train',
    download=True,
    transform=transform
)
all_val = datasets.Flowers102(
    root='./data',
    split='val',
    download=True,
    transform=transform
)
all_test = datasets.Flowers102(
    root='./data',
    split='test',
    download=True,
    transform=transform
)
full_dataset = ConcatDataset([all_train, all_val, all_test])

# 固定随机种子，保证每次划分一致
seed = 42
random.seed(seed)
torch.manual_seed(seed)

full_len = len(full_dataset)
train_len = int(0.8 * full_len)
val_len = int(0.1 * full_len)
test_len = full_len - train_len - val_len
indices = list(range(full_len))
random.shuffle(indices)
train_indices = indices[:train_len]
val_indices = indices[train_len:train_len+val_len]
test_indices = indices[train_len+val_len:]
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f"训练集数量: {len(train_dataset)}")
print(f"验证集数量: {len(val_dataset)}")
print(f"测试集数量: {len(test_dataset)}")


# # 查看按8:1:1分割后的各数据集的类别数
# def get_class_count(dataset):
#     labels = []
#     for item in dataset:
#         # Flowers102返回(image, label)
#         labels.append(item[1])
#     return len(set(labels)), Counter(labels)

# train_class_num, train_class_counter = get_class_count(train_dataset)
# val_class_num, val_class_counter = get_class_count(val_dataset)
# test_class_num, test_class_counter = get_class_count(test_dataset)

# print(f"训练集类别数: {train_class_num}")
# print(f"验证集类别数: {val_class_num}")
# print(f"测试集类别数: {test_class_num}")


# 如需详细类别分布可打印Counter
# print('训练集类别分布:', train_class_counter)
# print('验证集类别分布:', val_class_counter)
# print('测试集类别分布:', test_class_counter)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集加载器batch数: {len(train_loader)}")
print(f"验证集加载器batch数: {len(val_loader)}")
print(f"测试集加载器batch数: {len(test_loader)}")