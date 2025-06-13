import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import os
import time
import matplotlib.pyplot as plt
from datasets import *

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
    
if __name__ == "__main__":
    model = ConvNeXt(num_classes=102, variant='small')
    summary(model, input_size=(1, 3, 224, 224))
    