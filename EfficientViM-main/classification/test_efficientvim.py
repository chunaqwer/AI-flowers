import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from classification.models.EfficientViM import EfficientViM_M4
from torchinfo import summary

if __name__ == "__main__":
    # 创建模型
    model = EfficientViM_M4(num_classes=1000)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 测试输入
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape if isinstance(out, torch.Tensor) else [o.shape for o in out]}")
    # 打印模型结构
    summary(model, (1, 3, 224, 224), device=device.type)
