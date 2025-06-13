import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
# 自动添加 EfficientViM 路径，兼容 efficientvim_m3 和 custom_fused
vim_path = os.path.abspath('./EfficientViM-main/classification')
if vim_path not in sys.path:
    sys.path.append(vim_path)

import torch
from train import get_model, MODEL_LIST
from torchinfo import summary

# 输出文件
output_file = 'all_model_architectures.md'

# 测试输入尺寸
input_shape = (1, 3, 224, 224)

with open(output_file, 'w', encoding='utf-8') as f:
    for model_name in MODEL_LIST:
        try:
            print(f'正在测试模型: {model_name}')
            model = get_model(model_name, num_classes=102)
            arch_str = f'\n==================== {model_name} ====================\n'
            f.write(arch_str)
            # summary输出到字符串
            info = summary(model, input_size=input_shape, col_names=["input_size", "output_size", "num_params"], depth=3, verbose=0)
            f.write(str(info))
            f.write('\n')
        except Exception as e:
            f.write(f'\n[ERROR] {model_name}: {e}\n')
            print(f'[ERROR] {model_name}: {e}')
print(f'所有模型架构已保存到 {output_file}')
