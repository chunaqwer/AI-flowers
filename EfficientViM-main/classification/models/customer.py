import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

from models.EfficientViM import EfficientViM_M3

class ResNetBackbone(nn.Module):
    """ResNet模型（基于torchvision实现），移除分类头以输出特征图"""

    def __init__(self, variant='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        variants = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        default_weights = {
            'resnet18': models.ResNet18_Weights.DEFAULT,
            'resnet34': models.ResNet34_Weights.DEFAULT,
            'resnet50': models.ResNet50_Weights.DEFAULT,
            'resnet101': models.ResNet101_Weights.DEFAULT,
            'resnet152': models.ResNet152_Weights.DEFAULT
        }

        assert variant in variants, f"不支持的ResNet变体: {variant}"

        current_weights = default_weights[variant] if pretrained else None
        if pretrained and current_weights:
            print(f"Loading {variant} with pretrained weights: {current_weights}")
        elif pretrained and not current_weights:
            print(f"Warning: Pretrained weights requested for {variant}, but no default weights found/specified. Model will be initialized randomly.")
        else:
            print(f"Initializing {variant} without pretrained weights.")

        resnet = variants[variant](weights=current_weights)
        
        # 移除ResNet的avgpool和fc层，保留到layer4的输出
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 获取输出特征数 (layer4的输出通道数)
        # 对于ResNet50, ResNet101, ResNet152, layer4输出2048个特征
        # 对于ResNet18, ResNet34, layer4输出512个特征
        if variant in ['resnet18', 'resnet34']:
            self.out_features = 512
        else:
            self.out_features = 2048

    def forward(self, x):
        return self.features(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FusedModel(nn.Module):
    def __init__(self, num_classes=102, resnet_variant='resnet50', resnet_pretrained=True, efficientvim_pretrained=True):
        super().__init__()
        # Load ResNet model as the primary backbone
        self.resnet_backbone = ResNetBackbone(variant=resnet_variant, pretrained=resnet_pretrained)

        # ResNet output features (e.g., 2048 for ResNet50 from layer4)

        # Load EfficientViM_M3 model
        # We will use parts of EfficientViM to process features from ConvNeXt.
        # We need to adapt EfficientViM to take features from ConvNeXt instead of raw images.
        self.efficientvim_core = EfficientViM_M3(num_classes=1000) # Initialize for potential weight loading
        if efficientvim_pretrained:
            self.load_efficientvim_pretrained_weights(self.efficientvim_core)

        # EfficientViM_M3 first stage (patch_embed + stage1) expects 3 input channels and outputs 'embed_dim[0]' (e.g., 128 for M1, 224 for M3)
        # We need a projection layer if ResNet output features don't match EfficientViM's expected input dimension for its later stages.
        # Let's assume we'll use EfficientViM's stages 2 and 3, and its head.
        # EfficientViM_M3 embed_dim = [224, 320, 512] (for M3, these are output dims of stage1, stage2, stage3 respectively)
        # The input to stage2 is the output of stage1, which is embed_dim[0] = 224 for M3.
        # The patch_embed.conv[-1].conv.out_channels gives the output channels of the stem, which is the input to stage1.
        # For EfficientViM_M3, this is 128. So we project ResNet features to 128 channels.
        resnet_output_channels = self.resnet_backbone.out_features 
        efficientvim_stem_out_dim = self.efficientvim_core.patch_embed.conv[-1].conv.out_channels # This is input to EfficientViM stage1
        # We will feed the projected features into EfficientViM's stage1 (stages[0])
        # So, the projection should match the input dimension of stage1, which is efficientvim_stem_out_dim
        self.projection = nn.Conv2d(resnet_output_channels, efficientvim_stem_out_dim, kernel_size=1)
        
        # We will use all stages of EfficientViM after the projection
        self.efficientvim_stage1 = self.efficientvim_core.stages[0]
        self.efficientvim_stage2 = self.efficientvim_core.stages[1]
        self.efficientvim_stage3 = self.efficientvim_core.stages[2]
        self.efficientvim_norm = self.efficientvim_core.norm[-1] # Norm after the last stage used

        # New classifier head
        # The input dimension to the classifier will be the output of efficientvim_norm (after stage3)
        # For EfficientViM_M3, the output dimension of stage3 (stages[2]) is required.
        # This can be obtained from the 'dim' attribute of the last block in the last stage used.
        efficientvim_classifier_input_dim = self.efficientvim_core.stages[2].blocks[-1].dim
        self.classifier = nn.Linear(efficientvim_classifier_input_dim, num_classes)
        
        # Use later stages of EfficientViM
        self.efficientvim_stage2 = self.efficientvim_core.stages[1] # stage2 is at index 1
        self.efficientvim_stage3 = self.efficientvim_core.stages[2] # stage3 is at index 2
        self.efficientvim_norm = self.efficientvim_core.norm[-1] # Assuming we use the last norm for the final stage output

        # New classifier head
        # The input dimension to the classifier will be the output of efficientvim_norm, which is embed_dim[-1]
        # From EfficientViM_M3: embed_dim = [224, 224, 448, 448] (for M3, this is embed_dim for stages 0,1,2,3)
        # Our efficientvim_core (M3) has embed_dim=[224, 224, 448]. So after stage3 (index 2), output dim is 448.
        # The norm[-1] corresponds to embed_dim[2] which is 448 for M3.
        efficientvim_classifier_input_dim = self.efficientvim_core.stages[2].blocks[-1].dim # Get dim from last block of stage3
        self.classifier = nn.Linear(efficientvim_classifier_input_dim, num_classes)

        # Parts of efficientvim_core like patch_embed and stages[0] (stage1) are not used in the forward pass.
        # The original heads are also not used.
        pass # No deletion needed if not used in forward pass

    def load_efficientvim_pretrained_weights(self, model_to_load):
        default_path = 'models/EfficientViM_M3_e450.pth'
        if os.path.exists(default_path):
            try:
                checkpoint = torch.load(default_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict_ema', checkpoint.get('model', checkpoint))
                
                model_dict = model_to_load.state_dict()
                # Filter out keys that don't exist in the current model_to_load or have shape mismatches
                # This is important because we are using parts of EfficientViM
                pretrained_dict = {}
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        pretrained_dict[k] = v
                    # elif k.startswith('patch_embed.') or k.startswith('stage1.'):
                    #     print(f"Skipping {k} from pretrained EfficientViM as it's not used in FusedModel")
                    # elif k.startswith('head.'):
                    #     print(f"Skipping {k} from pretrained EfficientViM head as it's replaced")
                
                model_dict.update(pretrained_dict)
                missing_keys, unexpected_keys = model_to_load.load_state_dict(pretrained_dict, strict=False)
                # `strict=False` is important here as we are loading parts of the model.
                print(f"EfficientViM parts: Loaded pretrained weights from {default_path}")
                if missing_keys:
                    print(f"EfficientViM parts: Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"EfficientViM parts: Unexpected keys (might be ok if parts are intentionally unused): {unexpected_keys}")
            except Exception as e:
                print(f"Error loading EfficientViM pretrained weights for parts from {default_path}: {e}")
        else:
            print(f"EfficientViM pretrained weights not found at {default_path}. Relevant parts train from scratch.")

    def forward(self, x):
        # Get features from ResNet backbone
        resnet_features = self.resnet_backbone(x) 
        # resnet_features shape: (batch_size, resnet_output_channels, H/32, W/32)

        # Project ResNet features to match input dimension for EfficientViM's first stage
        projected_features = self.projection(resnet_features)

        # Pass through EfficientViM's stages
        # EfficientViM stages expect input like (B, C, H, W)
        # The `EfficientViMStage` in the original code takes (B,C,H,W) and returns (B,C',H',W')
        
        # Pass through EfficientViM stage 1, 2, 3
        # Note: The EfficientViM stages in the original implementation return (output, H, W)
        # We only need the feature tensor 'output' for the next stage or norm.
        x, _, _ = self.efficientvim_stage1(projected_features)
        x, _, _ = self.efficientvim_stage2(x)
        x, _, _ = self.efficientvim_stage3(x)
        
        # Apply the specific norm for the output of the last stage used (stage3)
        x = self.efficientvim_norm(x) # norm[-1] is LayerNorm2D, expects (B, C, H, W)
        x = x.mean([-2, -1]) # Global average pooling

        output = self.classifier(x)
        return output

if __name__ == '__main__':
    # Test the fused model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Update FusedModel instantiation to use ResNet parameters
    fused_model = FusedModel(num_classes=102, resnet_variant='resnet50', resnet_pretrained=True, efficientvim_pretrained=True).to(device)
    
    # It's good practice to load official pretrained weights for EfficientViM if available and path is known
    # For example, if 'EfficientViM_M3_e450.pth' is the correct weights file:
    try:
        # Assuming the script is run from EfficientViM-main directory or models/ is in PYTHONPATH
        checkpoint_path = 'EfficientViM_M3_e450.pth' # if run from models directory
        # If run from EfficientViM-main, path should be 'models/EfficientViM_M3_e450.pth'
        # For robustness, let's try to construct path relative to this file's directory
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_dir, 'EfficientViM_M3_e450.pth')

        if not os.path.exists(checkpoint_path):
             # Try path assuming run from project root
             project_root_checkpoint_path = os.path.join(os.path.dirname(current_dir), 'models', 'EfficientViM_M3_e450.pth')
             if os.path.exists(project_root_checkpoint_path):
                 checkpoint_path = project_root_checkpoint_path
             else:
                 # Fallback if not found in typical locations relative to script
                 # This was the original assumption if script is in models/ and weights file is there too
                 checkpoint_path = 'EfficientViM_M3_e450.pth' 

        print(f"Attempting to load checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint: # This block in __main__ seems to be for a standalone efficientvim, not the fused_model's core
            # The FusedModel already calls self.load_efficientvim_pretrained_weights(self.efficientvim_core)
            # This section in __main__ might be redundant or for a different purpose.
            # For now, let's comment it out to avoid confusion, as FusedModel handles its own weight loading.
            print("Skipping redundant EfficientViM weight loading in __main__ as FusedModel handles it.")
            # model_dict = fused_model.efficientvim_core.state_dict()
            # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and model_dict[k].shape == v.shape}
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'head' not in k and 'classifier' not in k}
            # model_dict.update(pretrained_dict)
            # missing_keys, unexpected_keys = fused_model.efficientvim_core.load_state_dict(model_dict, strict=False)
            print(f"Loaded EfficientViM_M3 pretrained weights from {checkpoint_path}.")
            # if missing_keys:
            #     print(f"Missing keys: {missing_keys}")
            # if unexpected_keys:
            #     print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Could not find 'model' key in checkpoint at {checkpoint_path}.")
    except FileNotFoundError:
        print(f"EfficientViM_M3 pretrained weights file not found at {checkpoint_path} or other attempted paths. Timm's default might be used if pretrained=True was effective.")
    except Exception as e:
        print(f"Error loading EfficientViM_M3 pretrained weights: {e}")

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print("\nFusedModel Summary (ResNet50 + EfficientViM_M3 Stages 1,2,3):")
    try:
        print("Attempting manual forward pass before torchinfo...")
        dummy_input_test = torch.randn(1, 3, 224, 224).to(device)
        fused_model.eval() # Ensure model is in eval mode for testing
        with torch.no_grad():
            output_test = fused_model(dummy_input_test)
        print(f"Manual forward pass successful. Output shape: {output_test.shape}")
    except Exception as e:
        print(f"Error during manual forward pass: {e}")
        import traceback
        traceback.print_exc()

    summary(fused_model, input_size=(1, 3, 224, 224), device=device.type)

    # Test forward pass
    try:
        output = fused_model(dummy_input)
        print("\nOutput shape:", output.shape)
    except Exception as e:
        print(f"\nError during forward pass: {e}")

    # Test individual backbones to ensure they are working as expected before fusion
    print("\nResNet Backbone Test:")
    resnet_test = ResNetBackbone(variant='resnet50', pretrained=True).to(device)
    summary(resnet_test, input_size=(1, 3, 224, 224))
    resnet_out = resnet_test(dummy_input)
    print("ResNet output shape:", resnet_out.shape)

    print("\nEfficientViM_M3 Backbone Test (with num_classes=102 to match intended use):")
    efficientvim_test = EfficientViM_M3(pretrained=True, num_classes=102).to(device)
    summary(efficientvim_test, input_size=(1, 3, 224, 224))
    efficientvim_out = efficientvim_test(dummy_input)
    print("EfficientViM_M3 output shape:", efficientvim_out.shape)