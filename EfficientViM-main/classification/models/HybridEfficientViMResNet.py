import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model as create_timm_model
from timm.layers import trunc_normal_
from torchinfo import summary
# Copied from classification/models/utils.py
class LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, groups=1, bias=False, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.norm = norm(out_dim) if norm is not None else None
        self.act = act_layer() if act_layer is not None else None
        if self.norm is not None and hasattr(self.norm, 'weight') and bn_weight_init is not None:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, groups=1, bias=False, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.norm = norm(out_dim) if norm is not None else None
        self.act = act_layer() if act_layer is not None else None
        if self.norm is not None and hasattr(self.norm, 'weight') and bn_weight_init is not None:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1, norm=LayerNorm2D, act_layer=act_layer) 
        self.fc2 = ConvLayer2D(dim, in_dim, 1, norm=None, act_layer=None, bn_weight_init=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=LayerNorm2D):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = ConvLayer2D(in_dim, out_dim, 2, 2, norm=norm_layer, act_layer=None)

    def forward(self, x):
        x = self.reduction(x)
        return x

# Copied from classification/models/EfficientViM.py
class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim = 64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim*3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3,1,1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0) 
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L= x.shape
        H = int(math.sqrt(L))
        
        BCdt = self.dw(self.BCdt_proj(x).view(batch,-1, H, H)).flatten(2)
        B,C,dt = torch.split(BCdt, [self.state_dim, self.state_dim,  self.state_dim], dim=1) 
        A = (dt + self.A.view(1,-1,1)).softmax(-1) 
        
        AB = (A * B) 
        h = x @ AB.transpose(-2,-1) 
        
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1) 
        h = self.out_proj(h * self.act(z)+ h * self.D)
        y = h @ C
        
        y = y.view(batch,-1,H,H).contiguous()
        return y, h

class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand,state_dim=state_dim)  
        self.norm = LayerNorm1D(dim)
        
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer = None)
        
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        
        self.alpha = nn.Parameter(1e-4 * torch.ones(4,dim), requires_grad=True)
        
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4,-1,1,1)
        
        x = (1-alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        x_prev = x
        # HSMSSD expects (B, D, L) input, norm is LayerNorm1D
        x_flat = x.flatten(2) # (B, D, H*W)
        x_normed = self.norm(x_flat)
        x_mixer_out, h = self.mixer(x_normed) # mixer output is (B, D, H, W)
        x = (1-alpha[1]) * x_prev + alpha[1] * x_mixer_out
        
        x = (1-alpha[2]) * x + alpha[2] * self.dwconv2(x)
        
        x = (1-alpha[3]) * x + alpha[3] * self.ffn(x)
        return x, h

class EfficientViMStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth,  mlp_ratio=4.,downsample=None, ssd_expand=1, state_dim=64):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in range(depth)])
        
        self.downsample = downsample(in_dim=in_dim, out_dim =out_dim) if downsample is not None else None

    def forward(self, x):
        h_last_block = None
        for blk in self.blocks:
            x, h_last_block = blk(x)
            
        x_out_stage = x # Output before potential downsampling
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out_stage, h_last_block

class HybridEfficientViMResNet(nn.Module):
    def __init__(self, num_classes=102,
                 resnet_model_name='resnet18', # Changed default for potentially faster iteration
                 resnet_pretrained=True,
                 # Configs for EViM stages to be inserted *after* ResNet layers 1, 2, 3
                 # Length of these lists determines how many EViM stages are inserted
                 efficientvim_dims=[128, 256, 512], # Dimension for each EViM stage
                 efficientvim_depths=[1, 1, 1],    # Depth for each EViM stage
                 efficientvim_mlp_ratio=4.,
                 efficientvim_ssd_expand=1.0,
                 efficientvim_state_dims=[49, 25, 9], # State_dim for HSMSSD in each EViM stage
                 insert_after_layer4=False # Whether to insert an EViM stage after ResNet's layer4
                ):
        super().__init__()
        self.num_classes = num_classes
        self.insert_after_layer4 = insert_after_layer4

        if not (len(efficientvim_dims) == len(efficientvim_depths) == len(efficientvim_state_dims)):
            raise ValueError("efficientvim_dims, depths, and state_dims must have the same length")

        # Load base ResNet model to access its layers
        if resnet_model_name == 'resnet18' and resnet_pretrained:
            from torchvision import models
            print("Loading ResNet18 with torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)")
            base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Manually define feature_info for torchvision resnet18 as timm's feature_info might not be directly compatible
            # These are typical values for ResNet18. Adjust if your specific use case differs.
            self.resnet_feature_info = [
                {'num_chs': 64, 'reduction': 2, 'module': 'act1'},
                {'num_chs': 64, 'reduction': 4, 'module': 'layer1'},
                {'num_chs': 128, 'reduction': 8, 'module': 'layer2'},
                {'num_chs': 256, 'reduction': 16, 'module': 'layer3'},
                {'num_chs': 512, 'reduction': 32, 'module': 'layer4'}
            ]
        else:
            print(f"Loading {resnet_model_name} with timm.create_timm_model(pretrained={resnet_pretrained})")
            base_resnet = create_timm_model(resnet_model_name, pretrained=resnet_pretrained)
            self.resnet_feature_info = create_timm_model(resnet_model_name, pretrained=False, features_only=True).feature_info

        # ResNet Stem
        if resnet_model_name == 'resnet18' and resnet_pretrained:
            # torchvision ResNet uses 'relu' as the attribute name for the first activation
            self.stem = nn.Sequential(
                base_resnet.conv1,
                base_resnet.bn1,
                base_resnet.relu, # Use relu for torchvision's ResNet
                base_resnet.maxpool
            )
        else:
            # timm ResNet uses 'act1'
            self.stem = nn.Sequential(
                base_resnet.conv1,
                base_resnet.bn1,
                base_resnet.act1,
                base_resnet.maxpool
            )

        self.resnet_layers = nn.ModuleList([
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4
        ])

        self.projections_to_evim = nn.ModuleList()
        self.interspersed_evim_stages = nn.ModuleList()
        self.projections_from_evim = nn.ModuleList()

        current_channels = self.resnet_feature_info[0]['num_chs'] # Channels after stem (input to layer1)
        # Note: timm feature_info indices are typically offset. For layer1 output, it's often index 1 for resnet50 like.
        # Let's use actual layer output channels directly.

        # Get ResNet layer output channels and next layer input channels
        # This is a bit manual and depends on ResNet architecture details
        # For standard ResNets: layerX output channels are layerX[0].conv1.in_channels * block_expansion (if applicable)
        # or more simply, use feature_info carefully.
        # feature_info[0] is after stem, input to layer1
        # feature_info[1] is after layer1, input to layer2
        # feature_info[2] is after layer2, input to layer3
        # feature_info[3] is after layer3, input to layer4
        # feature_info[4] is after layer4

        # Channels *after* ResNet layer i (and input to EViM stage i or next ResNet layer)
        # For resnet18: stem_out=64, L1_out=64, L2_out=128, L3_out=256, L4_out=512
        # For resnet50: stem_out=64, L1_out=256, L2_out=512, L3_out=1024, L4_out=2048
        
        # We will insert EViM stages *after* resnet_layers[0], resnet_layers[1], resnet_layers[2]
        # and optionally after resnet_layers[3]
        num_insertions = len(efficientvim_dims)

        # Input channels to ResNet layers (approximate, may need refinement based on specific ResNet)
        # This is simplified; actual input channels to layerX might differ if projections are complex
        # For simplicity, we assume projection_from_evim will match the *expected* input of next ResNet layer

        # Simplified: Get output channels from feature_info
        # resnet_layer_out_channels = [fi['num_chs'] for fi in self.resnet_feature_info]
        # This is for features_only=True mode. For a full model, it's more direct:
        # layer1_out_c = base_resnet.layer1[-1].bn3.num_features if hasattr(base_resnet.layer1[-1], 'bn3') else base_resnet.layer1[-1].bn2.num_features # Bottleneck vs Basic
        # This gets too complex. Let's use feature_info as a guide for channel sizes.

        # Channels entering each ResNet layer (after stem for layer1)
        # For resnet18: layer1_in=64, layer2_in=64, layer3_in=128, layer4_in=256
        # For resnet50: layer1_in=64, layer2_in=256, layer3_in=512, layer4_in=1024
        # These are the channels *before* the downsampling block in layer2,3,4 if present.

        # Channels *exiting* each ResNet layer
        # For resnet18: layer1_out=64, layer2_out=128, layer3_out=256, layer4_out=512
        # For resnet50: layer1_out=256, layer2_out=512, layer3_out=1024, layer4_out=2048
        # These are given by self.resnet_feature_info[i+1]['num_chs'] for layer i output

        # Let current_processing_channels be the output channels of the previous module (ResNet layer or EViM stage)
        # Initial channels are output of stem, input to layer1
        # This is self.resnet_feature_info[0]['num_chs']

        # Let's track channels more directly
        # After stem, before layer1: self.resnet_feature_info[0]['num_chs']
        # After layer1, before EViM1/layer2: self.resnet_feature_info[1]['num_chs']
        # After layer2, before EViM2/layer3: self.resnet_feature_info[2]['num_chs']
        # After layer3, before EViM3/layer4: self.resnet_feature_info[3]['num_chs']
        # After layer4: self.resnet_feature_info[4]['num_chs']

        # EViM stages are inserted after layer1, layer2, layer3 by default if num_insertions >= 3
        # And optionally after layer4 if insert_after_layer4 and num_insertions matches

        # This loop structure is tricky. Let's build sequentially.
        # The `current_block_out_channels` will be the output of the last processed block (ResNet layer or EViM stage)

        # Projections and EViM stages
        # Example: inserting after layer1, layer2, layer3
        # Number of EViM stages to actually build based on config length
        
        # Let's assume efficientvim_dims, _depths, _state_dims are for stages after layer1, layer2, layer3, layer4 respectively if provided.
        # We will build up to min(len(efficientvim_dims), 4 if insert_after_layer4 else 3) stages.

        # Channels from ResNet layers that will feed into EViM (or next ResNet layer if no EViM)
        # Output of layer1: self.resnet_feature_info[1]['num_chs']
        # Output of layer2: self.resnet_feature_info[2]['num_chs']
        # Output of layer3: self.resnet_feature_info[3]['num_chs']
        # Output of layer4: self.resnet_feature_info[4]['num_chs']

        # Input channels for ResNet layers (for projection from EViM)
        # Input to layer2: self.resnet_feature_info[1]['num_chs'] (output of layer1)
        # Input to layer3: self.resnet_feature_info[2]['num_chs'] (output of layer2)
        # Input to layer4: self.resnet_feature_info[3]['num_chs'] (output of layer3)

        # Max number of ResNet layers we might intersperse *after*
        max_resnet_layers_for_interspersion = 4 if self.insert_after_layer4 else 3
        actual_evim_insertions = min(num_insertions, max_resnet_layers_for_interspersion)

        for i in range(actual_evim_insertions):
            # Channels from ResNet layer i output (e.g., layer1 output for i=0)
            # This is self.resnet_feature_info[i+1]['num_chs']
            resnet_layer_output_channels = self.resnet_feature_info[i+1]['num_chs'] 
            evim_target_dim = efficientvim_dims[i]

            self.projections_to_evim.append(
                nn.Conv2d(resnet_layer_output_channels, evim_target_dim, kernel_size=1)
            )
            self.interspersed_evim_stages.append(
                EfficientViMStage(
                    in_dim=evim_target_dim, 
                    out_dim=evim_target_dim, # EViM stage doesn't change dim here
                    depth=efficientvim_depths[i],
                    mlp_ratio=efficientvim_mlp_ratio,
                    downsample=None, # No PatchMerging within these interspersed stages for now
                    ssd_expand=efficientvim_ssd_expand,
                    state_dim=efficientvim_state_dims[i]
                )
            )
            
            # Projection from EViM back to what the *next* ResNet layer expects
            # If this is the last EViM stage and no more ResNet layers follow, this projection is not needed for ResNet
            # but its output will go to the classifier.
            if i < (len(self.resnet_layers) - 1): # If there is a next ResNet layer
                # Next ResNet layer is self.resnet_layers[i+1]
                # Its input dimension is effectively the output dimension of self.resnet_layers[i]
                # which is resnet_layer_output_channels (if no downsampling block changes it drastically at start of layer)
                # More robustly: input to layer (i+1)+1 is feature_info[(i+1)+1-1] = feature_info[i+1]
                next_resnet_layer_input_channels = self.resnet_feature_info[i+1]['num_chs'] 
                self.projections_from_evim.append(
                    nn.Conv2d(evim_target_dim, next_resnet_layer_input_channels, kernel_size=1)
                )
            else: # This EViM stage is the last one before classifier (or after last ResNet layer)
                self.projections_from_evim.append(nn.Identity()) # Or handle classifier input directly

        # Determine final channels for classifier
        if actual_evim_insertions > 0:
            # If the last EViM stage was after resnet_layers[actual_evim_insertions-1]
            # its output dim is efficientvim_dims[actual_evim_insertions-1]
            final_channels = efficientvim_dims[actual_evim_insertions-1]
            # If this EViM stage was *not* the one after the *very last* ResNet layer (layer4),
            # and subsequent ResNet layers were processed, then final_channels is output of last ResNet layer.
            # This logic needs to be tied to the forward pass.
            # For now, assume if EViMs are inserted, the last EViM's output dim is key.
            # If the last EViM stage is followed by more ResNet layers, then it's the output of the last ResNet layer.
            
            # Let's trace the output channel dimension in the forward pass logic to set final_channels correctly.
            # For now, placeholder based on last EViM or last ResNet layer.
            if actual_evim_insertions == 4 and self.insert_after_layer4 : # EViM after layer4
                 final_channels = efficientvim_dims[3]
            elif actual_evim_insertions > 0 and actual_evim_insertions < 4 : # EViM inserted, but layer4 is still processed after last EViM
                 final_channels = self.resnet_feature_info[4]['num_chs'] # Output of layer4
            elif actual_evim_insertions == 0 : # No EViM stages, just ResNet
                 final_channels = self.resnet_feature_info[4]['num_chs'] # Output of layer4
            else: # Default to last EViM dim if logic above is not exhaustive
                 final_channels = efficientvim_dims[actual_evim_insertions-1] if actual_evim_insertions > 0 else self.resnet_feature_info[4]['num_chs']


        else: # No EViM stages inserted, plain ResNet
            final_channels = self.resnet_feature_info[4]['num_chs'] # Output of ResNet layer4

        self.final_norm = LayerNorm2D(final_channels) 
        self.final_avgpool = nn.AdaptiveAvgPool2d(1)
        self.final_head = nn.Linear(final_channels, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm1D, LayerNorm2D)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)

        # Iterate through ResNet layers and interspersed EViM stages
        # Max ResNet layers to process before considering EViM insertion logic for *that* layer
        max_r_layers_for_interspersion = 4 if self.insert_after_layer4 else 3
        num_evim_configured = len(self.interspersed_evim_stages)

        for i in range(len(self.resnet_layers)):
            x = self.resnet_layers[i](x)
            
            # Check if an EViM stage should be inserted after this ResNet layer
            if i < num_evim_configured and i < max_r_layers_for_interspersion:
                # (e.g., i=0 for after layer1, i=1 for after layer2, etc.)
                x_proj_to_evim = self.projections_to_evim[i](x)
                x_evim, _, _ = self.interspersed_evim_stages[i](x_proj_to_evim)
                x = self.projections_from_evim[i](x_evim)
            # If i >= num_evim_configured, no more EViM stages for subsequent ResNet layers
            # If i >= max_r_layers_for_interspersion, no EViM after this (e.g. after layer4 if not insert_after_layer4)
        
        # Final classification head
        x = self.final_norm(x)
        x = self.final_avgpool(x)
        x = x.flatten(1)
        output = self.final_head(x)
        return output

# Example of how to register if using timm's create_model later, or just for direct instantiation
# @register_model # from timm.models import register_model
# def hybrid_efficientvim_resnet50_s0(pretrained=False, **kwargs):
#     model = HybridEfficientViMResNet(
#         resnet_model_name='resnet50', resnet_out_indices=(0,),
#         projection_out_dim=128,
#         efficientvim_embed_dims=[128, 256], 
#         efficientvim_depths=[1, 1],
#         efficientvim_state_dims=[49,25],
#         **kwargs
#     )
#     return model

# @register_model
# def hybrid_efficientvim_resnet50_s3(pretrained=False, **kwargs):
#     # Uses ResNet50 layer4 (index 3) output
#     # Input to EfficientViM stages will have H,W = input_size / 32
#     # e.g., 224/32 = 7. So, L=49 for the first EViM stage if no further downsampling in projection.
#     model = HybridEfficientViMResNet(
#         resnet_model_name='resnet50', resnet_out_indices=(3,),
#         projection_out_dim=256, # ResNet50 layer4 has 2048 channels, projected to 256
#         efficientvim_embed_dims=[384, 512], # Example dimensions for EViM stages
#         efficientvim_depths=[2, 2],
#         efficientvim_state_dims=[49, 25], # Example state_dims, might need tuning based on actual L
#         **kwargs
#     )
#     return model

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import traceback

    # Configuration for the new interspersed hybrid model
    # Example: Insert EViM stages after ResNet layer1 and layer2
    hybrid_model_config = {
        'num_classes': 102,
        'resnet_model_name': 'resnet18', # Using resnet18 for example
        'resnet_pretrained': False, # Set to True for pretrained weights
        'efficientvim_dims': [128, 256],       # Dim for EViM after layer1, EViM after layer2
        'efficientvim_depths': [1, 1],          # Depth for these EViM stages
        'efficientvim_mlp_ratio': 4.0,
        'efficientvim_ssd_expand': 1.0,
        'efficientvim_state_dims': [ (224//4)**2, (224//8)**2 ], # L for HxW after stem (56x56), after L1 (56x56 for resnet18)
                                                                # L for HxW after L1 (56x56), after L2 (28x28 for resnet18)
                                                                # state_dims should match L = H*W of the input to EViM stage
        'insert_after_layer4': False # Don't insert EViM after layer4 in this example
    }

    # Adjust state_dims based on feature map sizes for resnet18 and 224x224 input:
    # Input: 224x224
    # After stem (conv1, maxpool): 224/4 = 56x56. (L=3136)
    # After layer1 (no spatial downsampling in layer1 of resnet18): 56x56. (L=3136)
    # After layer2 (downsamples by 2): 28x28. (L=784)
    # After layer3 (downsamples by 2): 14x14. (L=196)
    # After layer4 (downsamples by 2): 7x7. (L=49)

    # If EViM is after layer1 (input 56x56), state_dim[0] should be 56*56
    # If EViM is after layer2 (input 28x28), state_dim[1] should be 28*28
    # If EViM is after layer3 (input 14x14), state_dim[2] should be 14*14
    # If EViM is after layer4 (input 7x7), state_dim[3] should be 7*7
    
    # Example for inserting after L1, L2, L3 with resnet18
    if hybrid_model_config['resnet_model_name'] == 'resnet18':
        sds = []
        if len(hybrid_model_config['efficientvim_dims']) > 0: sds.append((224//4)**2) # After L1
        if len(hybrid_model_config['efficientvim_dims']) > 1: sds.append((224//8)**2) # After L2
        if len(hybrid_model_config['efficientvim_dims']) > 2: sds.append((224//16)**2)# After L3
        if len(hybrid_model_config['efficientvim_dims']) > 3 and hybrid_model_config['insert_after_layer4']: 
            sds.append((224//32)**2) # After L4
        hybrid_model_config['efficientvim_state_dims'] = sds[:len(hybrid_model_config['efficientvim_dims'])]

    print("Using config:", hybrid_model_config)

    model = HybridEfficientViMResNet(**hybrid_model_config).to(device)
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device) # Changed batch size to 1
    
    try:
        print(f"Testing model with input shape: {dummy_input.shape}")
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Model structure:")
        # print(model) # Can be very long
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params/1e6:.2f} M")

    except Exception as e:
        print("Error during model test:", e)
        traceback.print_exc()

# Example of how to register if using timm's create_model later, or just for direct instantiation
# @register_model # from timm.models import register_model
# def hybrid_efficientvim_resnet50_s0(pretrained=False, **kwargs):
#     model = HybridEfficientViMResNet(
#         resnet_model_name='resnet50', resnet_out_indices=(0,),
#         projection_out_dim=128,
#         efficientvim_embed_dims=[128, 256], 
#         efficientvim_depths=[1, 1],
#         efficientvim_state_dims=[49,25],
#         **kwargs
#     )
#     return model

# @register_model
# def hybrid_efficientvim_resnet50_s3(pretrained=False, **kwargs):
#     # Uses ResNet50 layer4 (index 3) output
#     # Input to EfficientViM stages will have H,W = input_size / 32
#     # e.g., 224/32 = 7. So, L=49 for the first EViM stage if no further downsampling in projection.
#     model = HybridEfficientViMResNet(
#         resnet_model_name='resnet50', resnet_out_indices=(3,),
#         projection_out_dim=256, # ResNet50 layer4 has 2048 channels, projected to 256
#         efficientvim_embed_dims=[384, 512], # Example dimensions for EViM stages
#         efficientvim_depths=[2, 2],
#         efficientvim_state_dims=[49, 25], # Example state_dims, might need tuning based on actual L
#         **kwargs
#     )
#     return model

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration for the interspersed hybrid model (example)
    # This should match the parameters used in the traceback if testing that specific config
    hybrid_model_config = {
        'num_classes': 102,
        'resnet_model_name': 'resnet18', # From user's traceback
        'resnet_pretrained': False,      # From user's traceback
        'efficientvim_dims': [128, 256], # From user's traceback
        'efficientvim_depths': [1, 1],   # From user's traceback
        'efficientvim_mlp_ratio': 4.0,   # From user's traceback
        'efficientvim_ssd_expand': 1.0,  # From user's traceback
        'efficientvim_state_dims': [3136, 784], # From user's traceback (224/4=56 -> 56*56=3136, 224/8=28 -> 28*28=784)
        'insert_after_layer4': False     # From user's traceback
    }

    # Example: Dynamically calculate state_dims based on input size and ResNet config
    # This is a more robust way if you change resnet_model_name or input_size
    input_size = 224
    resnet_model_name_for_config = hybrid_model_config['resnet_model_name']
    
    # Get reduction factors for ResNet layers (approximate)
    # These depend on the ResNet architecture (strides in layers)
    # For standard ResNets: layer1=4, layer2=8, layer3=16, layer4=32 (after stem's initial 2x downsample)
    reductions = {'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32}
    
    # Determine which layers will have EViM stages after them
    # Default is after layer1, layer2, layer3
    insertion_points_default = ['layer1', 'layer2', 'layer3']
    if hybrid_model_config.get('insert_after_layer4', False):
        insertion_points_default.append('layer4')

    # Ensure efficientvim_dims/depths match the number of insertion points
    num_evim_stages = len(hybrid_model_config['efficientvim_dims'])
    actual_insertion_points = insertion_points_default[:num_evim_stages]

    calculated_state_dims = []
    for layer_name in actual_insertion_points:
        reduction = reductions[layer_name]
        feature_map_size = input_size // reduction
        calculated_state_dims.append(feature_map_size * feature_map_size)
    
    # If you want to use the dynamically calculated state_dims:
    # hybrid_model_config['efficientvim_state_dims'] = calculated_state_dims
    # print(f"Using calculated state_dims: {calculated_state_dims}")
    # Otherwise, it will use the hardcoded ones from the user's traceback for this specific test.


    model = HybridEfficientViMResNet(**hybrid_model_config).to(device)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 3, 224, 224).to(device) # Batch size 2, 3 channels, 224x224 image
    summary(model, [1,3,224,224])
    # Perform a forward pass
    try:
        print(f"Testing model with input shape: {dummy_input.shape}")
        output = model(dummy_input)
        print(f"Output shape: {output.shape}") # Expected: (2, num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    except Exception as e:
        print(f"Error during model test: {e}")
        import traceback
        traceback.print_exc()