"""
Visual State Space (VSS) Block - Core component của Mamba-UNet
Dựa trên paper: "Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation"
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class SS2D(nn.Module):
    """
    Selective Scan 2D - Xử lý ảnh 2D với Mamba
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Mamba core
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply Mamba
        x_out = self.mamba(x_flat)  # (B, H*W, C)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x_out


class VSSBlock(nn.Module):
    """
    Visual State Space Block
    
    Architecture:
    Input -> LayerNorm -> SS2D -> DropPath -> (+) -> Output
              ↓_________________________________↑
    """
    def __init__(self, hidden_dim, drop_path=0., d_state=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
        
        # SS2D module
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state)
        
        # Stochastic depth
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if isinstance(drop_path, (list, tuple)):
            drop_path = float(drop_path[0])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # Residual connection
        shortcut = x
        
        # Layer norm: (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.ln(x)
        x = rearrange(x, 'b h w c -> b c h w')
        
        # SS2D
        x = self.ss2d(x)
        
        # Drop path + residual
        x = shortcut + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output