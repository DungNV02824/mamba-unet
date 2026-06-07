"""
Ablation models for Mamba-UNet research.

This module provides a flexible UNet-style architecture that can mix
standard CNN blocks and Mamba VSS blocks in the encoder and decoder.
"""

import torch
import torch.nn as nn

from .vss_block import VSSBlock
from .mamba_layers import PatchPartition, PatchMerging, PatchExpanding, LinearProjection


# -----------------------------------------------------------------------------
# CNN building blocks for the baseline comparison
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNNEncoderStage(nn.Module):
    def __init__(self, dim, depth=2, downsample=True):
        super().__init__()
        self.linear_embed = nn.Conv2d(dim, dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[ConvBlock(dim, dim) for _ in range(depth)]
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(dim * 2),
                nn.GELU()
            )
            if downsample else None
        )

    def forward(self, x):
        x = self.linear_embed(x)
        x = self.blocks(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down

        return x, x


class CNNDecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, depth=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_dim,
            in_dim // 2,
            kernel_size=2,
            stride=2
        )
        self.linear_proj = nn.Conv2d(
            in_dim // 2 + skip_dim,
            out_dim,
            kernel_size=1
        )
        self.blocks = nn.Sequential(
            *[ConvBlock(out_dim, out_dim) for _ in range(depth)]
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.linear_proj(x)
        x = self.blocks(x)
        return x


class CNNBottleneck(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBlock(dim, dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


# -----------------------------------------------------------------------------
# Mamba / VSS stages compatible with the current Mamba-UNet
# -----------------------------------------------------------------------------

class MambaEncoderStage(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0., downsample=True):
        super().__init__()
        self.linear_embed = nn.Conv2d(dim, dim, kernel_size=1)

        if isinstance(drop_path, (list, tuple)):
            dpr = list(drop_path)
        else:
            dpr = [drop_path] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x):
        x = self.linear_embed(x)
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down

        return x, x


class MambaDecoderStage(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, depth=2, drop_path=0.):
        super().__init__()
        self.upsample = PatchExpanding(in_dim)
        self.linear_proj = LinearProjection(
            in_dim=skip_dim + in_dim // 2,
            out_dim=out_dim
        )
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=out_dim, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.linear_proj(x)

        for blk in self.blocks:
            x = blk(x)

        return x


class MambaBottleneck(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[VSSBlock(hidden_dim=dim, drop_path=0.) for _ in range(depth)]
        )

    def forward(self, x):
        return self.blocks(x)


# -----------------------------------------------------------------------------
# Flexible ablation UNet
# -----------------------------------------------------------------------------

class AblationUNet(nn.Module):
    def __init__(
        self,
        img_size=512,
        in_chans=1,
        num_classes=2,
        embed_dim=96,
        depths=(2, 2, 9, 2),
        drop_path_rate=0.2,
        patch_size=4,
        encoder_type="cnn",
        decoder_type="cnn"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.patch_partition = PatchPartition(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            dim = int(embed_dim * 2 ** i)
            stage_kwargs = {
                "dim": dim,
                "depth": depths[i],
                "downsample": i < self.num_stages - 1
            }
            if encoder_type == "mamba":
                stage_kwargs["drop_path"] = dpr[cur:cur + depths[i]]
                stage = MambaEncoderStage(**stage_kwargs)
            else:
                stage = CNNEncoderStage(**stage_kwargs)

            self.encoder_stages.append(stage)
            cur += depths[i]

        bottleneck_dim = int(embed_dim * 2 ** (self.num_stages - 1))
        if decoder_type == "mamba":
            self.bottleneck = MambaBottleneck(bottleneck_dim, depth=2)
        else:
            self.bottleneck = CNNBottleneck(bottleneck_dim, depth=2)

        self.decoder_stages = nn.ModuleList()
        encoder_dims = [int(embed_dim * 2 ** i) for i in range(self.num_stages)]
        decoder_dims = encoder_dims[::-1]

        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            skip_dim = decoder_dims[i + 1]
            out_dim = decoder_dims[i + 1]
            depth = depths[self.num_stages - 2 - i]

            stage_kwargs = {
                "in_dim": in_dim,
                "skip_dim": skip_dim,
                "out_dim": out_dim,
                "depth": depth
            }

            if decoder_type == "mamba":
                stage_kwargs["drop_path"] = 0.
                stage = MambaDecoderStage(**stage_kwargs)
            else:
                stage = CNNDecoderStage(**stage_kwargs)

            self.decoder_stages.append(stage)

        self.final_expand = nn.ConvTranspose2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_partition(x)

        skip_connections = []
        for stage in self.encoder_stages:
            x_skip, x = stage(x)
            skip_connections.append(x_skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[:-1]

        for i, stage in enumerate(self.decoder_stages):
            skip = skip_connections[-(i + 1)]
            x = stage(x, skip)

        x = self.final_expand(x)
        x = self.seg_head(x)
        return x


# -----------------------------------------------------------------------------
# Specific ablation model variants
# -----------------------------------------------------------------------------

class UNet_CNN_CNN(AblationUNet):
    def __init__(self, **kwargs):
        super().__init__(encoder_type="cnn", decoder_type="cnn", **kwargs)


class UNet_Mamba_CNN(AblationUNet):
    def __init__(self, **kwargs):
        super().__init__(encoder_type="mamba", decoder_type="cnn", **kwargs)


class UNet_CNN_Mamba(AblationUNet):
    def __init__(self, **kwargs):
        super().__init__(encoder_type="cnn", decoder_type="mamba", **kwargs)


class UNet_Full_Mamba(AblationUNet):
    def __init__(self, **kwargs):
        super().__init__(encoder_type="mamba", decoder_type="mamba", **kwargs)


def get_ablation_model(name, **kwargs):
    variants = {
        "UNet_CNN_CNN": UNet_CNN_CNN,
        "UNet_Mamba_CNN": UNet_Mamba_CNN,
        "UNet_CNN_Mamba": UNet_CNN_Mamba,
        "UNet_Full_Mamba": UNet_Full_Mamba,
    }
    if name not in variants:
        raise ValueError(f"Unknown ablation model: {name}")
    return variants[name](**kwargs)
