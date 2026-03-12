import torch
from models.mamba_unet import create_mamba_unet

# Create model
model = create_mamba_unet(in_chans=1, num_classes=2, img_size=512)

# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params / 1e6:.2f}M")
print(f"Trainable params: {trainable_params / 1e6:.2f}M")

# Test forward pass
x = torch.randn(1, 1, 512, 512)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
