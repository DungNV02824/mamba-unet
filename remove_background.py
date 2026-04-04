import cv2
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models.mamba_unet import create_mamba_unet

def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained Mamba-UNet model"""

    model = create_mamba_unet(in_chans=1, num_classes=2, img_size=512)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load only model weights
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    return model, device

def preprocess_image(img_path, img_size=512):
    """Load and preprocess image to model input format"""
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None
    
    original_shape = img.shape
    
    # Resize to model input size
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor (1, 1, 512, 512)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, original_shape

@torch.no_grad()
def predict_mask(model, img_tensor, device):
    """Predict segmentation mask"""
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    
    # Get mask from output (class 1 - tooth)
    mask = torch.softmax(output, dim=1)  # (1, 2, 512, 512)
    mask = mask[0, 1, :, :].cpu().numpy()  # Get class 1 probability
    
    # Threshold at 0.5
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    return binary_mask

def remove_background(image_dir, output_dir, checkpoint_path, img_size=512):
    """Process all images in directory and remove backgrounds"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, device = load_model(checkpoint_path)
    print(f"Model loaded from {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        
        # Preprocess
        result = preprocess_image(img_path, img_size)
        if result is None:
            continue
        img_tensor, original_shape = result
        
        # Predict mask
        binary_mask = predict_mask(model, img_tensor, device)
        
        # Resize mask back to original size if needed
        if binary_mask.shape != original_shape:
            binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
        
        # Load original image
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply mask to remove background
        clean_img = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
        
        # Save result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, clean_img)
    
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    # Configuration
    # Find the latest checkpoint
    checkpoint_dir = "checkpoints"
    latest_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint, "best.pth")
    
    # Input/Output paths
    dataset_images = "/mnt/c/project/Mamba/data_yolov/valid/images"
    output_clean_images = "/mnt/c/project/Mamba/data_yolov/valid/cleaned_images"
    
    # Process
    remove_background(dataset_images, output_clean_images, checkpoint_path)