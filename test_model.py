import os
import cv2
import torch
import numpy as np
from models.mamba_unet import create_mamba_unet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/mnt/c/project/Mamba/checkpoints/20260128_234249/best.pth"
IMAGE_PATH = "./test_image.jpg"

SAVE_MASK_PATH = "./prediction_mask.png"
SAVE_OVERLAY_PATH = "./prediction_overlay.png"

IMG_SIZE = 512
NUM_CLASSES = 2


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không đọc được ảnh")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    img = (img - 0.5) / 0.5

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    return torch.from_numpy(img).float()


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def overlay_mask(image_gray, mask):
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    color_mask = np.zeros_like(image_color)
    color_mask[mask == 255] = (0, 0, 255)

    overlay = cv2.addWeighted(image_color, 0.75, color_mask, 0.25, 0)
    return overlay


def main():
    print("🚀 Loading model...")

    model = create_mamba_unet(
        in_chans=1,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded!")

    raw_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))

    image_tensor = preprocess_image(IMAGE_PATH).to(DEVICE)

    print("🖼️ Predicting...")

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy()

    mask = (pred * 255).astype(np.uint8)

    mask = clean_mask(mask)

    overlay = overlay_mask(raw_img, mask)

    cv2.imwrite(SAVE_MASK_PATH, mask)
    cv2.imwrite(SAVE_OVERLAY_PATH, overlay)

    print("✅ Saved:")
    print(" - Mask:", SAVE_MASK_PATH)
    print(" - Overlay:", SAVE_OVERLAY_PATH)


if __name__ == "__main__":
    main()
