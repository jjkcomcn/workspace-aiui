"""Image loading utilities for style transfer

Modified to support single image loading and preprocessing
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def load_image(image_path, size=512):
    """Load and preprocess single image for style transfer
    
    Args:
        image_path: Path to image file
        size: Target size for resizing
        
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    # Check file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms and add batch dimension
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, filename):
    """Save image tensor to file
    
    Args:
        tensor: Image tensor [1, C, H, W]
        filename: Output file path
    """
    # Inverse normalization
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    
    # Remove batch dimension and save
    image = tensor.squeeze(0).cpu().detach()
    image = transform(image)
    image.save(filename)
