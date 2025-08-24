import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional

def load_image(filename: str, size: Optional[int] = None, scale: Optional[float] = None) -> torch.Tensor:
    """加载并预处理图像
    
    Args:
        filename: 图像文件路径
        size: 调整大小(可选)
        scale: 缩放比例(可选)
        
    Returns:
        预处理后的图像张量 [1, C, H, W]
        
    Raises:
        ValueError: 如果图像加载失败
    """
    try:
        img = Image.open(filename).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.Resampling.LANCZOS)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(img).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Failed to load image {filename}: {str(e)}")

def image_to_tensor(image: Image.Image, size: Optional[int] = None) -> torch.Tensor:
    """转换PIL图像为张量
    
    Args:
        image: PIL图像对象
        size: 调整大小(可选)
        
    Returns:
        图像张量 [1, C, H, W]
    """
    if size is not None:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def save_image(filename: str, tensor: torch.Tensor):
    """保存张量为图像
    
    Args:
        filename: 保存路径
        tensor: 图像张量 [1, C, H, W]
        
    Raises:
        ValueError: 如果图像保存失败
    """
    try:
        img = tensor_to_image(tensor)
        img.save(filename)
    except Exception as e:
        raise ValueError(f"Failed to save image {filename}: {str(e)}")

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """转换张量为PIL图像
    
    Args:
        tensor: 图像张量 [1, C, H, W]
        
    Returns:
        PIL图像对象
    """
    img = tensor.squeeze(0).cpu().detach()
    img = img.numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """计算Gram矩阵
    
    Args:
        features: 特征张量 [B, C, H, W]
        
    Returns:
        Gram矩阵 [B, C, C]
    """
    batch_size, channel, height, width = features.size()
    features = features.view(batch_size * channel, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channel * height * width + 1e-8)

def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """批量归一化
    
    Args:
        batch: 输入批量张量
        
    Returns:
        归一化后的张量
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

def denormalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """反归一化
    
    Args:
        batch: 输入批量张量
        
    Returns:
        反归一化后的张量
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return batch * std + mean

def match_histograms(source: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """直方图匹配
    
    Args:
        source: 源图像张量
        template: 模板图像张量
        
    Returns:
        匹配后的图像张量
    """
    # 实现直方图匹配算法
    pass
