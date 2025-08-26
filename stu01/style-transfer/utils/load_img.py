"""图像处理工具模块

包含图像加载、保存和转换功能
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

# 图像归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(image_path: str, target_size: int = None) -> torch.Tensor:
    """加载并预处理图像(先进行预缩放降低CPU压力)
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸(长边)
        
    Returns:
        预处理后的图像张量 [1, 3, H, W]
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # 转换颜色空间
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 预缩放降低处理压力
    pre_scale = DEFAULT_CONFIG.get('PRE_SCALE_FACTOR', 0.5)
    if pre_scale < 1.0:
        h, w = img.shape[:2]
        pre_h = int(h * pre_scale)
        pre_w = int(w * pre_scale)
        img = cv2.resize(img, (pre_w, pre_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 最终调整到目标尺寸
    if target_size is not None:
        h, w = img.shape[:2]
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 归一化并转换为张量
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img

def save_image(output_path: str, tensor: torch.Tensor):
    """保存图像张量到文件
    
    Args:
        output_path: 输出文件路径
        tensor: 图像张量 [1, 3, H, W]
    """
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 转换张量为图像
    img = tensor_to_image(tensor)
    
    # 根据文件扩展名设置保存参数
    ext = Path(output_path).suffix.lower()
    if ext == '.jpg' or ext == '.jpeg':
        # 保存JPEG图像，质量为95%
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == '.png':
        # 保存PNG图像，压缩级别为9(最高质量)
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    else:
        # 其他格式使用默认参数
        cv2.imwrite(output_path, img)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将张量转换为OpenCV图像
    
    Args:
        tensor: 图像张量 [1, 3, H, W]
        
    Returns:
        OpenCV格式的图像 [H, W, 3]
    """
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    tensor = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

def image_to_tensor(img: np.ndarray, target_size: int = None) -> torch.Tensor:
    """将OpenCV图像转换为张量
    
    Args:
        img: OpenCV格式的图像 [H, W, 3]
        target_size: 目标尺寸(长边)
        
    Returns:
        图像张量 [1, 3, H, W]
    """
    return load_image_from_array(img, target_size)

def load_image_from_array(img: np.ndarray, target_size: int = None) -> torch.Tensor:
    """从numpy数组加载图像
    
    Args:
        img: numpy数组图像 [H, W, 3]
        target_size: 目标尺寸(长边)
        
    Returns:
        图像张量 [1, 3, H, W]
    """
    # 转换颜色空间
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    if target_size is not None:
        h, w = img.shape[:2]
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 归一化并转换为张量
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img
