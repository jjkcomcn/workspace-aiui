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
        tensor: 图像张量 [1, C, H, W] 或 [C, H, W]
        
    Returns:
        PIL图像对象
        
    Raises:
        ValueError: 如果输入张量维度不正确
    """
    # 确保输入是3D或4D张量
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
        
    # 如果是4D张量，只处理第一个图像
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            print(f"Warning: Processing only first image from batch of size {tensor.size(0)}")
        tensor = tensor[0]  # 取第一个图像
        
    # 处理通道数
    if tensor.size(0) == 1:
        # 灰度图转RGB
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.size(0) == 2:
        # 2通道图像，添加第三个通道
        tensor = torch.cat([tensor, torch.zeros_like(tensor[0:1])], dim=0)
    elif tensor.size(0) != 3:
        raise ValueError(f"Expected 1, 2 or 3 channels, got {tensor.size(0)}")
        
    # 转换到CPU和numpy
    img = tensor.cpu().detach()
    
    try:
        # 验证张量形状
        if tensor.dim() != 3 or tensor.size(0) not in [1, 2, 3]:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}")
            
        # 转换为numpy数组
        img = img.numpy()
        
        # 检查并处理NaN值
        if np.isnan(img).any():
            import logging
            logging.warning("NaN values detected in image tensor, replacing with zeros")
            img = np.nan_to_num(img)
        
        # 确保是3通道
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        elif img.shape[0] == 2:
            img = np.concatenate([img, np.zeros_like(img[0:1])], axis=0)
            
        # 维度转换
        if img.shape[0] != 3:
            raise ValueError(f"Expected 3 channels after conversion, got {img.shape[0]}")
            
        img = img.transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8))
    except Exception as e:
        raise ValueError(f"Failed to convert tensor {tensor.shape} to image: {str(e)}")

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
        source: 源图像张量 [1, C, H, W]
        template: 模板图像张量 [1, C, H, W]
        
    Returns:
        匹配后的图像张量 [1, C, H, W]
    """
    def _match_channel(source_ch: np.ndarray, template_ch: np.ndarray) -> np.ndarray:
        """单通道直方图匹配"""
        src_values, src_idx, src_counts = np.unique(source_ch, return_inverse=True, return_counts=True)
        tgt_values, tgt_counts = np.unique(template_ch, return_counts=True)
        
        # 计算累积分布函数
        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]
        tgt_cdf = np.cumsum(tgt_counts).astype(np.float64)
        tgt_cdf /= tgt_cdf[-1]
        
        # 插值匹配
        interp_values = np.interp(src_cdf, tgt_cdf, tgt_values)
        return interp_values[src_idx].reshape(source_ch.shape)
    
    # 转换到CPU和numpy
    source_np = source.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    template_np = template.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # 对每个通道进行匹配
    matched = np.zeros_like(source_np)
    for ch in range(source_np.shape[2]):
        matched[..., ch] = _match_channel(source_np[..., ch], template_np[..., ch])
    
    # 转换回tensor
    matched = torch.from_numpy(matched.transpose(2, 0, 1)).unsqueeze(0).to(source.device)
    return matched

def sharpen_image(image: torch.Tensor, amount: float = 0.5) -> torch.Tensor:
    """图像锐化
    
    Args:
        image: 输入图像张量 [1, C, H, W]
        amount: 锐化强度 (0-1)
        
    Returns:
        锐化后的图像张量
    """
    kernel = torch.tensor([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]], dtype=torch.float32, device=image.device) * amount
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 9.0
    return torch.nn.functional.conv2d(image, kernel, padding=1, groups=3)

def save_image(filename: str, tensor: torch.Tensor, quality: int = 95):
    """保存张量为图像
    
    Args:
        filename: 保存路径
        tensor: 图像张量 [1, C, H, W]
        quality: 保存质量 (1-100)
        
    Raises:
        ValueError: 如果图像保存失败
    """
    try:
        img = tensor_to_image(tensor)
        img.save(filename, quality=quality, subsampling=0)
    except Exception as e:
        raise ValueError(f"Failed to save image {filename}: {str(e)}")
