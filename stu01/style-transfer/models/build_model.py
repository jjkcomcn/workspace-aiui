"""模型构建模块

包含风格迁移模型构建和损失函数初始化
"""

from typing import List, Tuple
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights
from .loss import ContentLoss, StyleLoss, GramMatrix
from configs.settings import DEFAULT_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_style_model_and_loss(
    cnn: nn.Module,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    style_weight: float = DEFAULT_CONFIG['STYLE_WEIGHT'],
    content_weight: float = DEFAULT_CONFIG['CONTENT_WEIGHT'],
    content_layers: List[str] = DEFAULT_CONFIG['CONTENT_LAYERS'],
    style_layers: List[str] = DEFAULT_CONFIG['STYLE_LAYERS']
) -> Tuple[nn.Sequential, List[StyleLoss], List[ContentLoss]]:
    """构建风格迁移模型并初始化损失函数
    
    Args:
        cnn: 预训练的CNN模型
        style_img: 风格图像张量
        content_img: 内容图像张量
        style_weight: 风格损失权重
        content_weight: 内容损失权重
        content_layers: 内容层列表
        style_layers: 风格层列表
        
    Returns:
        包含模型、风格损失列表、内容损失列表的元组
    """
    # 参数验证
    if not isinstance(cnn, nn.Module):
        raise TypeError("cnn must be a nn.Module")
    
    # 验证输入张量并确保使用float32类型
    for name, tensor in [('style_img', style_img), ('content_img', content_img)]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be torch.Tensor")
        if tensor.dim() != 4 or tensor.size(0) != 1:
            raise ValueError(f"{name} must be a 4D tensor with batch size 1")
        if tensor.dtype != torch.float32:
            logger.warning(f"Converting {name} to float32")
            tensor = tensor.to(torch.float32)
    
    # 验证权重参数
    if not (1e4 <= style_weight <= 1e6):
        logger.warning(f"style_weight {style_weight} is outside recommended range [1e4, 1e6]")
    if not (0.01 <= content_weight <= 1):
        logger.warning(f"content_weight {content_weight} is outside recommended range [0.01, 1]")
        
    # 确保模型在正确设备上
    cnn = cnn.to(DEVICE)
    # 冻结模型参数并启用no_grad
    with torch.no_grad():
        for param in cnn.parameters():
            param.requires_grad_(False)
    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []
    model = nn.Sequential().to(DEVICE)
    gram = GramMatrix().to(DEVICE)
    
    # 验证层名有效性
    all_layers = {f'conv_{i}' for i in range(1, 6)} | {f'pool_{i}' for i in range(1, 6)} | {f'relu_{i}' for i in range(1, 6)}
    invalid_layers = set(content_layers + style_layers) - all_layers
    if invalid_layers:
        raise ValueError(f"Invalid layer names: {invalid_layers}. Valid layers are: {all_layers}")

    # 获取features子模块
    features = cnn.features
    conv_counter = 1
    
    for layer in features.children():
        if isinstance(layer, nn.Conv2d):
            layer_name = f'conv_{conv_counter}'
            model.add_module(layer_name, layer)
            
            # 添加内容损失层
            if layer_name in content_layers:
                target = model(content_img.detach())
                # 检查特征值范围
                if torch.isnan(target).any() or torch.isinf(target).any():
                    raise ValueError(f"NaN/Inf detected in content features at layer {layer_name}")
                content_loss = ContentLoss(target, content_weight)
                model.add_module(f'content_loss_{conv_counter}', content_loss)
                content_losses.append(content_loss)
            
            # 添加风格损失层
            if layer_name in style_layers:
                target = model(style_img.detach())
                # 检查特征值范围
                if torch.isnan(target).any() or torch.isinf(target).any():
                    raise ValueError(f"NaN/Inf detected in style features at layer {layer_name}")
                target = gram(target)
                style_loss = StyleLoss(target, style_weight)
                model.add_module(f'style_loss_{conv_counter}', style_loss)
                style_losses.append(style_loss)
                
            conv_counter += 1
            
        elif isinstance(layer, nn.MaxPool2d):
            model.add_module(f'pool_{conv_counter}', layer)
            
        elif isinstance(layer, nn.ReLU):
            model.add_module(f'relu_{conv_counter}', layer)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Model built with {len(style_losses)} style layers and {len(content_losses)} content layers")
    return model, style_losses, content_losses
