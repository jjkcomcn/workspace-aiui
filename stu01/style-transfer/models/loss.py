"""损失函数模块

包含风格迁移的内容损失和风格损失实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ContentLoss(nn.Module):
    """内容损失函数，衡量生成图像与内容图像在特征空间的距离"""
    
    def __init__(self, target: torch.Tensor, weight: float):
        """初始化内容损失
        
        Args:
            target: 目标特征张量 [batch, channel, height, width]
            weight: 损失权重 (推荐值范围: 0.01-1)
        """
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.register_buffer('target', target.detach() * self.weight)
        self.criterion = nn.MSELoss()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """计算内容损失
        
        Args:
            input: 输入特征张量
            
        Returns:
            输入特征张量的克隆
        """
        self.loss = self.criterion(input * self.weight, self.target)
        return input.clone()

class GramMatrix(nn.Module):
    """Gram矩阵计算模块"""
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """高效计算Gram矩阵(优化版)
        
        Args:
            input: 输入特征张量 [batch, channel, height, width]
            
        Returns:
            计算得到的Gram矩阵
        """
        batch_size, channel, height, width = input.size()
        
        # 添加输入值范围检查
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Input contains NaN or Inf values")
            
        features = input.view(batch_size, channel, -1)  # [batch, channel, height*width]
        
        # 使用配置参数增强数值稳定性
        eps = DEFAULT_CONFIG['GRAM_EPSILON']
        max_val = DEFAULT_CONFIG['GRAM_MAX_VALUE']
        
        # 输入值裁剪
        features = features.clamp(
            min=-max_val,
            max=max_val
        )
        
        # 计算Gram矩阵
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # 更安全的归一化
        norm_factor = channel * height * width + eps
        gram = gram.clamp(
            min=-max_val,
            max=max_val
        )
        gram = gram / norm_factor * DEFAULT_CONFIG['GRAM_NORM_FACTOR']
        
        # 确保没有极值
        gram = torch.nan_to_num(
            gram,
            nan=0.0,
            posinf=max_val,
            neginf=-max_val
        )
        
        # 检查输出
        if torch.isnan(gram).any() or torch.isinf(gram).any():
            raise ValueError("Gram matrix contains NaN or Inf values")
        return gram

class StyleLoss(nn.Module):
    """风格损失函数，衡量生成图像与风格图像在Gram矩阵空间的差异"""
    
    def __init__(self, target: torch.Tensor, weight: float):
        """初始化风格损失
        
        Args:
            target: 目标Gram矩阵 [channel, channel]
            weight: 损失权重 (推荐值范围: 1e4-1e6)
        """
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.register_buffer('target', target.detach() * self.weight)
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """计算风格损失(优化版)
        
        Args:
            input: 输入特征张量
            
        Returns:
            输入特征张量的克隆
        """
        G = self.gram(input)
        
        # 数值稳定处理
        eps = DEFAULT_CONFIG['LOSS_EPSILON']
        safe_weight = max(self.weight, eps)
        safe_numel = max(G.numel(), 1)
        
        # 计算损失
        loss = F.mse_loss(
            G, 
            self.target, 
            reduction='mean'
        )
        
        # 损失限制
        self.loss = loss.clamp(
            min=0, 
            max=DEFAULT_CONFIG['MAX_LOSS_VALUE']
        ) * self.weight
        
        # 确保损失不为NaN
        if torch.isnan(self.loss):
            self.loss = torch.tensor(0.0, device=self.loss.device)
        return input.clone()
