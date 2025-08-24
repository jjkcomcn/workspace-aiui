"""损失函数模块

包含风格迁移的内容损失和风格损失实现
"""

import torch
import torch.nn as nn
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
        """高效计算Gram矩阵
        
        Args:
            input: 输入特征张量 [batch, channel, height, width]
            
        Returns:
            计算得到的Gram矩阵
        """
        batch_size, channel, height, width = input.size()
        features = input.view(batch_size, channel, -1)  # [batch, channel, height*width]
        gram = torch.bmm(features, features.transpose(1, 2))  # [batch, channel, channel]
        return gram.div_(channel * height * width + 1e-8)

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
        """计算风格损失
        
        Args:
            input: 输入特征张量
            
        Returns:
            输入特征张量的克隆
        """
        G = self.gram(input)
        self.loss = self.criterion(G * self.weight, self.target)
        return input.clone()
