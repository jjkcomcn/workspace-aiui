"""Style transfer loss functions

Implements content loss, style loss and Gram matrix calculation
"""

import torch
import torch.nn as nn
from configs import settings

class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    
    def forward(self, input):
        batch_size, num_channels, h, w = input.size()
        features = input.view(batch_size * num_channels, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * num_channels * h * w)

class ContentLoss(nn.Module):
    """Content loss between target and input features"""
    
    def __init__(self, target, weight):
        super().__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss = None
        
    def forward(self, input):
        self.loss = torch.mean((input - self.target)**2) * self.weight
        return input
        
class StyleLoss(nn.Module):
    """Style loss between target and input Gram matrices"""
    
    def __init__(self, target, weight):
        super().__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss = None
        
    def forward(self, input):
        gram = GramMatrix()(input)
        self.loss = torch.mean((gram - self.target)**2) * self.weight
        return input

def get_loss():
    """Get dummy loss function (not used in style transfer)"""
    return nn.MSELoss()
