"""测试模块

包含项目核心功能的单元测试
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from style_transfer.models.build_model import get_style_model_and_loss
from style_transfer.models.loss import ContentLoss, StyleLoss, GramMatrix
from style_transfer.utils.load_img import load_image, save_image, tensor_to_image

@pytest.fixture
def test_images(tmp_path):
    """创建测试图像"""
    content_path = tmp_path / "content.jpg"
    style_path = tmp_path / "style.jpg"
    
    # 创建简单的测试图像
    content_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    style_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 保存图像
    import cv2
    cv2.imwrite(str(content_path), content_img)
    cv2.imwrite(str(style_path), style_img)
    
    return content_path, style_path

def test_load_save_image(tmp_path):
    """测试图像加载和保存功能"""
    # 创建测试图像
    img_path = tmp_path / "test.jpg"
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(img_path), test_img)
    
    # 加载图像
    tensor = load_image(str(img_path))
    assert tensor.dim() == 4
    assert tensor.size(0) == 1
    assert tensor.size(1) == 3
    
    # 保存图像
    output_path = tmp_path / "output.jpg"
    save_image(str(output_path), tensor)
    assert output_path.exists()

def test_gram_matrix():
    """测试Gram矩阵计算"""
    gram = GramMatrix()
    test_tensor = torch.randn(1, 3, 10, 10)
    result = gram(test_tensor)
    
    assert result.dim() == 3
    assert result.size(0) == 1
    assert result.size(1) == 3
    assert result.size(2) == 3

def test_content_loss():
    """测试内容损失计算"""
    target = torch.randn(1, 3, 10, 10)
    content_loss = ContentLoss(target, weight=1.0)
    input_tensor = torch.randn(1, 3, 10, 10, requires_grad=True)
    
    output = content_loss(input_tensor)
    assert output.size() == input_tensor.size()
    assert hasattr(content_loss, 'loss')

def test_style_loss():
    """测试风格损失计算"""
    target = torch.randn(1, 3, 3)
    style_loss = StyleLoss(target, weight=1.0)
    input_tensor = torch.randn(1, 3, 10, 10, requires_grad=True)
    
    output = style_loss(input_tensor)
    assert output.size() == input_tensor.size()
    assert hasattr(style_loss, 'loss')

def test_model_building(test_images):
    """测试模型构建功能"""
    content_path, style_path = test_images
    content_img = load_image(str(content_path))
    style_img = load_image(str(style_path))
    
    # 加载预训练模型
    import torchvision.models as models
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
    
    # 构建风格迁移模型
    model, style_losses, content_losses = get_style_model_and_loss(
        cnn, style_img, content_img
    )
    
    assert len(style_losses) > 0
    assert len(content_losses) > 0
    assert isinstance(model, torch.nn.Sequential)

def test_tensor_image_conversion():
    """测试张量和图像转换功能"""
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tensor = tensor_to_image(image_to_tensor(test_img))
    
    assert tensor.shape == test_img.shape
    assert tensor.dtype == np.uint8
