"""风格迁移项目配置

包含默认参数和设置
"""

DEFAULT_CONFIG = {
    # 图片处理配置
    'IMAGE_SIZE': 256,  # 默认图片尺寸
    'IMAGE_MEAN': [0.485, 0.456, 0.406],  # 图像归一化均值
    'IMAGE_STD': [0.229, 0.224, 0.225],   # 图像归一化标准差
    
    # 损失权重
    'STYLE_WEIGHT': 1e1,    # 风格权重(推荐范围: 1e4-1e6)
    'CONTENT_WEIGHT': 1,     # 内容权重(推荐范围: 0.01-1)
    
    # 优化器参数
    'LEARNING_RATE': 0.1,    # 学习率s
    'OPTIMIZER_STEPS': 5,  # 优化步数(代码中s实际步数为num_steps * 2)
    
    # 层配置
    'CONTENT_LAYERS': ['conv_4'],  # 内容层
    'STYLE_LAYERS': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],  # 风格层
}

def validate_config(config: dict) -> bool:
    """验证配置参数有效性
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 配置是否有效
    """
    try:
        # 验证图片尺寸
        if not (64 <= config['IMAGE_SIZE'] <= 1024):
            raise ValueError("IMAGE_SIZE should be between 64 and 1024")
            
        # 验证权重参数
        if not (1e4 <= config['STYLE_WEIGHT'] <= 1e6):
            raise ValueError("STYLE_WEIGHT should be between 1e4 and 1e6")
            
        if not (0.01 <= config['CONTENT_WEIGHT'] <= 1):
            raise ValueError("CONTENT_WEIGHT should be between 0.01 and 1")
            
        return True
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid config value: {e}")

__all__ = ['DEFAULT_CONFIG', 'validate_config']
