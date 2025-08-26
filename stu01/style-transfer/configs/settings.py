"""风格迁移项目配置

包含默认参数和设置
"""

DEFAULT_CONFIG = {
    # 图片处理配置
    'IMAGE_SIZE': 192,  # 平衡分辨率
    'IMAGE_MEAN': [0.0, 0.0, 0.0],  # 保持禁用归一化
    'IMAGE_STD': [1.0, 1.0, 1.0],   # 保持禁用归一化
    'COLOR_ENHANCE': True,  # 保持色彩增强
    'STYLE_LOSS_SCALE': 5e-10,  # 降低风格损失缩放因子
    
    # 损失权重
    'STYLE_WEIGHT': 1e4,    # 降低风格权重
    'CONTENT_WEIGHT': 0.5,   # 调整内容权重
    
    # 优化器参数
    'LEARNING_RATE': 0.0015, # 学习率
    'OPTIMIZER_STEPS': 500,  # 优化步数
    'LR_DECAY_STEPS': 50,  # 学习率衰减步数
    
    # 层配置
    'CONTENT_LAYERS': ['conv_4', 'conv_5'],  # 增加更深层内容特征
    'STYLE_LAYERS': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'pool_1', 'pool_2', 'pool_3', 'pool_4'],  # 增加所有池化层
    
    # CPU优化参数
    'CPU_THREADS': 8,  # 增加CPU线程数
    'BATCH_SIZE': 4,   # 增大批处理大小
    'USE_PARALLEL': True,  # 启用并行处理
    'MEMORY_OPTIMIZATION': True,  # 是否启用内存优化(默认True)
    'CPU_AFFINITY': True,  # 启用CPU核心绑定
    'PREFETCH_FACTOR': 4,  # 增加数据预取因子
    'NUM_WORKERS': 4,  # 增加工作线程数
    'CPU_SCHED_POLICY': 'throughput',  # CPU调度策略调整为高吞吐
    'THREAD_SPIN_TIME': 150,  # 增加线程自旋等待时间(微秒)
    'PRE_SCALE_FACTOR': 0.3,  # 降低图像预缩放比例
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
            
        # 验证CPU优化参数
        if not (1 <= config['CPU_THREADS'] <= 32):
            raise ValueError("CPU_THREADS should be between 1 and 32")
            
        if not (1 <= config['BATCH_SIZE'] <= 32):
            raise ValueError("BATCH_SIZE should be between 1 and 32")
            
        if 'NUM_WORKERS' in config and not (0 <= config['NUM_WORKERS'] <= 8):
            raise ValueError("NUM_WORKERS should be between 0 and 8")
            
        if 'PREFETCH_FACTOR' in config and not (1 <= config['PREFETCH_FACTOR'] <= 4):
            raise ValueError("PREFETCH_FACTOR should be between 1 and 4")
            
        if 'PRE_SCALE_FACTOR' in config and not (0.1 <= config['PRE_SCALE_FACTOR'] <= 1.0):
            raise ValueError("PRE_SCALE_FACTOR should be between 0.1 and 1.0")
            
        return True
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid config value: {e}")

__all__ = ['DEFAULT_CONFIG', 'validate_config']
