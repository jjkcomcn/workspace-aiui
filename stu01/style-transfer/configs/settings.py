"""风格迁移项目配置

包含默认参数和设置
"""

DEFAULT_CONFIG = {
    # 数据路径配置
    'DATA_ROOT': 'data/',  # 数据根目录
    'DATA_INPUT_DIR': 'data/input/',  # 输入图片目录
    'DATA_OUTPUT_DIR': 'data/output/',  # 输出图片目录
    'DATA_STYLES_DIR': 'data/styles/',  # 风格图片目录
    'MODEL_WEIGHTS_PATH': 'data/models/',  # 预训练模型权重路径
    
    # 图片处理配置
    'IMAGE_SIZE': 1024,  # 提高输出分辨率
    'IMAGE_MEAN': [0.485, 0.456, 0.406],  # 恢复标准均值
    'IMAGE_STD': [0.229, 0.224, 0.225],   # 恢复标准标准差
    'COLOR_ENHANCE': True,   # 启用色彩增强
    'COLOR_SATURATION': 1.2, # 适度增强色彩饱和度
    'STYLE_LOSS_SCALE': 1e-8,  # 增强风格损失影响
    'PRE_SCALE_FACTOR': 0.7,  # 优化内存使用
    'INTERPOLATION': 'lanczos',  # 使用高质量插值
    'KEEP_ASPECT_RATIO': True,  # 保持宽高比
    
    # 损失权重
    'STYLE_WEIGHT': 2e6,    # 极端风格权重(动漫风格必须)
    'CONTENT_WEIGHT': 0.2,   # 极低内容权重
    'HISTOGRAM_MATCH': False,  # 禁用直方图匹配
    'SHARPEN_AMOUNT': 0.2,   # 轻微锐化
    
    # 优化器参数
    'LEARNING_RATE': 0.001,  # 更稳定的学习率
    'OPTIMIZER_STEPS': 500,   # 增加优化步数
    'LR_DECAY_STEPS': 50,     # 学习率衰减步数
    'LBFGS_HISTORY_SIZE': 50, # LBFGS历史记录大小
    'LBFGS_MAX_ITER': 10,     # LBFGS每次迭代最大步数
    'LBFGS_TOL_GRAD': 1e-6,   # 梯度容忍度
    'LBFGS_TOL_CHANGE': 1e-9, # 变化容忍度
    
    # 进度条配置
    'PROGRESS_BAR': True,     # 是否显示进度条
    'PROGRESS_UNIT': 'step',  # 进度条单位
    'PROGRESS_DESC': 'Style Transfer Progress',  # 进度条描述
    
    # 图像后处理
    'CLIP_MIN': 0.15,          # 标准亮度下限
    'CLIP_MAX': 0.85,         # 标准亮度上限
    
    # 图像归一化参数
    'IMAGE_MEAN': [0.485, 0.456, 0.406],  # 图像均值
    'IMAGE_STD': [0.229, 0.224, 0.225],   # 图像标准差
    
    # 数值稳定性参数
    'EPSILON': 1e-10,        # 数值稳定因子
    'MAX_FEATURE_VALUE': 1e4, # 特征值最大限制
    'MIN_FEATURE_VALUE': -1e4, # 特征值最小限制
    
    # Gram矩阵配置
    'GRAM_EPSILON': 1e-8,    # Gram矩阵计算稳定因子
    'GRAM_MAX_VALUE': 1e4,   # Gram矩阵值最大限制
    'GRAM_NORM_FACTOR': 1.0, # Gram矩阵归一化因子
    
    # 损失计算配置
    'LOSS_EPSILON': 1e-6,    # 损失计算稳定因子
    'MAX_LOSS_VALUE': 1e4,   # 损失值最大限制
    
    # 设备配置
    'FORCE_CPU': False,      # 启用CUDA加速
    'LOG_FORMAT': '%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    
    # 层配置
    'CONTENT_LAYERS': ['conv_3', 'conv_4'],  # 使用较深层获取更多内容结构
    'STYLE_LAYERS': ['conv_1', 'conv_2', 'conv_3'],  # 使用更浅层提取动漫风格
    'MAX_GRAD_NORM': 1.0,  # 添加梯度裁剪阈值
    
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
        # 验证路径配置
        if not isinstance(config['DATA_ROOT'], str):
            raise ValueError("DATA_ROOT should be a string path")
        if not isinstance(config['DATA_INPUT_DIR'], str):
            raise ValueError("DATA_INPUT_DIR should be a string path")
        if not isinstance(config['DATA_OUTPUT_DIR'], str):
            raise ValueError("DATA_OUTPUT_DIR should be a string path")
        if not isinstance(config['DATA_STYLES_DIR'], str):
            raise ValueError("DATA_STYLES_DIR should be a string path")
        if not isinstance(config['MODEL_WEIGHTS_PATH'], str):
            raise ValueError("MODEL_WEIGHTS_PATH should be a string path")
        if not isinstance(config['OUTPUT_DIR'], str):
            raise ValueError("OUTPUT_DIR should be a string path")
            
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

import torch

# 设备配置
DEVICE = torch.device("cpu" if DEFAULT_CONFIG['FORCE_CPU'] else 
                     "cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['DEFAULT_CONFIG', 'validate_config', 'DEVICE']
