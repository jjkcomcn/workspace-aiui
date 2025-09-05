"""项目配置模块

该模块集中管理所有项目配置参数，
包括路径设置、视频处理参数和日志配置。

配置分类：
1. DEFAULT_PATHS - 默认路径配置
2. VIDEO_CONFIG - 视频处理参数
3. STYLE_TRANSFER_CONFIG - 风格迁移参数
4. LOG_CONFIG - 日志设置

使用示例：
    from config.settings import DEFAULT_PATHS, VIDEO_CONFIG
    input_dir = DEFAULT_PATHS['input_dir']
    fps = VIDEO_CONFIG['default_fps']
"""
from pathlib import Path

# 项目根目录(自动计算)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径配置
DEFAULT_PATHS = {
    'input_dir': PROJECT_ROOT / 'data/input',    # 默认输入目录
    'output_dir': PROJECT_ROOT / 'data/output',  # 默认输出目录
    'style_dir': PROJECT_ROOT / 'data/styles',   # 风格图片目录
    'temp_dir': PROJECT_ROOT / 'temp'           # 临时文件目录
}

# 视频处理配置
VIDEO_CONFIG = {
    'default_fps': 30,           # 默认帧率
    'codec': 'mp4v',            # 默认视频编码格式
    'max_frames': 1000          # 最大处理帧数限制(防止内存溢出)
}

# 风格迁移配置
STYLE_TRANSFER_CONFIG = {
    'default_size': 512,         # 默认处理尺寸(像素)
    'style_weight': 1e5,         # 风格权重
    'content_weight': 1,         # 内容权重
    'steps': 300                 # 默认迭代步数
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',             # 日志级别: DEBUG/INFO/WARNING/ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
}
