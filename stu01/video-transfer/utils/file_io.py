"""文件操作工具模块

该模块提供文件和目录操作的实用工具函数，
包括路径处理、目录创建和文件列表获取等功能。

主要函数：
ensure_dir - 确保目录存在
list_images - 获取目录中的图片文件列表

使用示例：
    # 确保目录存在
    data_dir = ensure_dir("data/input")
    
    # 获取目录中的图片
    images = list_images("data/input")
"""
from pathlib import Path
from typing import Union, List

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保指定目录存在，如不存在则创建
    
    参数:
        path: 要确保存在的目录路径 (类型: str或Path对象)
        
    返回:
        Path: 创建或已存在的目录Path对象
        
    异常:
        OSError: 当目录创建失败时抛出
        
    示例:
        >>> data_dir = ensure_dir("data/input")
        >>> print(f"目录已确保: {data_dir}")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_images(dir_path: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """获取目录中指定扩展名的图片文件列表
    
    参数:
        dir_path: 要扫描的目录路径 (类型: str或Path对象)
        extensions: 要包含的文件扩展名列表
                   (默认: ['.jpg', '.png', '.jpeg'])
        
    返回:
        List[Path]: 匹配的图片文件路径列表，按文件名排序
        
    异常:
        ValueError: 当目录不存在时抛出
        
    示例:
        >>> images = list_images("data/input")
        >>> print(f"找到 {len(images)} 张图片")
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"不是有效的目录: {dir_path}")
        
    extensions = extensions or ['.jpg', '.png', '.jpeg']
    image_files = []
    
    for ext in extensions:
        image_files.extend(dir_path.glob(f'*{ext}'))
        
    return sorted(image_files)
