import sys
import importlib.metadata
import os
import logging
from typing import Dict, List
from packaging import version

def check_environment() -> bool:
    """检查项目运行环境
    
    Returns:
        bool: 环境检查是否通过
    """
    logger = logging.getLogger(__name__)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # 检查依赖包
    required_packages: Dict[str, str] = {
        'numpy': '1.21.0',
        'tensorflow': '2.6.0',
        'pandas': '1.3.0',
        'matplotlib': '3.4.0',
        'packaging': '21.0'
    }
    
    missing_packages: List[str] = []
    for package, min_version in required_packages.items():
        try:
            installed_version = importlib.metadata.version(package)
            if version.parse(installed_version) < version.parse(min_version):
                logger.warning(
                    f"Package {package} version {installed_version} is below minimum required {min_version}"
                )
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install them using: pip install -r requirements.txt")
        return False
    
    # 检查数据目录
    required_dirs = [
        'data/input',
        'data/output',
        'data/processed'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {', '.join(missing_dirs)}")
        logger.info("Please create them manually")
        return False
    
    logger.info("✅ Environment check passed")
    return True
