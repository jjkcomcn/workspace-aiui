"""环境检查模块

检查运行环境是否满足要求
"""

import sys
import torch
import numpy as np
import cv2
from typing import Tuple

def check_environment() -> Tuple[bool, str]:
    """检查运行环境
    
    Returns:
        Tuple[bool, str]: (是否满足要求, 环境信息)
    """
    info = []
    all_ok = True
    
    # Python版本
    py_version = sys.version.split()[0]
    info.append(f"Python: {py_version}")
    
    # PyTorch版本
    try:
        torch_version = torch.__version__
        info.append(f"PyTorch: {torch_version}")
    except Exception as e:
        info.append(f"PyTorch: Not available ({str(e)})")
        all_ok = False
    
    # CUDA可用性
    try:
        cuda_available = torch.cuda.is_available()
        info.append(f"CUDA: {'Available' if cuda_available else 'Not available'}")
        if cuda_available:
            info.append(f"CUDA devices: {torch.cuda.device_count()}")
            info.append(f"Current device: {torch.cuda.current_device()}")
            info.append(f"Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        info.append(f"CUDA check failed: {str(e)}")
        all_ok = False
    
    # OpenCV版本
    try:
        cv_version = cv2.__version__
        info.append(f"OpenCV: {cv_version}")
    except Exception as e:
        info.append(f"OpenCV: Not available ({str(e)})")
        all_ok = False
    
    # NumPy版本
    try:
        np_version = np.__version__
        info.append(f"NumPy: {np_version}")
    except Exception as e:
        info.append(f"NumPy: Not available ({str(e)})")
        all_ok = False
    
    # 检查PyTorch和CUDA版本兼容性
    if all_ok:
        try:
            torch.zeros(1).cuda()  # 简单CUDA测试
            info.append("PyTorch-CUDA compatibility: OK")
        except Exception as e:
            info.append(f"PyTorch-CUDA compatibility error: {str(e)}")
            all_ok = False
    
    return all_ok, "\n".join(info)

if __name__ == '__main__':
    ok, env_info = check_environment()
    print("Environment check:")
    print(env_info)
    print("\nEnvironment is", "OK" if ok else "NOT OK")
