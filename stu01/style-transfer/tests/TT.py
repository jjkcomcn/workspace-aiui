import sys
from pathlib import Path
import os
import logging
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))
print(f"Current working directory: {sys.path}")