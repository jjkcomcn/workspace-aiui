"""风格应用模块

该模块负责将风格迁移算法应用于帧序列或单帧，支持批量处理和实时处理。

主要类：
StyleApplier - 处理风格迁移应用的核心类

使用示例：
    applier = StyleApplier("style.jpg")
    # 批量处理
    styled_frames = applier.apply(frames, "styled_frames")
    # 单帧处理
    styled_frame = applier.apply_frame(frame)
"""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, List
from tqdm import tqdm

class StyleApplier:
    def __init__(self, style_image_path: str):
        """初始化风格应用器
        
        参数:
            style_image_path: 用于风格迁移的风格图片路径
                           (类型: str)
                           
        属性:
            style_image_path: 风格图片路径字符串
        """
        self.style_image_path = style_image_path
        
    def apply(self, frame_paths: List[Union[str, Path]], output_dir: Union[str, Path]) -> List[Path]:
        """对帧序列应用风格迁移
        
        参数:
            frame_paths: 要处理的帧路径列表 (类型: List[str|Path])
            output_dir: 风格化帧输出目录 (类型: str|Path)
            
        返回:
            List[Path]: 风格化后的帧路径列表，保持原始顺序
            
        注意:
            依赖于外部style_transfer模块进行实际风格迁移
            
        示例:
            >>> applier = StyleApplier("style.jpg")
            >>> styled = applier.apply(["frame1.jpg", "frame2.jpg"], "output")
            >>> print(f"生成了{len(styled)}张风格化帧")
        """
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from importlib import import_module
        style_transfer = import_module("style-transfer.scripts.run_main").main
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        styled_frames = []
        for frame_path in tqdm(frame_paths, desc="应用风格迁移"):
            output_path = output_dir / f"styled_{frame_path.name}"
            style_transfer(
                content=str(frame_path),
                style=self.style_image_path,
                output=str(output_path),
                size=512  # 可配置
            )
            styled_frames.append(output_path)
            
        return styled_frames
        
    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        """应用风格迁移到单帧
        
        参数:
            frame: 输入帧(numpy数组)
            
        返回:
            风格化后的帧(numpy数组)
            
        注意:
            1. 输入应为BGR格式的numpy数组
            2. 输出为BGR格式的numpy数组
            
        示例:
            >>> frame = cv2.imread("frame.jpg")
            >>> styled = applier.apply_frame(frame)
            >>> cv2.imshow("Styled", styled)
        """
        import sys
        from pathlib import Path
        
        # 确保所有可能的父目录都在路径中
        project_root = str(Path(__file__).parent.parent.parent.parent)  # workspace-aiui目录
        style_transfer_root = str(Path(__file__).parent.parent.parent / "style-transfer")
        scripts_dir = str(Path(__file__).parent.parent.parent / "style-transfer/scripts")
        
        for path in [project_root, style_transfer_root, scripts_dir]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        try:
            from importlib import import_module
            style_transfer = import_module("style-transfer.scripts.run_main").main # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Failed to import style transfer module. Checked paths:\n"
                f"- Project root: {project_root}\n"
                f"- Style transfer root: {style_transfer_root}\n"
                f"- Scripts dir: {scripts_dir}\n"
                f"Current sys.path: {sys.path}\n"
                f"Original error: {str(e)}"
            )
        
        # 临时保存输入帧
        temp_input = Path("temp_input.jpg")
        temp_output = Path("temp_output.jpg")
        
        # 保存输入帧
        cv2.imwrite(str(temp_input), frame)
        
        # 应用风格迁移
        style_transfer(
            content=str(temp_input),
            style=self.style_image_path,
            output=str(temp_output),
            size=frame.shape[1]  # 保持原始宽度
        )
        
        # 读取并返回结果
        styled_frame = cv2.imread(str(temp_output))
        
        # 清理临时文件
        temp_input.unlink(missing_ok=True)
        temp_output.unlink(missing_ok=True)
        
        return styled_frame
