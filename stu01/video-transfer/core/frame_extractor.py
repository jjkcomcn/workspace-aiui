"""视频帧提取模块

该模块提供从视频文件中提取帧序列的功能，支持控制提取帧率，
并将提取的帧保存为图像文件。

主要类：
FrameExtractor - 处理视频帧提取的核心类

使用示例：
    extractor = FrameExtractor("output_frames")
    frames = extractor.extract("input.mp4", fps=24)
    # frames 将包含所有提取的帧路径列表
"""
import cv2
from pathlib import Path
from typing import Union, List
from tqdm import tqdm

class FrameExtractor:
    def __init__(self, output_dir: Union[str, Path]):
        """初始化帧提取器
        
        参数:
            output_dir: 帧输出目录，提取的帧将保存到此目录
                       (类型: str或Path对象)
        
        属性:
            output_dir: Path对象，确保的帧输出目录路径
        """
        self.output_dir = Path(output_dir)
        
    def extract(self, video_path: Union[str, Path], fps: int = None) -> List[Path]:
        """从视频文件中提取帧序列
        
        参数:
            video_path: 要提取的视频文件路径 (类型: str或Path对象)
            fps: 目标提取帧率，None表示使用视频原始帧率 (类型: int)
            
        返回:
            List[Path]: 提取的帧文件路径列表，按帧顺序排序
            
        异常:
            ValueError: 当视频文件无法打开时抛出
            
        示例:
            >>> extractor = FrameExtractor("output_frames")
            >>> frames = extractor.extract("input.mp4", fps=24)
            >>> print(f"提取了{len(frames)}帧")
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = fps or original_fps
        frame_interval = int(round(original_fps / target_fps))
        
        frame_paths = []
        frame_count = 0
        success, frame = cap.read()
        
        with tqdm(desc="提取视频帧") as pbar:
            while success:
                if frame_count % frame_interval == 0:
                    frame_path = self.output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)
                    pbar.update(1)
                
                success, frame = cap.read()
                frame_count += 1
                
        cap.release()
        return frame_paths
