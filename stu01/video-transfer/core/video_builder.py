"""视频重建模块

该模块负责从风格化后的帧序列重建视频文件，
支持多种视频编码格式和输出参数配置。

主要类：
VideoBuilder - 处理视频重建的核心类

使用示例：
    builder = VideoBuilder(codec='mp4v', fps=30)
    video_path = builder.build(frames, "output.mp4")
    # video_path 为生成的视频文件路径
"""
import cv2
from pathlib import Path
from typing import Union, List
from tqdm import tqdm

class VideoBuilder:
    def __init__(self, codec: str = 'mp4v', fps: int = 30):
        """初始化视频构建器
        
        参数:
            codec: 视频编码格式 (默认: 'mp4v')
                    可选值: 'mp4v', 'avc1', 'X264'等
            fps: 输出视频帧率 (默认: 30)
            
        属性:
            codec: 视频编码格式字符串
            fps: 输出帧率整数
        """
        self.codec = codec
        self.fps = fps
        
    def build(self, frame_paths: List[Union[str, Path]], output_path: Union[str, Path]) -> Path:
        """从帧序列构建视频文件
        
        参数:
            frame_paths: 帧路径列表，按视频顺序排列 (类型: List[str|Path])
            output_path: 输出视频文件路径 (类型: str|Path)
            
        返回:
            Path: 生成的视频文件路径
            
        异常:
            ValueError: 当帧序列为空时抛出
            
        注意:
            使用OpenCV的VideoWriter进行视频写入
            第一帧的尺寸决定输出视频的尺寸
            
        示例:
            >>> builder = VideoBuilder(codec='mp4v', fps=24)
            >>> video = builder.build(["frame1.jpg", "frame2.jpg"], "out.mp4")
            >>> print(f"视频已生成: {video}")
        """
        if not frame_paths:
            raise ValueError("没有可用的帧序列")
            
        # 获取第一帧的尺寸
        sample_frame = cv2.imread(str(frame_paths[0]))
        height, width, _ = sample_frame.shape
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
        
        # 逐帧写入视频
        for frame_path in tqdm(frame_paths, desc="构建视频"):
            frame = cv2.imread(str(frame_path))
            out.write(frame)
            
        out.release()
        return Path(output_path)
