#!/usr/bin/env python3
"""视频风格迁移主程序

使用示例:
python main.py --video input.mp4 --style style.jpg --output output.mp4
"""
import argparse
import logging
from pathlib import Path

from core.frame_extractor import FrameExtractor
from core.style_applier import StyleApplier
from core.video_builder import VideoBuilder
from config.settings import (
    DEFAULT_PATHS,
    VIDEO_CONFIG,
    STYLE_TRANSFER_CONFIG,
    LOG_CONFIG
)
from utils.file_io import ensure_dir
from utils.progress import ProgressTracker

# 配置日志
logging.basicConfig(
    level=LOG_CONFIG['level'],
    format=LOG_CONFIG['format']
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频风格迁移')
    parser.add_argument('--video', required=True, 
                       help='输入视频路径')
    parser.add_argument('--style', required=True,
                       help='风格图片路径')
    parser.add_argument('--output', required=True,
                       help='输出视频路径')
    parser.add_argument('--fps', type=int, 
                       default=VIDEO_CONFIG['default_fps'],
                       help=f'处理帧率(默认: {VIDEO_CONFIG["default_fps"]})')
    parser.add_argument('--temp_dir', 
                       default=DEFAULT_PATHS['temp_dir'],
                       help=f'临时目录(默认: {DEFAULT_PATHS["temp_dir"]})')
    parser.add_argument('--max_frames', type=int,
                       default=VIDEO_CONFIG['max_frames'],
                       help=f'最大处理帧数(默认: {VIDEO_CONFIG["max_frames"]})')
    return parser.parse_args()

def main():
    """主处理流程"""
    args = parse_args()
    
    try:
        # 准备目录
        temp_dir = ensure_dir(args.temp_dir)
        frame_dir = ensure_dir(temp_dir / "original_frames")
        styled_dir = ensure_dir(temp_dir / "styled_frames")
        
        # 1. 提取视频帧
        logger.info("开始提取视频帧...")
        extractor = FrameExtractor(frame_dir)
        with ProgressTracker("提取视频帧") as progress:
            frames = extractor.extract(args.video, args.fps)[:args.max_frames]
            progress.update(len(frames))
        logger.info(f"成功提取 {len(frames)} 帧")
        
        # 2. 应用风格迁移
        logger.info("开始应用风格迁移...")
        applier = StyleApplier(args.style)
        with ProgressTracker("风格迁移") as progress:
            styled_frames = applier.apply(frames, styled_dir)
            progress.update(len(styled_frames))
        logger.info("风格迁移完成")
        
        # 3. 重建视频
        logger.info("开始重建视频...")
        builder = VideoBuilder(
            codec=VIDEO_CONFIG['codec'],
            fps=args.fps
        )
        with ProgressTracker("重建视频") as progress:
            output_path = builder.build(styled_frames, args.output)
            progress.update(len(styled_frames))
        logger.info(f"视频重建完成: {output_path}")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

if __name__ == '__main__':
    main()
