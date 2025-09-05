# 视频风格迁移项目

## 项目简介
该项目实现将艺术风格迁移到视频上的功能，通过深度学习技术将指定的艺术风格应用到输入视频的每一帧，最终生成风格化的视频。

## 功能特点
- 支持多种视频格式输入
- 支持实时摄像头风格迁移
- 可自定义风格图片
- 控制处理帧率和最大帧数
- 实时进度显示和FPS监控
- 完善的日志记录
- GPU加速支持

## 安装要求
- Python 3.8+
- pip 20+

### 推荐安装方式 (可编辑模式)
```bash
# 进入项目目录
cd path/to/video-transfer

# 安装包(开发模式)
pip install -e .

# 或者直接安装依赖
pip install opencv-python torch tqdm
```

### 验证安装
```bash
video-transfer --help
camera-processor --help
```

## 使用说明

### 视频文件处理
```bash
python -m video_transfer.main \
  --video [输入视频路径] \
  --style [风格图片路径] \
  --output [输出视频路径] \
  [--fps 输出帧率] \
  [--max_frames 最大处理帧数]

# 示例
python -m video_transfer.main \
  --video input.mp4 \
  --style style.jpg \
  --output output.mp4 \
  --fps 24 \
  --max_frames 500
```

### 实时摄像头处理
```bash
python -m video_transfer.scripts.camera_processor \
  --style [风格图片路径] \
  [--size 处理宽度(默认640)]

# 示例
python -m video_transfer.scripts.camera_processor \
  --style van_gogh.jpg \
  --size 480
```

### 性能优化建议
1. 降低处理分辨率(--size参数)
2. 使用轻量级风格图片
3. 启用GPU加速(需安装CUDA版本PyTorch)
4. 按'q'键可退出实时处理

## 项目结构
```
video-transfer/
├── __init__.py
├── main.py              # 主程序入口
├── config/
│   └── settings.py      # 参数配置
├── core/                # 核心处理模块
│   ├── frame_extractor.py   # 帧提取
│   ├── style_applier.py     # 风格迁移
│   └── video_builder.py     # 视频重建
└── utils/               # 工具模块
    ├── file_io.py           # 文件操作
    └── progress.py          # 进度显示
```

## 配置说明
可在`config/settings.py`中修改默认配置：
- 默认路径
- 视频编码格式
- 风格迁移参数
- 日志设置

## 注意事项
1. 确保有足够的磁盘空间存放临时帧
2. 处理高分辨率视频需要较大内存
3. 风格迁移过程较耗时，建议先测试小段视频
