# 图片风格迁移项目

基于PyTorch实现的神经风格迁移(NST)项目，可以将一张图片的内容与另一张图片的风格相结合。

## 功能特性

- 支持任意尺寸的输入图片
- 可调节内容和风格的权重
- 支持CPU和GPU加速
- 提供完整的API接口

## 安装

1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/style-transfer.git
cd style-transfer
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用示例

### 命令行使用
```bash
python style-transfer/scripts/run_main.py \
    --content style-transfer/data/input/your_content.jpg \
    --style style-transfer/data/styles/your_style.jpg \
    --output style-transfer/data/output/result.jpg \
    --size 512 \
    --steps 500
```

### 参数说明
- `--content`: 内容图片路径
- `--style`: 风格图片路径
- `--output`: 输出图片路径
- `--size`: 输出图片尺寸(长边)
- `--steps`: 优化迭代次数
- `--style_weight`: 风格权重(默认1e5)
- `--content_weight`: 内容权重(默认1)

## 项目结构
```
stu01/style-transfer/
├── configs/               # 配置文件
│   ├── __init__.py
│   └── settings.py
│
├── data/                  # 数据文件
│   ├── input/            # 输入图片
│   ├── output/           # 输出图片
│   └── styles/           # 风格图片
│
├── models/                # 模型相关
│   ├── __init__.py
│   ├── build_model.py    # 模型构建
│   └── loss.py           # 损失函数
│
├── scripts/               # 运行脚本
│   ├── __init__.py
│   └── run_main.py       # 主运行脚本
│
├── tests/                 # 测试
│   ├── __init__.py
│   └── test.py
│
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── image_loader.py    # 图片加载
│   └── image_saver.py     # 图片保存
│
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
```

## 贡献指南

欢迎提交Pull Request或Issue报告问题。

## 许可证

MIT License
