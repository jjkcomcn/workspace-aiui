# 高级图片风格迁移工具

基于PyTorch实现的神经风格迁移(NST)系统，采用VGG19网络提取特征，通过优化内容与风格的组合，实现艺术风格转换。

## 核心功能

- 高质量风格迁移：保留内容结构的同时完美融合艺术风格
- 多风格支持：油画、水彩、素描等多种艺术风格
- 参数精细控制：
  - 内容/风格权重调节
  - 输出尺寸控制
  - 迭代次数设置
- 性能优化：
  - 支持GPU加速(CUDA)
  - 多线程图像处理
  - 内存高效利用
- 扩展接口：
  - 完整的Python API
  - 命令行工具
  - 可集成到其他应用

## 安装指南

### 基础安装
```bash
git clone https://github.com/yourusername/style-transfer.git
cd style-transfer
pip install -r requirements.txt
```

### GPU加速支持(可选)
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

## 使用示例

### 基础用法
```bash
python scripts/run_main.py \
    --content data/input/photo.jpg \
    --style data/styles/starry_night.jpg \
    --output data/output/result.jpg
```

### 高级控制
```bash
python scripts/run_main.py \
    --content content.jpg \
    --style style.jpg \
    --output output.jpg \
    --size 1024 \          # 输出尺寸
    --steps 1000 \         # 迭代次数
    --style_weight 5e4 \   # 风格权重
    --content_weight 1 \   # 内容权重
    --init random \        # 初始化方式
    --device cuda          # 使用GPU
```

### 批量处理
```bash
# 处理目录下所有图片
for img in data/input/*.jpg; do
    python scripts/run_main.py \
        --content "$img" \
        --style data/styles/monet.jpg \
        --output "data/output/$(basename "$img")"
done
```

## 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--content` | str | 必填 | 内容图片路径 |
| `--style` | str | 必填 | 风格图片路径 |
| `--output` | str | 必填 | 输出图片路径 |
| `--size` | int | 512 | 输出图片长边尺寸(px) |
| `--steps` | int | 500 | 优化迭代次数 |
| `--style_weight` | float | 1e5 | 风格损失权重 |
| `--content_weight` | float | 1 | 内容损失权重 |
| `--init` | str | 'content' | 初始化方式(content/random) |
| `--device` | str | 'cpu' | 计算设备(cpu/cuda) |

## 效果展示

### 示例1: 梵高风格
```
内容图片 + 星月夜风格 = 艺术效果
```

### 示例2: 水彩风格
```
照片 + 水彩风格 = 水彩画效果
```

### 示例3: 素描风格
```
人像 + 素描风格 = 素描效果
```

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
