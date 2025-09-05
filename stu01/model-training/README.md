# 模型训练项目

这是一个用于训练深度学习模型的Python项目，支持计算机视觉相关的模型训练和评估。

## 项目结构
```
model-training/
├── configs/               # 配置文件
│   ├── check_environment.py  # 环境检查脚本
│   └── settings.py        # 项目设置
├── data/                  # 数据目录
│   ├── input/             # 原始输入数据
│   ├── processed/         # 预处理后的数据
│   └── output/            # 训练输出
├── models/                # 模型相关
│   ├── build_model.py     # 模型构建
│   └── loss.py            # 损失函数
├── scripts/               # 运行脚本
│   └── run_main.py        # 主运行脚本
└── utils/                 # 工具函数
    ├── image_enhancer.py  # 图像增强
    └── load_img.py        # 图像加载
```

## 功能特性
- 支持多种CNN模型架构
- 完整的数据预处理和增强流程
- 训练过程实时可视化
- 模型评估和导出功能
- 支持GPU加速训练
- 可配置的训练参数

## 安装指南

1. 确保Python 3.8+环境
2. 推荐使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. 安装依赖：
```bash
pip install -r requirements.txt
```
4. 验证安装：
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 配置说明

编辑`configs/settings.py`配置以下参数：
- 数据路径
- 模型类型
- 训练参数(epochs, batch_size等)
- 输出目录

## 使用流程

1. 准备数据：
   - 将训练数据放入`data/input/`目录
   - 支持JPG/PNG格式图像

2. 开始训练：
```bash
# 从项目根目录运行
cd stu01/model-training
python -m scripts.run_main
```

3. 训练进度：
   - 每个epoch会显示训练损失和验证准确率
   - 进度条显示当前epoch/总epochs
   - 预计剩余时间显示

4. 结果保存：
   - 训练日志：`data/output/logs/training_[timestamp].log`
     - 包含训练参数、每个epoch的指标
   - 模型权重：`data/output/models/model_[timestamp].h5`
     - 包含模型结构和参数
   - 训练曲线：`data/output/plots/training_curve.png`
     - 损失和准确率变化曲线

## 高级用法

- 自定义模型：修改`models/build_model.py`
- 自定义数据增强：修改`utils/image_enhancer.py`
- 多GPU训练：设置`CUDA_VISIBLE_DEVICES`环境变量

## 常见问题

Q: 如何更改训练设备(CPU/GPU)?
A: 在`configs/settings.py`中设置`device`参数

Q: 训练中断后如何恢复?
A: 使用`--resume`参数指定检查点路径
