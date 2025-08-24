#!/usr/bin/env python3
"""主运行脚本，实现图片风格迁移功能

使用说明:
    python run_main.py --content <内容图片路径> --style <风格图片路径> --output <输出路径>

示例:
    python run_main.py --content data/input/1.jpg --style data/styles/1.jpg --output data/output/result.jpg
    python run_main.py --content data/input/1.jpg --style data/styles/1.jpg --output data/output/result.jpg --size 512 --steps 500
"""

import argparse
import logging
from pathlib import Path
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import models

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))
logging.info(f"Current working directory: {sys.path}")
# 导入项目模块
from models.build_model import get_style_model_and_loss
from utils.image_enhancer import load_image, save_image
from configs.check_environment import check_environment
from configs.settings import DEFAULT_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析并验证命令行参数
    
    返回:
        argparse.Namespace: 包含以下属性的对象:
            - content (str): 内容图片路径(必须存在)
            - style (str): 风格图片路径(必须存在)
            - output (str): 输出图片保存路径
            - size (int): 图片尺寸(默认从settings.py读取)
            - steps (int): 优化步数(默认从settings.py读取)
            - style_weight (float): 风格损失权重(默认从settings.py读取)
            - content_weight (float): 内容损失权重(默认从settings.py读取)
            
    异常:
        SystemExit: 当参数验证失败时退出程序
    """
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True,
                      help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                      help='Path to style image')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save output image')
    parser.add_argument('--size', type=int, default=DEFAULT_CONFIG['IMAGE_SIZE'],
                      help=f'Image size (default: {DEFAULT_CONFIG["IMAGE_SIZE"]})')
    parser.add_argument('--steps', type=int, default=DEFAULT_CONFIG['OPTIMIZER_STEPS'],
                      help=f'Number of optimization steps (default: {DEFAULT_CONFIG["OPTIMIZER_STEPS"]})')
    parser.add_argument('--style_weight', type=float, default=DEFAULT_CONFIG['STYLE_WEIGHT'],
                      help=f'Style weight (default: {DEFAULT_CONFIG["STYLE_WEIGHT"]})')
    parser.add_argument('--content_weight', type=float, default=DEFAULT_CONFIG['CONTENT_WEIGHT'],
                      help=f'Content weight (default: {DEFAULT_CONFIG["CONTENT_WEIGHT"]})')
    return parser.parse_args()

def run_style_transfer(content_img, style_img, input_img, model, style_losses, content_losses, num_steps=300):
    """执行神经风格迁移优化过程
    
    参数:
        content_img (Tensor): 内容图像张量 [1, C, H, W]
        style_img (Tensor): 风格图像张量 [1, C, H, W]
        input_img (Tensor): 输入/生成图像张量 [1, C, H, W]
        model (nn.Sequential): 包含损失计算的风格迁移模型
        style_losses (list): 风格损失计算模块列表
        content_losses (list): 内容损失计算模块列表
        num_steps (int): 优化迭代次数
        
    返回:
        Tensor: 风格迁移后的图像张量 [1, C, H, W]
        
    说明:
        1. 使用LBFGS优化器进行图像优化
        2. 每50步打印一次损失值
        3. 最终裁剪像素值到[0,1]范围
    """
    # 配置LBFGS优化器参数
    optimizer = optim.LBFGS([input_img.requires_grad_()], 
                          lr=0.2,        # 学习率 - 控制参数更新步长
                          max_iter=100,  # 每次优化迭代的最大内部迭代次数
                          tolerance_grad=1e-10,  # 梯度变化容差 - 更小的值提高精度
                          tolerance_change=1e-10, # 参数变化容差 - 更小的值提高精度
                          history_size=100)  # 存储的历史更新步数 - 影响内存使用
    
    logger.info('Starting optimization...')
    
    # 创建进度条
    # 创建进度条 - 总步数为num_steps*2因为每次优化迭代包含多次内部迭代
    progress = tqdm(range(num_steps * 2), desc="Style Transfer Progress", unit="step")
    
    for step in progress:
        def closure():
            """LBFGS优化器的闭包函数，计算损失并执行反向传播
            
            说明:
                1. 裁剪输入图像像素值到[0,1]范围
                2. 计算风格损失和内容损失
                3. 每50步打印一次损失信息
            """
            # 确保像素值在合理范围内(0-1)
            input_img.data.clamp_(0, 1)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 - 计算特征和损失
            model(input_img)
            
            # 计算总风格损失(所有风格层的加权和)
            style_score = sum(sl.loss for sl in style_losses)
            # 计算总内容损失(所有内容层的加权和)
            content_score = sum(cl.loss for cl in content_losses)
            # 计算加权总损失
            total_loss = style_score + content_score
            
            # 反向传播计算梯度
            total_loss.backward()
            
            # 每50步打印一次损失信息
            if step % 50 == 0:
                logger.info(
                    f"Step {step}/{num_steps} | "
                    f"Style Loss: {style_score.item():.4f} | "
                    f"Content Loss: {content_score.item():.4f}"
                )
                # 更新进度条显示
                progress.set_postfix({
                    "Style Loss": f"{style_score.item():.4f}",
                    "Content Loss": f"{content_score.item():.4f}"
                })
            
            return total_loss  # 返回损失值供优化器使用
        
        optimizer.step(closure)
    
    # 最终裁剪
    input_img.data.clamp_(0, 1)
    return input_img

def main():
    """风格迁移主函数
    
    执行流程:
        1. 解析命令行参数
        2. 检查运行环境(GPU/CPU)
        3. 加载并预处理输入图像
        4. 初始化风格迁移模型
        5. 执行风格迁移优化
        6. 保存结果图像
        
    异常处理:
        - 输入文件不存在时退出
        - 图像加载失败时抛出异常
        - 模型加载失败时退出
        - 风格迁移过程中出错时记录日志并抛出异常
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查运行环境(GPU是否可用)
    available, env_info = check_environment()  # 返回环境信息和GPU可用状态
    logger.info(env_info)
    if not available:
        logger.warning("Environment not properly configured for GPU acceleration! Falling back to CPU mode.")
        # 打印环境警告信息
        print("\n=== Environment Warning ===")
        print(env_info)
        print("\nWill fall back to CPU mode.")
        print("For better performance, you can:")
        print("1. Install NVIDIA driver from http://www.nvidia.com/Download/index.aspx")
        print("2. Install matching CUDA toolkit")
    
    try:
        # 验证输入文件是否存在
        # 检查内容和风格图像路径
        for path in [args.content, args.style]:
            if not Path(path).exists():
                logger.error(f"Input file not found: {path}")
                return  # 文件不存在时直接退出

        # 加载内容图像并转换为float32张量
        logger.info(f'Loading content image from {args.content}')
        content_img = load_image(args.content, args.size).to(torch.float32)  # 确保使用float32类型
        
        logger.info(f'Loading style image from {args.style}')
        style_img = load_image(args.style, args.size).to(torch.float32)
        
        # 调整风格图像尺寸与内容图像完全匹配
        # 如果尺寸不匹配，使用双线性插值调整风格图像尺寸
        if content_img.shape != style_img.shape:
            logger.info(f'Resizing style image from {style_img.shape} to match content image {content_img.shape}')
            # 使用双线性插值保持图像质量
            style_img = torch.nn.functional.interpolate(
                style_img, 
                size=(content_img.shape[2], content_img.shape[3]),  # 匹配高度和宽度
                mode='bilinear',  # 双线性插值
                align_corners=False
            )

        # 确保输出目录存在，不存在则创建
        output_dir = Path(args.output).parent
        if not output_dir.exists():
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录
        
        # 初始化输入图像(使用内容图像作为起点)并确保内存连续布局
        input_img = content_img.clone().contiguous()  # 克隆并确保内存连续
        
        # 加载预训练的VGG19模型(用于特征提取)
        logger.info('Loading VGG19 model...')
        cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()  # 使用预训练权重并设为评估模式
        
        # 构建风格迁移模型(包含内容和风格损失计算)
        logger.info('Building style transfer model...')
        model, style_losses, content_losses = get_style_model_and_loss(
            cnn,  # VGG19模型
            style_img,  # 风格图像
            content_img,  # 内容图像
            style_weight=args.style_weight,  # 风格权重
            content_weight=args.content_weight  # 内容权重
        )
        
        # 执行风格迁移优化过程
        logger.info(f'Running style transfer for {args.steps} steps...')
        output = run_style_transfer(
            content_img,  # 内容图像
            style_img,  # 风格图像
            input_img,  # 输入/生成图像
            model,  # 风格迁移模型
            style_losses,  # 风格损失计算模块
            content_losses,  # 内容损失计算模块
            args.steps  # 迭代步数
        )
        
        # 保存生成的风格迁移图像
        logger.info(f'Saving output to {args.output}')
        save_image(args.output, output)  # 使用image_enhancer保存图像
        logger.info('Style transfer completed successfully!')
        
    except Exception as e:
        # 捕获并记录所有异常
        logger.error(f'Error during style transfer: {str(e)}')
        raise  # 重新抛出异常

if __name__ == '__main__':
    main()
