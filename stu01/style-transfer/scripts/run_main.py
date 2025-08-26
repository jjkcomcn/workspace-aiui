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
                          lr=0.05,       # 降低学习率
                          max_iter=20,   # 减少内部迭代次数
                          tolerance_grad=1e-6,   # 放宽梯度容差
                          tolerance_change=1e-6, # 放宽参数变化容差
                          history_size=50)      # 减少历史记录
    
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        2. 配置CPU优化参数
        3. 检查运行环境(GPU/CPU)
        4. 加载并预处理输入图像
        5. 初始化风格迁移模型
        6. 执行风格迁移优化
        7. 保存结果图像
        
    异常处理:
        - 输入文件不存在时退出
        - 图像加载失败时抛出异常
        - 模型加载失败时退出
        - 风格迁移过程中出错时记录日志并抛出异常
    """
    # 解析命令行参数
    args = parse_args()
    
    # 配置CPU优化参数
    import torch
    import os
    from configs.settings import DEFAULT_CONFIG
    
    # 设置CPU线程数
    torch.set_num_threads(DEFAULT_CONFIG['CPU_THREADS'])
    logger.info(f"Using {DEFAULT_CONFIG['CPU_THREADS']} CPU threads")
    
    # 配置CPU调度优化
    if DEFAULT_CONFIG['CPU_AFFINITY']:
        try:
            import psutil
            p = psutil.Process()
            cores = list(range(DEFAULT_CONFIG['CPU_THREADS']))
            p.cpu_affinity(cores)
            
            # 根据调度策略设置线程优先级
            if DEFAULT_CONFIG['CPU_SCHED_POLICY'] == 'throughput':
                # 吞吐量优先模式
                for core in cores:
                    psutil.Process().cpu_affinity([core])
                    psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                # 延迟优先模式
                for core in cores:
                    psutil.Process().cpu_affinity([core])
                    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)
            
            logger.info(f"Set CPU affinity to cores {cores} with {DEFAULT_CONFIG['CPU_SCHED_POLICY']} policy")
            
            # 设置线程自旋等待时间
            if hasattr(psutil, 'thread_spin_time'):
                psutil.thread_spin_time(DEFAULT_CONFIG['THREAD_SPIN_TIME'])
        except ImportError:
            logger.warning("psutil not installed, cannot optimize CPU scheduling")
    
    # 配置并行处理
    if DEFAULT_CONFIG['USE_PARALLEL']:
        torch.set_float32_matmul_precision('high')
        logger.info("Enabled parallel processing")
    
    # 内存优化配置
    if DEFAULT_CONFIG['MEMORY_OPTIMIZATION']:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled memory optimization")
    
    # 设置数据加载器参数
    torch.set_num_interop_threads(DEFAULT_CONFIG['NUM_WORKERS'] or 1)
    
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

        # 优化图像数据加载
        logger.info('Optimizing image loading...')
        with torch.no_grad():
            # 使用多线程加载图像
            if DEFAULT_CONFIG['NUM_WORKERS'] > 0:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=DEFAULT_CONFIG['NUM_WORKERS']) as executor:
                    content_future = executor.submit(load_image, args.content, args.size)
                    style_future = executor.submit(load_image, args.style, args.size)
                    content_img = content_future.result().to(torch.float32)
                    style_img = style_future.result().to(torch.float32)
                    
                    # 确保多线程加载后形状正确
                    if content_img.dim() != 4:
                        content_img = content_img.unsqueeze(0)
                    if style_img.dim() != 4:
                        style_img = style_img.unsqueeze(0)
            else:
                content_img = load_image(args.content, args.size).to(torch.float32)
                style_img = load_image(args.style, args.size).to(torch.float32)
            
            # 优化内存布局
            content_img = content_img.cpu().contiguous()
            style_img = style_img.cpu().contiguous()
            
            # 确保张量形状正确 [1, C, H, W]
            content_img = content_img.unsqueeze(0) if content_img.dim() == 3 else content_img
            style_img = style_img.unsqueeze(0) if style_img.dim() == 3 else style_img
        
        # 优化图像尺寸匹配处理
        if content_img.shape != style_img.shape:
            logger.info(f'Optimizing style image resizing from {style_img.shape} to {content_img.shape}')
            # 使用更高效的插值方法
            with torch.no_grad():
                style_img = torch.nn.functional.interpolate(
                    style_img,
                    size=(content_img.shape[2], content_img.shape[3]),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True  # 启用抗锯齿以获得更好质量
                ).contiguous()  # 确保内存连续

        # 并行创建输出目录和初始化输入图像
        output_dir = Path(args.output).parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # 优化输入图像初始化(支持批处理)
        with torch.no_grad():
            input_img = torch.cat([content_img] * DEFAULT_CONFIG['BATCH_SIZE'], dim=0).contiguous()
            # 确保内存对齐
            if input_img.stride()[1] % 64 != 0:
                input_img = input_img.contiguous()
        
        # 优化模型加载
        logger.info('Optimizing VGG19 model loading...')
        with torch.no_grad():
            cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
            # 减少模型内存占用
            for param in cnn.parameters():
                param.requires_grad_(False)
        
        # 优化模型构建过程
        logger.info('Optimizing style transfer model building...')
        with torch.no_grad():
            model, style_losses, content_losses = get_style_model_and_loss(
                cnn,
                style_img,
                content_img,
                style_weight=args.style_weight,
                content_weight=args.content_weight
            )

        
        # 优化风格迁移过程
        logger.info(f'Optimizing style transfer for {args.steps} steps...')
        with torch.no_grad():
            # 调整优化器参数以提高CPU性能
            if not torch.cuda.is_available():
                args.steps = min(args.steps, 200)  # CPU上减少迭代次数
                
            # 批处理优化
            output = None
            for i in range(DEFAULT_CONFIG['BATCH_SIZE']):
                current_output = run_style_transfer(
                    content_img,
                    style_img,
                    input_img[i:i+1],
                    model,
                    style_losses,
                    content_losses,
                    args.steps
                )
                if output is None:
                    output = current_output
                else:
                    output = torch.cat([output, current_output], dim=0)
        
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
