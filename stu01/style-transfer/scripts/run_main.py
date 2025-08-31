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
from configs.settings import DEFAULT_CONFIG, DEVICE

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
    parser.add_argument('--content', type=str, 
                       default='data/input/',
                       help='内容图片目录路径 (默认: data/input/)')
    parser.add_argument('--style', type=str, 
                       default='data/styles/style.png',
                       help='风格图片路径 (默认: data/styles/style.png)')
    parser.add_argument('--output', type=str,
                       default='data/output/',
                       help='输出图片保存路径 (默认: data/output/)')
    parser.add_argument('--size', type=int, default=DEFAULT_CONFIG['IMAGE_SIZE'],
                       help=f'图片尺寸(默认: {DEFAULT_CONFIG["IMAGE_SIZE"]})')
    parser.add_argument('--steps', type=int, default=DEFAULT_CONFIG['OPTIMIZER_STEPS'],
                       help=f'优化步数(默认: {DEFAULT_CONFIG["OPTIMIZER_STEPS"]})')
    parser.add_argument('--style_weight', type=float, default=DEFAULT_CONFIG['STYLE_WEIGHT'],
                       help=f'风格损失权重(默认: {DEFAULT_CONFIG["STYLE_WEIGHT"]})')
    parser.add_argument('--content_weight', type=float, default=DEFAULT_CONFIG['CONTENT_WEIGHT'],
                       help=f'内容损失权重(默认: {DEFAULT_CONFIG["CONTENT_WEIGHT"]})')
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
    """
    # 确保input_img是叶子张量
    input_img = input_img.clone().detach().requires_grad_(True)
    
    # 从配置初始化LBFGS优化器
    optimizer = optim.LBFGS(
        [input_img],
        lr=DEFAULT_CONFIG['LEARNING_RATE'],
        max_iter=DEFAULT_CONFIG['LBFGS_MAX_ITER'],
        history_size=DEFAULT_CONFIG['LBFGS_HISTORY_SIZE'],
        line_search_fn='strong_wolfe',
        tolerance_grad=DEFAULT_CONFIG['LBFGS_TOL_GRAD'],
        tolerance_change=DEFAULT_CONFIG['LBFGS_TOL_CHANGE']
    )
    logger.info(f"Optimizer configured with: "
               f"lr={DEFAULT_CONFIG['LEARNING_RATE']}, "
               f"max_iter={DEFAULT_CONFIG['LBFGS_MAX_ITER']}, "
               f"history_size={DEFAULT_CONFIG['LBFGS_HISTORY_SIZE']}")
    
    # 创建进度条(如果配置启用)
    if DEFAULT_CONFIG['PROGRESS_BAR']:
        progress = tqdm(
            range(num_steps * 2),
            desc=DEFAULT_CONFIG['PROGRESS_DESC'],
            unit=DEFAULT_CONFIG['PROGRESS_UNIT']
        )
    else:
        progress = range(num_steps * 2)
    
    for step in progress:
        def closure():
            """LBFGS优化器的闭包函数，计算损失并执行反向传播"""
            optimizer.zero_grad()
            model(input_img)
            
            style_score = torch.tensor(0., device=input_img.device)
            content_score = torch.tensor(0., device=input_img.device)
            for sl in style_losses:
                style_score = style_score + sl.loss
            for cl in content_losses:
                content_score = content_score + cl.loss
            
            loss = style_score + content_score
            # 确保loss是标量张量且保留梯度
            if loss.dim() > 0:
                loss = loss.sum()
            loss = loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [input_img], 
                max_norm=DEFAULT_CONFIG['MAX_GRAD_NORM'],
                norm_type=2
            )
            
            return loss
            
        optimizer.step(closure)
    
    # 使用配置中的范围进行最终裁剪
    input_img.data.clamp_(
        min=DEFAULT_CONFIG['CLIP_MIN'],
        max=DEFAULT_CONFIG['CLIP_MAX']
    )
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
    """
    args = parse_args()
    
    # 配置CPU优化参数
    torch.set_num_threads(DEFAULT_CONFIG['CPU_THREADS'])
    logger.info(f"Using {DEFAULT_CONFIG['CPU_THREADS']} CPU threads")
    
    # 配置CPU调度优化
    if DEFAULT_CONFIG['CPU_AFFINITY']:
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity(list(range(DEFAULT_CONFIG['CPU_THREADS'])))
        except ImportError:
            logger.warning("psutil not available, cannot set CPU affinity")

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
    
    # 检查运行环境
    available, env_info = check_environment()
    logger.info(env_info)
    if not available:
        logger.warning("Environment not properly configured for GPU acceleration! Falling back to CPU mode.")

    try:
        # 验证输入文件/目录是否存在
        if not Path(args.style).exists():
            raise FileNotFoundError(f"Style file not found: {args.style}")
        if not Path(args.content).exists():
            raise FileNotFoundError(f"Content path not found: {args.content}")
        
        # 处理内容目录下的所有图片
        content_dir = Path(args.content)
        content_files = list(content_dir.glob('*.jpg')) + list(content_dir.glob('*.png'))
        
        if not content_files:
            raise FileNotFoundError(f"No image files found in content directory: {args.content}")
        
        # 加载风格图像
        style_img = load_image(args.style, size=args.size)
        
        # 创建输出目录
        output_dir = Path(args.output if args.output.endswith('/') else args.output + '/')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载ResNet50模型(更适合动漫风格)
        cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        for param in cnn.parameters():
            param.requires_grad_(False)
        
        # 调整风格权重到推荐范围
        if not (1e4 <= args.style_weight <= 1e6):
            logger.warning(f"Adjusting style_weight from {args.style_weight} to recommended range")
            args.style_weight = max(1e4, min(1e6, args.style_weight))
        
        # 处理每张内容图片
        for content_file in content_files:
            # 加载内容图像
            content_img = load_image(str(content_file), size=args.size)
            
            # 创建可优化输入图像
            input_img = content_img.clone()
            input_img.requires_grad_(True)
            input_img = input_img.to(DEVICE)
            
            # 构建风格迁移模型
            model, style_losses, content_losses = get_style_model_and_loss(
                cnn, style_img, content_img,
                style_weight=args.style_weight,
                content_weight=args.content_weight
            )
            
            # 执行风格迁移
            output = run_style_transfer(
                content_img, style_img, input_img,
                model, style_losses, content_losses,
                num_steps=args.steps
            )
            
            # 生成输出文件名并保存
            output_file = output_dir / f"{content_file.stem}-01{content_file.suffix}"
            save_image(str(output_file), output)
        logger.info('Style transfer completed successfully!')
        
    except Exception as e:
        logger.error(f'Error during style transfer: {str(e)}')
        raise

if __name__ == '__main__':
    main()
