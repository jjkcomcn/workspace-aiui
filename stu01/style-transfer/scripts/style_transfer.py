import torch
import torch.optim as optim
import argparse
import logging
import sys
import time
from pathlib import Path
from models.build_model import get_style_model_and_loss
from utils.image_enhancer import load_image, save_image
from torchvision.models.vgg import VGG19_Weights
from configs.check_environment import print_environment_info

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='神经风格迁移')
    parser.add_argument('--content', type=str, default='picture/1.jpg',
                       help='内容图片路径 (默认: picture/1.jpg)')
    parser.add_argument('--style', type=str, default='picture/2.jpeg',
                       help='风格图片路径 (默认: picture/2.jpeg)')
    parser.add_argument('--output', type=str, default='picture/output.jpg',
                       help='输出图片路径 (默认: picture/output.jpg)')
    parser.add_argument('--size', type=int, default=512,
                       help='图片尺寸 (默认: 512)')
    parser.add_argument('--steps', type=int, default=300,
                       help='训练步数 (默认: 300)')
    parser.add_argument('--style_weight', type=float, default=1e5,
                       help='风格权重 (默认: 1e5)')
    parser.add_argument('--content_weight', type=float, default=1,
                       help='内容权重 (默认: 1)')
    return parser.parse_args()

def validate_paths(content_path: str, style_path: str, output_path: str):
    """验证输入输出路径

    Args:
        content_path: 内容图片路径
        style_path: 风格图片路径
        output_path: 输出图片路径

    Raises:
        FileNotFoundError: 如果路径不存在
    """
    if not Path(content_path).exists():
        logger.error(f"内容图片不存在: {content_path}")
        raise FileNotFoundError(f"内容图片不存在: {content_path}")
    if not Path(style_path).exists():
        logger.error(f"风格图片不存在: {style_path}")
        raise FileNotFoundError(f"风格图片不存在: {style_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

def run_style_transfer():
    """执行风格迁移"""
    args = parse_arguments()
    
    try:
        # 打印环境信息
        print_environment_info()
        
        # 验证路径
        validate_paths(args.content, args.style, args.output)
        
        # 加载图片
        logger.info("加载图片...")
        content_img = load_image(args.content, size=args.size)
        style_img = load_image(args.style, size=args.size)
        
        # 初始化输入图像
        input_img = content_img.clone().requires_grad_(True)
        
        # 记录开始时间
        start_time = time.time()
        
        # 自动检测CUDA，默认使用GPU(如果可用)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content_img, style_img, input_img = content_img.to(DEVICE), style_img.to(DEVICE), input_img.to(DEVICE)
        
        # 加载模型
        logger.info("加载VGG19模型...")
        cnn = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', 
                           weights=VGG19_Weights.IMAGENET1K_V1)
        
        # 构建风格迁移模型
        logger.info("构建风格迁移模型...")
        model, style_losses, content_losses = get_style_model_and_loss(
            cnn, style_img, content_img,
            style_weight=args.style_weight,
            content_weight=args.content_weight
        )
        
        # 优化器
        optimizer = optim.LBFGS([input_img])
        
        # 训练
        logger.info(f"开始训练，共{args.steps}步...")
        for step in range(args.steps):
            def closure():
                optimizer.zero_grad()
                model(input_img)
                
                # 计算损失
                style_score = sum(sl.loss for sl in style_losses)
                content_score = sum(cl.loss for cl in content_losses)
                loss = style_score + content_score
                loss.backward()
                
                # 记录进度
                if step % 50 == 0:
                    logger.info(f"Step {step}: Style Loss: {style_score.item():.4f}, "
                             f"Content Loss: {content_score.item():.4f}")
                return loss
                
            optimizer.step(closure)
        
        # 保存结果
        logger.info(f"保存结果到 {args.output}")
        save_image(args.output, input_img)
        
        # 计算总耗时
        end_time = time.time()
        logger.info(f"风格迁移完成! 总耗时: {end_time - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"风格迁移失败: {str(e)}")
        raise

if __name__ == '__main__':
    run_style_transfer()
