"""Main script supporting both classification and style transfer"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from models.build_model import build_model
from models.loss import get_loss
from utils.load_img import load_image, save_image, load_dataset
from configs import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_classification(model, train_loader, val_loader, epochs, optimizer, criterion):
    """Train classification model"""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        val_acc = evaluate(model, val_loader, criterion)
        
        logger.info(
            f'Epoch {epoch+1}: '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Acc: {train_acc:.4f}, '
            f'Val Acc: {val_acc:.4f}'
        )

def train_style_transfer(content_img, style_img, output_path, steps=500):
    """Train style transfer model"""
    model = build_model(model_type='style_transfer')
    model, style_losses, content_losses = model.build_model(style_img, content_img)
    
    input_img = content_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([input_img], lr=settings.LEARNING_RATE)
    
    with tqdm(total=steps, desc="Style transfer") as pbar:
        for step in range(steps):
            def closure():
                optimizer.zero_grad()
                model(input_img)
                
                style_loss = sum(sl.loss for sl in style_losses)
                content_loss = sum(cl.loss for cl in content_losses)
                total_loss = style_loss + content_loss
                total_loss.backward()
                
                pbar.set_postfix({
                    "style": f"{style_loss.item():.2f}",
                    "content": f"{content_loss.item():.2f}"
                })
                return total_loss
                
            optimizer.step(closure)
            pbar.update(1)
    
    save_image(input_img, output_path)
    logger.info(f"Style transfer result saved to {output_path}")

def evaluate(model, dataloader, criterion):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--model_type', choices=['classification', 'style_transfer'], 
                      default='classification', help='Training mode')
    parser.add_argument('--content', help='Content image path (style transfer only)')
    parser.add_argument('--style', help='Style image path (style transfer only)')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--size', type=int, default=512, help='Image size')
    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/output", exist_ok=True)
    
    if args.model_type == 'classification':
        model = build_model(model_type='classification')
        optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
        criterion = get_loss()
        
        train_loader, val_loader = load_dataset(
            batch_size=args.batch_size,
            img_size=(args.size, args.size)
        )
        
        train_classification(
            model, train_loader, val_loader,
            args.epochs, optimizer, criterion
        )
        
    elif args.model_type == 'style_transfer':
        content_img = load_image(args.content, size=args.size)
        style_img = load_image(args.style, size=args.size)
        output_path = Path(args.output or f"data/output/result_{timestamp}.jpg")
        
        train_style_transfer(
            content_img, style_img,
            output_path, steps=args.epochs
        )

if __name__ == '__main__':
    main()
