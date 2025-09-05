"""Model builder supporting both classification and style transfer"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights

class ClassificationModel(nn.Module):
    """Original image classification model"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class StyleTransferModel:
    """Style transfer model builder"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        for param in self.cnn.parameters():
            param.requires_grad_(False)
            
    def build_model(self, style_img, content_img):
        """Build style transfer model with losses"""
        model = nn.Sequential()
        gram = GramMatrix().to(self.device)
        content_losses = []
        style_losses = []
        
        conv_counter = 1
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{conv_counter}'
                model.add_module(name, layer)
                
                if name in settings.CONTENT_LAYERS:
                    target = model(content_img).detach()
                    content_loss = ContentLoss(target, settings.CONTENT_WEIGHT)
                    model.add_module(f'content_loss_{conv_counter}', content_loss)
                    content_losses.append(content_loss)
                
                if name in settings.STYLE_LAYERS:
                    target = model(style_img).detach()
                    target = gram(target)
                    style_loss = StyleLoss(target, settings.STYLE_WEIGHT)
                    model.add_module(f'style_loss_{conv_counter}', style_loss)
                    style_losses.append(style_loss)
                    
                conv_counter += 1
                
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{conv_counter}'
                model.add_module(name, layer)
                
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{conv_counter}'
                model.add_module(name, layer)
                
        return model, style_losses, content_losses

def build_model(model_type='classification', **kwargs):
    """Factory function to build different model types"""
    if model_type == 'classification':
        return ClassificationModel(**kwargs)
    elif model_type == 'style_transfer':
        return StyleTransferModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
