import torch
import torch.nn as nn
from torchvision import models

def create_resnet50_model(num_classes=8):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model