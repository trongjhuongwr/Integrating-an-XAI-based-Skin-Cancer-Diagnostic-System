import torch
import torch.nn as nn
from torchvision import models

def create_resnet50_model(num_classes=8):
    """
    Tạo một mô hình ResNet50 với lớp FC cuối được tùy chỉnh.
    Lưu ý: Không bọc model trong nn.DataParallel ở đây.
    """
    # Tải mô hình ResNet50 đã được huấn luyện trước trên ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
    # Lấy số lượng đặc trưng đầu vào của lớp fully-connected cuối cùng
    num_ftrs = model.fc.in_features
    
    # Thay thế lớp cuối cùng bằng một lớp mới phù hợp với bài toán
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model