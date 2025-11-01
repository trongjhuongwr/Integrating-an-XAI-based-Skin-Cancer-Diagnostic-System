import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchvision import models
from collections import OrderedDict
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_pretrained_resnet50_model(num_classes=8, device='cpu', repo_id="Sura3607/resnet50_skin_cancer", filename="resnet50_skin_cancer.pth"):
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Loading model weights from: {weights_path}")
    # Tải trọng số đã huấn luyện
    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Pretrained model loaded successfully.")
    model.to(device)
    return model

def predict_prob(pil_image, model, transform, device):
    """
    Dự đoán xác suất cho một ảnh PIL.
    
    Args:
        pil_image (PIL.Image): Ảnh đầu vào.
        model (torch.nn.Module): Mô hình PyTorch đã train.
        transform (transforms.Compose): Bộ transform tiền xử lý.
        device (torch.device): 'cuda' hoặc 'cpu'.

    Returns:
        np.ndarray: Một mảng 1D chứa xác suất cho mỗi lớp.
    """
    model.eval()
    
    input_tensor = transform(pil_image)
    
    input_batch = input_tensor.unsqueeze(0)
    
    # Đưa lên device
    input_batch = input_batch.to(device)

    with torch.no_grad():
        logits = model(input_batch)
        probabilities = F.softmax(logits, dim=1)
        
    return probabilities.cpu().numpy().squeeze()
