import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from lime import lime_image
from skimage.segmentation import mark_boundaries



def explain_with_GRAD_CAM(input_tensor, model, target_layer, target_class_index):
    input_batch = input_tensor.unsqueeze(0) # Shape: [1, C, H, W]
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Định nghĩa mục tiêu (lớp cụ thể)
    targets = [ClassifierOutputTarget(target_class_index)]
    

    grayscale_cam = cam(input_tensor=input_batch, targets=targets)
    
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam


def explain_with_LIME(pil_image, model, transform, device):
    def predict_fn_for_lime(numpy_images):
        model.eval()
        
        batch_tensors = []
        for img_np in numpy_images:
            img_pil = Image.fromarray(img_np.astype('uint8'), 'RGB') 
            batch_tensors.append(transform(img_pil))
            
        # Stack thành batch
        input_batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = model(input_batch)
            probabilities = F.softmax(logits, dim=1)
            
        return probabilities.cpu().numpy()
    
    
    # Khởi tạo explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Chuyển ảnh PIL sang numpy (H, W, C)
    image_np = np.array(pil_image.convert('RGB'))

    explanation = explainer.explain_instance(
        image=image_np, 
        classifier_fn=predict_fn_for_lime, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000 
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, # Chỉ hiện vùng positive
        num_features=5, # Hiện 5 vùng quan trọng nhất
        hide_rest=False 
    )

    lime_image_result = mark_boundaries(temp / 255.0, mask)
    
    return lime_image_result