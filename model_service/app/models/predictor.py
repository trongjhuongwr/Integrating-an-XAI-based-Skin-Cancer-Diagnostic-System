import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from io import BytesIO
import numpy as np
import json

from app.schemas.predict import PredictionScore, XAIResponse, PredictResponse
from app.models.xai import run_grad_cam, unnormalize_tensor

IMAGE_SIZE = 224

preprocess_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    tensor = preprocess_transforms(image).unsqueeze(0)
    return tensor.to(device)

def run_prediction_and_xai(model, image_bytes: bytes, metadata: dict, class_mapping: dict, device: torch.device) -> PredictResponse:

    input_tensor = preprocess_image(image_bytes, device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    all_scores = []
    for i, prob in enumerate(probabilities):
        class_name = class_mapping.get(str(i), f"Class_{i}")
        all_scores.append(PredictionScore(class_name=class_name, score=prob.item()))

    all_scores.sort(key=lambda x: x.score, reverse=True)

    top_pred = all_scores[0]
    pred_label_idx = probabilities.argmax().item()

    unnormalized_input_tensor = unnormalize_tensor(input_tensor)
    
    heatmap_base64 = run_grad_cam(
        model=model,
        normalized_tensor=input_tensor,       
        unnormalized_tensor=unnormalized_input_tensor, 
        target_label_idx=pred_label_idx,
        device=device
    )
    # Ensure heatmap_base64 is a string (pydantic requires str).
    if not isinstance(heatmap_base64, str):
        heatmap_base64 = ""

    response = PredictResponse(
        temp_id=metadata.get("temp_id", "unknown"), 
        input_metadata=metadata,
        prediction=top_pred,
        all_scores=all_scores,
        xai_explanation=XAIResponse(
            type="Grad-CAM",
            heatmap_base64=heatmap_base64
        )
    )
    
    return response