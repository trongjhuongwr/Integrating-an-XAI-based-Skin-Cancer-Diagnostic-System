from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
import json
import torch
import logging

from app.schemas.predict import PredictResponse
from app.models import predictor
from app.core.config import Settings
from app.dependencies import get_model, get_app_settings, get_device

router = APIRouter()
logger = logging.getLogger("app.routes.predict")

@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    image: UploadFile = File(...),
    metadata: str = Form(...),
    model: torch.nn.Module = Depends(get_model),
    settings: Settings = Depends(get_app_settings),
    device: torch.device = Depends(get_device)
):

    try:
        image_bytes = await image.read()
        
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata is not valid JSON.")
            
        class_mapping = json.loads(settings.CLASS_MAPPING_JSON)

        # Chạy dự đoán và XAI
        response_data = predictor.run_prediction_and_xai(
            model=model,
            image_bytes=image_bytes,
            metadata=metadata_dict,
            class_mapping=class_mapping,
            device=device
        )
        
        return response_data

    except Exception as e:
        logger.exception("Error in predict_endpoint: %s", e)
        raise HTTPException(status_code=500, detail="Error in endpoint /predict")