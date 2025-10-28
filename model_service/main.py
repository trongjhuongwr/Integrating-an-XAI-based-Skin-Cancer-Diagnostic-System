import logging
import torch
import json
from fastapi import FastAPI , HTTPException, Request, logger
from collections import OrderedDict
from huggingface_hub import hf_hub_download

from app.core.config import get_settings, Settings
from app.models.model import create_resnet50_model
from app.routes import predict

app = FastAPI(title="Skin Cancer ML Service")

# Biến global để giữ model và settings
app.state.model = None
app.state.settings = None
app.state.device = None

@app.on_event("startup")
def load_model_on_startup():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_service.main")

    logger.info("MODEL LOADING...")
    settings = get_settings()
    app.state.settings = settings
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    logger.info(f"USE: {device}")

    try:
        # Tải file model từ Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=settings.HUGGING_FACE_REPO_ID,
            filename=settings.MODEL_FILE_NAME
        )
        logger.info(f"MODEL DOWNLOADED: {model_path}")

        model = create_resnet50_model(num_classes=settings.NUM_CLASSES)

        state_dict = torch.load(model_path, map_location=device)


        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # loại bỏ 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
            logger.info("STATE_DICT PROCESSED.")

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        app.state.model = model
        logger.info("SUCCESS: Model loaded and initialized.")

    except Exception as e:
        logger.exception("ERROR: Unable to load model.")



app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ML service is running"}