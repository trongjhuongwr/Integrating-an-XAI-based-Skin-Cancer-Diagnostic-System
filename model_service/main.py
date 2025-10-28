import logging
import torch
import json
from fastapi import FastAPI , HTTPException, Request
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
    """
    Hàm này được gọi khi FastAPI khởi động.
    Nó tải settings, download model từ Hugging Face, và load model vào bộ nhớ.
    """
    # Configure basic logging for the ML service if not already configured by the
    # server process (uvicorn). Using INFO as default level.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_service.main")

    logger.info("--- Đang tải model khi khởi động ---")
    settings = get_settings()
    app.state.settings = settings
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    logger.info(f"Sử dụng thiết bị: {device}")

    try:
        # Tải file model từ Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=settings.HUGGING_FACE_REPO_ID,
            filename=settings.MODEL_FILE_NAME
        )
        logger.info(f"Model đã được tải về tại: {model_path}")

        # 1. Khởi tạo kiến trúc model
        model = create_resnet50_model(num_classes=settings.NUM_CLASSES)

        # 2. Tải state_dict
        state_dict = torch.load(model_path, map_location=device)

        # 3. Xử lý state_dict nếu nó được lưu từ nn.DataParallel (có prefix 'module.')
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # Bỏ 7 ký tự 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            logger.info("Đã xử lý state_dict từ DataParallel.")

        # 4. Load state_dict đã xử lý vào model
        model.load_state_dict(state_dict)

        # 5. Chuyển model sang device và đặt ở chế độ eval
        model.to(device)
        model.eval()

        # 6. Lưu model vào state của app
        app.state.model = model
        logger.info("--- Model đã được tải và khởi tạo thành công ---")

    except Exception as e:
        logger.exception("LỖI: Không thể tải model.")



app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ML service is running"}