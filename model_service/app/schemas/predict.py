from pydantic import BaseModel
from typing import List, Dict, Any

# Cấu trúc của một điểm số dự đoán
class PredictionScore(BaseModel):
    class_name: str
    score: float

# Cấu trúc của phần giải thích XAI
class XAIResponse(BaseModel):
    type: str
    heatmap_base64: str

# Cấu trúc response trả về cho /predict
class PredictResponse(BaseModel):
    temp_id: str  # Sẽ được tạo bởi service_api, nhưng ml service nhận và trả về
    input_metadata: Dict[str, Any]
    prediction: PredictionScore
    all_scores: List[PredictionScore]
    xai_explanation: XAIResponse