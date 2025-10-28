from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionScore(BaseModel):
    class_name: str
    score: float

class XAIResponse(BaseModel):
    type: str
    heatmap_base64: str

class PredictResponse(BaseModel):
    temp_id: str  
    input_metadata: Dict[str, Any]
    prediction: PredictionScore
    all_scores: List[PredictionScore]
    xai_explanation: XAIResponse