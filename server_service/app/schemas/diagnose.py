from pydantic import BaseModel
from typing import List, Optional
import datetime

class DiagnosisResponse(BaseModel):
    id: str
    created_at: datetime.datetime
    user_id: str
    image_url: str
    prediction_class: str
    prediction_score: float
    grad_cam_url: str
    shap_status: str
    shap_url: Optional[str] = None

    class Config:
        orm_mode = True