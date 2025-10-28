from pydantic import BaseModel
from typing import List, Dict, Any

class UserAuthRequest(BaseModel):
    username: str
    password: str

class UserAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    