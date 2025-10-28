from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    HUGGING_FACE_REPO_ID: str
    MODEL_FILE_NAME: str
    NUM_CLASSES: int
    CLASS_MAPPING_JSON: str
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

@lru_cache()
def get_settings():
    return Settings()