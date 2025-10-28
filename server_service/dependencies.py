from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client
from authlib.integrations.starlette_client import UserInfo
from app.core.config import settings
import os

def get_supabase_service_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/signin")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        anon_key = os.environ.get("SUPABASE_ANON_KEY") 
        supabase_auth_client = create_client(settings.SUPABASE_URL, anon_key)
        
        supabase_auth_client.auth.set_session(access_token=token, refresh_token="")
        
        user_response = supabase_auth_client.auth.get_user()
        
        user = user_response.user
        if not user:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid user")
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )