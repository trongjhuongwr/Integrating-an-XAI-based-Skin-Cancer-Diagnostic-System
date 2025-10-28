import os
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# --- Cấu hình JWT ---
# Tạo một key bí mật (RẤT QUAN TRỌNG, nên giữ bí mật)
# Bạn có thể tạo bằng: openssl rand -hex 32
SECRET_KEY = os.environ.get("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TokenData(BaseModel):
    username: str | None = None

# --- Cơ chế OAuth2 ---
# Báo cho FastAPI biết nó sẽ lấy token từ URL "/auth/signin"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/signin")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Tạo ra một JWT access token mới."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Giải mã token và trả về thông tin user hiện tại."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # (Vì không có DB) Chúng ta chỉ tạo một đối tượng TokenData
        token_data = TokenData(username=username)
        
    except JWTError:
        raise credentials_exception
    
    # user = get_user_from_db(username=token_data.username)
    # if user is None:
    #     raise credentials_exception
    
    # Vì không có DB, chúng ta chỉ trả về username
    return token_data.username