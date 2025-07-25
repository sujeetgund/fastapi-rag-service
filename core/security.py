from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import settings
from core.exceptions import AuthenticationError

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token"""
    if credentials.credentials != settings.BEARER_TOKEN:
        raise AuthenticationError()
    return credentials.credentials