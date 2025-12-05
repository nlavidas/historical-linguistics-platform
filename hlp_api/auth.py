"""
HLP API Auth - Authentication Layer

This module provides authentication and authorization for the API.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class UserRole(Enum):
    """User roles"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    ANNOTATOR = "annotator"
    VIEWER = "viewer"


@dataclass
class AuthConfig:
    """Authentication configuration"""
    secret_key: str = "hlp-secret-key-change-in-production"
    
    algorithm: str = "HS256"
    
    access_token_expire_minutes: int = 60
    
    refresh_token_expire_days: int = 7
    
    enable_api_keys: bool = True
    
    enable_jwt: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    
    role: UserRole = UserRole.VIEWER
    
    is_active: bool = True
    
    created_at: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Token:
    """Token model"""
    access_token: str
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token
        }


_users: Dict[str, User] = {}
_tokens: Dict[str, Dict[str, Any]] = {}
_api_keys: Dict[str, str] = {}
_config: AuthConfig = AuthConfig()


def configure_auth(config: AuthConfig):
    """Configure authentication"""
    global _config
    _config = config


def hash_password(password: str) -> str:
    """Hash a password"""
    return hashlib.sha256(
        (password + _config.secret_key).encode()
    ).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return hash_password(plain_password) == hashed_password


def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None
) -> Token:
    """Create an access token"""
    if expires_delta is None:
        expires_delta = timedelta(minutes=_config.access_token_expire_minutes)
    
    expires_at = datetime.now() + expires_delta
    
    token_data = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role.value,
        "exp": expires_at.timestamp()
    }
    
    access_token = secrets.token_urlsafe(32)
    
    _tokens[access_token] = token_data
    
    refresh_token = secrets.token_urlsafe(32)
    
    return Token(
        access_token=access_token,
        expires_at=expires_at,
        refresh_token=refresh_token
    )


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify an access token"""
    token_data = _tokens.get(token)
    
    if not token_data:
        return None
    
    if datetime.now().timestamp() > token_data.get("exp", 0):
        del _tokens[token]
        return None
    
    return token_data


def create_api_key(user_id: str) -> str:
    """Create an API key"""
    api_key = f"hlp_{secrets.token_urlsafe(32)}"
    _api_keys[api_key] = user_id
    return api_key


def verify_api_key(api_key: str) -> Optional[str]:
    """Verify an API key"""
    return _api_keys.get(api_key)


def register_user(
    username: str,
    email: str,
    password: str,
    role: UserRole = UserRole.VIEWER
) -> User:
    """Register a new user"""
    user_id = secrets.token_urlsafe(16)
    
    user = User(
        id=user_id,
        username=username,
        email=email,
        role=role
    )
    
    _users[username] = user
    _users[f"pwd_{username}"] = hash_password(password)
    
    return user


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user"""
    user = _users.get(username)
    
    if not user:
        return None
    
    stored_password = _users.get(f"pwd_{username}")
    
    if not stored_password or not verify_password(password, stored_password):
        return None
    
    if not user.is_active:
        return None
    
    return user


def get_user(user_id: str) -> Optional[User]:
    """Get a user by ID"""
    for user in _users.values():
        if isinstance(user, User) and user.id == user_id:
            return user
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get the current authenticated user"""
    if not credentials:
        return None
    
    token = credentials.credentials
    
    if token.startswith("hlp_"):
        user_id = verify_api_key(token)
        if user_id:
            return get_user(user_id)
    else:
        token_data = verify_token(token)
        if token_data:
            return get_user(token_data.get("user_id"))
    
    return None


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Require authentication"""
    user = await get_current_user(credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def require_role(
    required_roles: List[UserRole],
    user: User = Depends(require_auth)
) -> User:
    """Require specific roles"""
    if user.role not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    return user


def require_admin(user: User = Depends(require_auth)) -> User:
    """Require admin role"""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


def require_researcher(user: User = Depends(require_auth)) -> User:
    """Require researcher role or higher"""
    if user.role not in [UserRole.ADMIN, UserRole.RESEARCHER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher access required"
        )
    return user
