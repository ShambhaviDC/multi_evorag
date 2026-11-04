# auth.py – authentication routes for EvoRAG

"""FastAPI router handling user signup, login, and token authentication.

The implementation uses:
- **Passlib** (bcrypt) for secure password hashing.
- **Python‑jose** for JWT creation/verification.
- **SQLModel** sessions to store and retrieve ``User`` records.

Google OAuth is left as a stub – integration can be added later using
``authlib.integrations.starlette_client``.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from sqlmodel import Session, select

from models import User
from database import get_session

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Security utilities
# ---------------------------------------------------------------------------

import bcrypt

oauth_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Settings – in a real deployment these would come from .env
JWT_SECRET_KEY = "supersecretkey"  # placeholder; override via env
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8')[:72],
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(
        password.encode('utf-8')[:72],
        bcrypt.gensalt()
    ).decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None


class UserCreate(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None

    class Config:
        from_attributes = True


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    statement = select(User).where(User.email == email)
    result = session.exec(statement).first()
    return result


def authenticate_user(session: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(session, email)
    if not user or not user.hashed_password:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth_scheme), session: Session = Depends(get_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id_raw = payload.get("sub")
        if user_id_raw is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id_raw))
    except (JWTError, ValueError):
        raise credentials_exception
    user = session.get(User, token_data.user_id)
    if user is None:
        raise credentials_exception
    return user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/signup", response_model=UserRead, summary="Create a new account")
def signup(user_in: UserCreate, session: Session = Depends(get_session)):
    # Ensure email uniqueness
    if get_user_by_email(session, user_in.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/login", response_model=Token, summary="Obtain a JWT access token")
async def login(
    request: Request,
    session: Session = Depends(get_session),
):
    username = None
    password = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        username = body.get("username") or body.get("email")
        password = body.get("password")
    else:
        form = await request.form()
        username = form.get("username") or form.get("email")
        password = form.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    user = authenticate_user(session, username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=access_token)


# Add Google OAuth routes
from authlib.integrations.starlette_client import OAuth
import os

# OAuth configuration – expects GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars
oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    client_kwargs={'scope': 'openid email profile'},
)

@router.get('/google/login', summary='Redirect to Google for OAuth')
def google_login(request: Request):
    redirect_uri = request.url_for('google_callback')
    return oauth.google.authorize_redirect(request, redirect_uri)

@router.get('/google/callback', summary='Google OAuth callback')
def google_callback(request: Request, session: Session = Depends(get_session)):
    token = oauth.google.authorize_access_token(request)
    user_info = oauth.google.parse_id_token(request, token)
    google_sub = user_info['sub']
    email = user_info.get('email')

    # Find or create user
    user = session.exec(select(User).where(User.google_sub == google_sub)).first()
    if not user:
        # If a user with this email exists, link the Google account
        user = session.exec(select(User).where(User.email == email)).first()
        if user:
            user.google_sub = google_sub
        else:
            user = User(email=email, full_name=user_info.get('name'), google_sub=google_sub)
        session.add(user)
        session.commit()
        session.refresh(user)

    # Issue JWT
    access_token = create_access_token(data={'sub': str(user.id)})
    response = RedirectResponse(url='/static/chat.html')
    response.set_cookie(key='access_token', value=access_token, httponly=True, max_age=60*60*24*7)
    return response



# Example protected route (can be used for testing)
@router.get("/me", response_model=UserRead, summary="Get current user profile")
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

