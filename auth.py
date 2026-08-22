"""Authentication helpers and FastAPI dependencies."""
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from psycopg2.extras import RealDictCursor

from config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY
from db import get_db_connection
from models import UserProfile


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plaintext password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, email: str) -> str:
    """Create a signed JWT access token."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expires_at}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _build_user_profile(row: dict) -> UserProfile:
    return UserProfile(
        id=str(row["id"]),
        email=str(row["email"]),
        full_name=row.get("full_name"),
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


def get_user_by_email(email: str) -> UserProfile | None:
    """Fetch user profile by email."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, email, full_name, created_at, updated_at
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _build_user_profile(row)


def get_user_with_password(email: str) -> dict | None:
    """Fetch user row including password hash."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, email, full_name, hashed_password, created_at, updated_at
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            return cur.fetchone()


def create_user(email: str, hashed_password: str, full_name: str | None) -> UserProfile:
    """Create a user and return profile."""
    with get_db_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, hashed_password, full_name)
                    VALUES (%s, %s, %s)
                    RETURNING id, email, full_name, created_at, updated_at
                    """,
                    (email, hashed_password, full_name),
                )
                row = cur.fetchone()
                conn.commit()
                if row is None:
                    raise RuntimeError("Failed to create user.")
                return _build_user_profile(row)
        except Exception:
            conn.rollback()
            raise


def get_user_by_id(user_id: str) -> UserProfile | None:
    """Fetch user profile by id."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, email, full_name, created_at, updated_at
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _build_user_profile(row)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> UserProfile:
    """Resolve current user from bearer token."""
    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret key is not configured.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    user = get_user_by_id(str(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    return user
