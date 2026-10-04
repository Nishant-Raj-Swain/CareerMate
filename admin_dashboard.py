import os
import sqlite3
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from pydantic import BaseModel

from backend_updates.auth import (
    authenticate_admin,
    create_session,
    validate_session,
    terminate_session,
    is_rate_limited,
    record_failed_attempt,
    clear_failed_attempts,
)

router = APIRouter()

# --- Request / Response Models ---
class LoginRequest(BaseModel):
    username: str
    password: str


# --- Database Helper ---
def get_analytics_db_connection():
    db_path = os.getenv("ANALYTICS_DB_PATH", "bot_analytics.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# --- IP Resolution Helper ---
def get_client_ip(request: Request) -> str:
    """Safely resolves client IP address, checking standard proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


# --- FastAPI Dependency for Route Protection ---
async def require_admin(authorization: Optional[str] = Header(None)) -> str:
    """
    Validates dynamic Bearer token from the Authorization header.
    Returns the authenticated username or raises 401 Unauthorized.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header scheme."
        )
    
    token = authorization.split("Bearer ")[1].strip()
    username = validate_session(token)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalid or expired."
        )
    return username


# --- Authentication Endpoints ---

@router.post("/api/auth/login")
async def login(credentials: LoginRequest, request: Request):
    """
    Authenticates admin credentials using hashed SQLite verification with 
    dual-layer rate limiting (IP + username).
    """
    client_ip = get_client_ip(request)
    username_clean = credentials.username.strip().lower()

    # 1. Check rate-limit block on IP or account
    if is_rate_limited(client_ip) or is_rate_limited(username_clean):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again in 15 minutes."
        )

    # 2. Verify credentials via auth module
    if not authenticate_admin(username_clean, credentials.password):
        record_failed_attempt(client_ip)
        record_failed_attempt(username_clean)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    # 3. Clear failed counter & issue dynamic session token
    clear_failed_attempts(client_ip)
    clear_failed_attempts(username_clean)

    session_token = create_session(username_clean, duration_hours=24)
    return {
        "status": "success",
        "token": session_token,
        "token_type": "bearer",
        "username": username_clean
    }


@router.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """Terminates active session in the database upon logout."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
        terminate_session(token)
    return {"status": "success", "message": "Logged out successfully."}


@router.get("/api/auth/verify")
async def verify_token(current_user: str = Depends(require_admin)):
    """Validates token health for frontend route guards."""
    return {"status": "valid", "username": current_user}


# --- Analytics Logs Endpoint ---

@router.get("/api/analytics/logs")
async def get_analytics_logs(current_user: str = Depends(require_admin)) -> List[Dict[str, Any]]:
    """
    Returns recent user logs recorded in bot_analytics.db.
    Requires a valid admin session.
    """
    try:
        conn = get_analytics_db_connection()
        cursor = conn.cursor()

        # Ensure legacy user_logs table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                whatsapp_no TEXT NOT NULL,
                command TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        rows = cursor.execute(
            "SELECT whatsapp_no, command, timestamp FROM user_logs ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()

        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}"
        )
