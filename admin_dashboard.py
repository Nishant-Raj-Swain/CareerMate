import os
import sqlite3
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

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

# --- Authentication Endpoints ---
@router.post("/api/auth/login")
async def login(credentials: LoginRequest):
    """
    Authenticates admin user credentials.
    Replace 'admin' and 'admin123' or integrate with your auth module as needed.
    """
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")

    if credentials.username == admin_user and credentials.password == admin_pass:
        return {"token": "valid-admin-session-token-789"}
    
    raise HTTPException(status_code=401, detail="Invalid username or password")


@router.post("/api/auth/logout")
async def logout(authorization: str = Header(None)):
    """Handles session termination on backend."""
    return {"status": "Logged out successfully"}


# --- Analytics Logs Endpoint ---
@router.get("/api/analytics/logs")
async def get_analytics_logs(authorization: str = Header(None)):
    """
    Returns analytics logs recorded in bot_analytics.db for the Streamlit dashboard.
    """
    if not authorization or authorization != "Bearer valid-admin-session-token-789":
        raise HTTPException(status_code=401, detail="Unauthorized session")

    try:
        conn = get_analytics_db_connection()
        cursor = conn.cursor()
        
        # Ensure user_logs table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                whatsapp_no TEXT NOT NULL,
                command TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        rows = cursor.execute(
            "SELECT whatsapp_no, command, timestamp FROM user_logs ORDER BY timestamp DESC"
        ).fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
