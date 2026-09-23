import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta

# Unified SQLite Database file
DB_NAME = "bot_analytics.db"


def hash_password(password: str) -> str:
    """Generates a secure PBKDF2 password hash with a random 32-byte salt."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ":" + key.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies a plain password against the stored salt:hash string."""
    try:
        salt_hex, key_hex = stored_password.split(":")
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(stored_key, new_key)
    except Exception:
        return False


def init_auth_db():
    """Initializes authentication, active session, and rate limiting tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Table 1: Admin Credentials
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Table 2: Active User Sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at DATETIME NOT NULL
        )
    ''')
    
    # Table 3: Login Attempt Rate Limiting
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_attempts (
            ip_address TEXT PRIMARY KEY,
            attempts INTEGER DEFAULT 0,
            last_attempt DATETIME
        )
    ''')
    
    conn.commit()
    conn.close()


def seed_default_admin():
    """Seeds the initial admin account from environment variables if table is empty."""
    admin_user = os.getenv("DEFAULT_ADMIN_USER", "admin")
    admin_pass = os.getenv("DEFAULT_ADMIN_PASS", "AdminPass123!")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM admin_users")
    if cursor.fetchone()[0] == 0:
        pw_hash = hash_password(admin_pass)
        cursor.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
            (admin_user.lower(), pw_hash)
        )
        conn.commit()
    conn.close()


def is_rate_limited(ip_address: str = "127.0.0.1") -> bool:
    """Checks if an IP address has exceeded failed login limits (5 attempts / 15 mins)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT attempts, last_attempt FROM login_attempts WHERE ip_address = ?", (ip_address,))
    row = cursor.fetchone()
    
    if row:
        attempts, last_attempt = row
        last_dt = datetime.strptime(last_attempt, "%Y-%m-%d %H:%M:%S")
        
        # Block if 5 or more failed attempts within 15 minutes
        if datetime.now() - last_dt < timedelta(minutes=15) and attempts >= 5:
            conn.close()
            return True
        elif datetime.now() - last_dt >= timedelta(minutes=15):
            # Reset counter after cooldown
            cursor.execute("UPDATE login_attempts SET attempts = 0 WHERE ip_address = ?", (ip_address,))
            conn.commit()
            
    conn.close()
    return False


def record_failed_attempt(ip_address: str = "127.0.0.1"):
    """Increments failed login counter for a given IP."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO login_attempts (ip_address, attempts, last_attempt)
        VALUES (?, 1, ?)
        ON CONFLICT(ip_address) DO UPDATE SET
            attempts = attempts + 1,
            last_attempt = excluded.last_attempt
    ''', (ip_address, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def clear_failed_attempts(ip_address: str = "127.0.0.1"):
    """Clears failed login attempts upon successful authentication."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM login_attempts WHERE ip_address = ?", (ip_address,))
    conn.commit()
    conn.close()


def authenticate_admin(username: str, password: str) -> bool:
    """Validates provided admin username and password against stored database hashes."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM admin_users WHERE username = ?", (username.lower(),))
    row = cursor.fetchone()
    conn.close()
    
    if row and verify_password(row[0], password):
        return True
    return False


def create_session(username: str) -> str:
    """Generates a secure 24-hour session token and stores it in active_sessions."""
    session_id = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(hours=24)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO active_sessions (session_id, username, expires_at) VALUES (?, ?, ?)",
        (session_id, username.lower(), expires_at.strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    return session_id


def validate_session(session_id: str) -> bool:
    """Validates whether a session ID exists and has not expired."""
    if not session_id:
        return False
        
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT expires_at FROM active_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    
    if row:
        if datetime.now() < datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"):
            conn.close()
            return True
        else:
            # Clean up expired session
            cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            
    conn.close()
    return False


def terminate_session(session_id: str):
    """Deletes an active session (Logout)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
