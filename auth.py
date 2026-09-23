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


def seed_default_admin():
    """Seeds the initial admin account from environment variables if table is empty."""
    admin_user = os.getenv("DEFAULT_ADMIN_USER", "admin")
    admin_pass = os.getenv("DEFAULT_ADMIN_PASS", "AdminPass123!")
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM admin_users")
        if cursor.fetchone()[0] == 0:
            pw_hash = hash_password(admin_pass)
            cursor.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                (admin_user.lower(), pw_hash)
            )
            conn.commit()


def init_auth_db():
    """Initializes authentication, active session, rate limiting tables, and seeds admin."""
    with sqlite3.connect(DB_NAME) as conn:
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

    # Automatically trigger admin seeding during DB init
    seed_default_admin()


def is_rate_limited(ip_address: str = "127.0.0.1") -> bool:
    """Checks if an IP address has exceeded failed login limits (5 attempts / 15 mins)."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT attempts, last_attempt FROM login_attempts WHERE ip_address = ?", (ip_address,))
        row = cursor.fetchone()
        
        if row:
            attempts, last_attempt = row
            try:
                last_dt = datetime.strptime(last_attempt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return False

            if datetime.now() - last_dt < timedelta(minutes=15) and attempts >= 5:
                return True
            elif datetime.now() - last_dt >= timedelta(minutes=15):
                cursor.execute("UPDATE login_attempts SET attempts = 0 WHERE ip_address = ?", (ip_address,))
                conn.commit()
                
    return False


def record_failed_attempt(ip_address: str = "127.0.0.1"):
    """Increments failed login counter for a given IP."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO login_attempts (ip_address, attempts, last_attempt)
            VALUES (?, 1, ?)
            ON CONFLICT(ip_address) DO UPDATE SET
                attempts = attempts + 1,
                last_attempt = excluded.last_attempt
        ''', (ip_address, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()


def clear_failed_attempts(ip_address: str = "127.0.0.1"):
    """Clears failed login attempts upon successful authentication."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_attempts WHERE ip_address = ?", (ip_address,))
        conn.commit()


def authenticate_admin(username: str, password: str) -> bool:
    """Validates provided admin username and password against stored database hashes."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM admin_users WHERE username = ?", (username.lower(),))
        row = cursor.fetchone()
        
    if row and verify_password(row[0], password):
        return True
    return False


def create_session(username: str) -> str:
    """Generates a secure 24-hour session token and stores it in active_sessions."""
    session_id = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(hours=24)
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO active_sessions (session_id, username, expires_at) VALUES (?, ?, ?)",
            (session_id, username.lower(), expires_at.strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        
    return session_id


def validate_session(session_id: str) -> bool:
    """Validates whether a session ID exists and has not expired."""
    if not session_id:
        return False
        
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT expires_at FROM active_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            try:
                exp_dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return False

            if datetime.now() < exp_dt:
                return True
            else:
                cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                
    return False


def terminate_session(session_id: str):
    """Deletes an active session (Logout)."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
