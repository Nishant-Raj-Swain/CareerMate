import sqlite3
import hashlib
import os
import secrets
from datetime import datetime, timedelta

DB_NAME = "bot_analytics.db"

def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + ":" + key.hex()

def verify_password(stored_password: str, provided_password: str) -> bool:
    try:
        salt_hex, key_hex = stored_password.split(":")
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(stored_key, new_key)
    except Exception:
        return False

def init_auth_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at DATETIME NOT NULL
        )
    ''')
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
    """Generates the main admin user from environment variables or defaults."""
    admin_user = os.getenv("DEFAULT_ADMIN_USER", "admin")
    admin_pass = os.getenv("DEFAULT_ADMIN_PASS", "AdminPass123!")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM admin_users")
    if cursor.fetchone()[0] == 0:
        pw_hash = hash_password(admin_pass)
        cursor.execute("INSERT INTO admin_users (username, password_hash) VALUES (?, ?)", (admin_user.lower(), pw_hash))
        conn.commit()
    conn.close()

def is_rate_limited(ip_address: str = "127.0.0.1") -> bool:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT attempts, last_attempt FROM login_attempts WHERE ip_address = ?", (ip_address,))
    row = cursor.fetchone()
    if row:
        attempts, last_attempt = row
        last_dt = datetime.strptime(last_attempt, "%Y-%m-%d %H:%M:%S")
        if datetime.now() - last_dt < timedelta(minutes=15) and attempts >= 5:
            conn.close()
            return True
        elif datetime.now() - last_dt >= timedelta(minutes=15):
            cursor.execute("UPDATE login_attempts SET attempts = 0 WHERE ip_address = ?", (ip_address,))
            conn.commit()
    conn.close()
    return False

def record_failed_attempt(ip_address: str = "127.0.0.1"):
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
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM login_attempts WHERE ip_address = ?", (ip_address,))
    conn.commit()
    conn.close()

def authenticate_admin(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM admin_users WHERE username = ?", (username.lower(),))
    row = cursor.fetchone()
    conn.close()
    if row and verify_password(row[0], password):
        return True
    return False

def create_session(username: str) -> str:
    session_id = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(hours=24)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO active_sessions (session_id, username, expires_at) VALUES (?, ?, ?)",
                   (session_id, username, expires_at.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return session_id

def validate_session(session_id: str) -> bool:
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
            cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
    conn.close()
    return False

def terminate_session(session_id: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
