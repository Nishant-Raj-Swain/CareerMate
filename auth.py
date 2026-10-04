import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

# Unified SQLite Database file
DB_NAME = "bot_analytics.db"
HASH_ITERATIONS = 600000  # Modern OWASP / NIST standard for PBKDF2-HMAC-SHA256


def hash_password(password: str, salt: bytes = None) -> str:
    """Generates a secure PBKDF2 password hash with 600,000 iterations and a 32-byte salt."""
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, HASH_ITERATIONS)
    return salt.hex() + ":" + key.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies a plain password against the stored salt:hash string in constant time."""
    try:
        salt_hex, key_hex = stored_password.split(":")
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, HASH_ITERATIONS)
        return hmac.compare_digest(stored_key, new_key)
    except Exception:
        return False


def verify_webhook_signature(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    """
    Validates Meta WhatsApp HMAC SHA256 signatures safely using constant-time comparison
    to eliminate timing attacks.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    
    expected_sig = signature_header.split("sha256=")[1].strip()
    calculated_sig = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(calculated_sig, expected_sig)


def seed_default_admin():
    """Seeds the initial admin account from environment variables if the admin table is empty."""
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
    """Initializes authentication, active session, rate-limiting tables, and seeds default admin."""
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL
            )
        ''')
        
        # Table 3: Login Attempt Rate Limiting (Supports both IP and Account tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_attempts (
                identifier TEXT PRIMARY KEY,
                attempts INTEGER DEFAULT 0,
                last_attempt DATETIME
            )
        ''')
        
        conn.commit()

    seed_default_admin()


def is_rate_limited(identifier: str, max_attempts: int = 5, window_minutes: int = 15) -> bool:
    """Checks if an identifier (IP address or Username) has exceeded failed login limits."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT attempts, last_attempt FROM login_attempts WHERE identifier = ?", (identifier,))
        row = cursor.fetchone()
        
        if row:
            attempts, last_attempt = row
            try:
                last_dt = datetime.strptime(last_attempt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return False

            if datetime.utcnow() - last_dt < timedelta(minutes=window_minutes) and attempts >= max_attempts:
                return True
            elif datetime.utcnow() - last_dt >= timedelta(minutes=window_minutes):
                cursor.execute("UPDATE login_attempts SET attempts = 0 WHERE identifier = ?", (identifier,))
                conn.commit()
                
    return False


def record_failed_attempt(identifier: str):
    """Increments failed login counter for a given identifier (IP or Username)."""
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO login_attempts (identifier, attempts, last_attempt)
            VALUES (?, 1, ?)
            ON CONFLICT(identifier) DO UPDATE SET
                attempts = attempts + 1,
                last_attempt = excluded.last_attempt
        ''', (identifier, now_str))
        conn.commit()


def clear_failed_attempts(identifier: str):
    """Clears failed login attempts upon successful authentication."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_attempts WHERE identifier = ?", (identifier,))
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


def create_session(username: str, duration_hours: int = 24) -> str:
    """Generates a secure 256-bit session token and stores it in active_sessions."""
    session_id = secrets.token_hex(32)
    expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO active_sessions (session_id, username, expires_at) VALUES (?, ?, ?)",
            (session_id, username.lower(), expires_at.strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        
    return session_id


def validate_session(session_id: str) -> str:
    """
    Validates whether a session ID exists and has not expired.
    Returns the associated username if valid, or None if invalid/expired.
    """
    if not session_id:
        return None
        
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Purge stale sessions
        cursor.execute("DELETE FROM active_sessions WHERE expires_at < ?", (now_str,))
        conn.commit()
        
        cursor.execute("SELECT username FROM active_sessions WHERE session_id = ? AND expires_at > ?", (session_id, now_str))
        row = cursor.fetchone()
        
        return row[0] if row else None


def terminate_session(session_id: str):
    """Deletes an active session token (Logout)."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
