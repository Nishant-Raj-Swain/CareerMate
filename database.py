import os
import json
import time
import psycopg2
import psycopg2.extras

def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(database_url, cursor_factory=psycopg2.extras.RealDictCursor)

def initialize():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    phone TEXT PRIMARY KEY,
                    resume TEXT,
                    state TEXT,
                    doc_text TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    created_at DOUBLE PRECISION,
                    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id SERIAL PRIMARY KEY,
                    user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                    event_type TEXT NOT NULL,
                    command TEXT,
                    success BOOLEAN DEFAULT TRUE,
                    response_time DOUBLE PRECISION,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS resume_analytics (
                    id SERIAL PRIMARY KEY,
                    user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                    ats_score INTEGER,
                    analysis_type TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS quiz_analytics (
                    id SERIAL PRIMARY KEY,
                    user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                    topic TEXT,
                    score INTEGER,
                    completed BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_errors (
                    id SERIAL PRIMARY KEY,
                    error_type TEXT,
                    details TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
        conn.commit()

def claim_message(message_id: str) -> bool:
    now = time.time()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM processed_messages WHERE created_at < %s", (now - 604800,))
            try:
                cur.execute("INSERT INTO processed_messages (message_id, created_at) VALUES (%s, %s)", (message_id, now))
                conn.commit()
                return True
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return False

def get_user(phone: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE phone = %s", (phone,))
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "INSERT INTO users (phone, resume, state, doc_text, created_at) VALUES (%s, %s, %s, %s, NOW()) ON CONFLICT (phone) DO NOTHING",
                    (phone, "", "{}", "")
                )
                conn.commit()
                return {"phone": phone, "resume": "", "state": {}, "doc_text": ""}
            
            state_data = row["state"]
            if isinstance(state_data, str):
                try:
                    state_data = json.loads(state_data or "{}")
                except Exception:
                    state_data = {}
            elif not state_data:
                state_data = {}

            return {
                "phone": row["phone"],
                "resume": row["resume"] or "",
                "state": state_data,
                "doc_text": row["doc_text"] or ""
            }

def save_resume(phone: str, text: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET resume = %s WHERE phone = %s", (text, phone))
        conn.commit()

def save_state(phone: str, state: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET state = %s WHERE phone = %s", (json.dumps(state), phone))
        conn.commit()

def save_document_text(phone: str, text: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET doc_text = %s WHERE phone = %s", (text, phone))
        conn.commit()

def delete_user(phone: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE phone = %s", (phone,))
        conn.commit()

def log_analytics_event(user_phone: str, event_type: str, command: str = None, success: bool = True, response_time: float = 0.0):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analytics_events (user_phone, event_type, command, success, response_time, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (user_phone, event_type, command, success, response_time))
        conn.commit()


def log_quiz_result(user_phone: str, topic: str, score: int, attempted: int, completed: bool):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO quiz_analytics (user_phone, topic, score, attempted, completed, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (user_phone, topic, score, attempted, completed))
        conn.commit()
