import sqlite3
import time

DB_FILE = "career_bot.db"

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                phone TEXT PRIMARY KEY,
                resume TEXT,
                state TEXT,
                doc_text TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id TEXT PRIMARY KEY,
                created_at REAL
            )
        """)
        conn.commit()

def claim_message(message_id: str) -> bool:
    now = time.time()
    with get_connection() as conn:
        conn.execute("DELETE FROM processed_messages WHERE created_at < ?", (now - 604800,))
        try:
            conn.execute("INSERT INTO processed_messages VALUES (?, ?)", (message_id, now))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def get_user(phone: str) -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        if not row:
            conn.execute("INSERT INTO users VALUES (?, '', '{}', '')", (phone,))
            conn.commit()
            return {"phone": phone, "resume": "", "state": {}, "doc_text": ""}
        import json
        return {
            "phone": row["phone"],
            "resume": row["resume"],
            "state": json.loads(row["state"] or "{}"),
            "doc_text": row["doc_text"] or ""
        }

def save_resume(phone: str, text: str):
    with get_connection() as conn:
        conn.execute("UPDATE users SET resume = ? WHERE phone = ?", (text, phone))
        conn.commit()

def save_state(phone: str, state: dict):
    import json
    with get_connection() as conn:
        conn.execute("UPDATE users SET state = ? WHERE phone = ?", (json.dumps(state), phone))
        conn.commit()

def save_document_text(phone: str, text: str):
    with get_connection() as conn:
        conn.execute("UPDATE users SET doc_text = ? WHERE phone = ?", (text, phone))
        conn.commit()

def delete_user(phone: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE phone = ?", (phone,))
        conn.commit()