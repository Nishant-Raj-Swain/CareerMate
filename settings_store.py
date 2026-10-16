# settings_store.py
# Small persisted key/value store for the admin panel's own preferences
# (bot display name, welcome message, notification toggles, theme).
#
# Deliberately separate from real secrets: WhatsApp tokens, the Meta app
# secret, and the database URL are never read from or written to this
# table. Those stay in environment variables on Render, and the Settings
# page only ever displays whether they're configured, never their value.

import json
from urllib.parse import urlparse

import analytics

DEFAULTS = {
    "bot_name": "CareerMate",
    "welcome_message": (
        "Welcome back, {user}!\n"
        "I'm your CareerMate bot. Use the sidebar to explore all features "
        "and find the best opportunities!"
    ),
    "notify_new_user": True,
    "notify_new_job": True,
    "notify_system_errors": True,
    "notify_weekly_reports": False,
    "theme": "dark",
    "primary_color": "cyan",
    "sidebar_style": "expanded",
    "bot_active": True,
    "bot_avatar": "🤖",
}


def _conn():
    return analytics._conn()


def init_settings_db():
    """Creates the admin_settings table in Postgres if needed (safe on every startup)."""
    with _conn() as conn:
        conn.raw('''
            CREATE TABLE IF NOT EXISTS admin_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        ''')


def get_settings() -> dict:
    with _conn() as conn:
        rows = conn.execute("SELECT key, value FROM admin_settings").fetchall()
    stored = {r["key"]: json.loads(r["value"]) for r in rows}
    merged = dict(DEFAULTS)
    merged.update(stored)
    return merged


ALLOWED_KEYS = set(DEFAULTS.keys())


def save_settings(patch: dict) -> dict:
    """Merges `patch` into the stored settings, ignoring any key that
    isn't one of the known, safe, non-secret preference fields."""
    clean = {k: v for k, v in (patch or {}).items() if k in ALLOWED_KEYS}
    with _conn() as conn:
        for k, v in clean.items():
            conn.raw(
                "INSERT INTO admin_settings (key, value, updated_at) VALUES (%s, %s, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
                (k, json.dumps(v))
            )
    return get_settings()


def config_info(base_url: str) -> dict:
    """Read-only, NON-secret facts for the Settings page: which integrations
    are configured (booleans only - never the values), the public webhook
    URL, and a live record count. Secret values are never returned."""
    import os

    def has(*names):
        return any((os.getenv(n) or "").strip() for n in names)

    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM analytics_events").fetchone()["c"]
        total += conn.raw("SELECT COUNT(*) c FROM users").fetchone()["c"]

    # connection string without credentials, for display only
    u = urlparse(os.getenv("DATABASE_URL") or "")
    db_url = f"postgresql://{u.hostname}{':' + str(u.port) if u.port else ''}{u.path}" if u.hostname else "postgresql://"

    base = base_url.rstrip("/")
    if base.startswith("http://") and "localhost" not in base and "127.0.0.1" not in base:
        base = "https://" + base[len("http://"):]   # Render terminates TLS in front of the app
    return {
        "webhook_url": base + "/webhook",
        "graph_api_version": (os.getenv("WHATSAPP_API_VERSION") or os.getenv("GRAPH_API_VERSION") or "").strip() or None,
        "whatsapp": {
            "access_token_set": has("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_TOKEN"),
            "phone_number_id_set": has("WHATSAPP_PHONE_NUMBER_ID", "PHONE_NUMBER_ID"),
            "verify_token_set": has("WEBHOOK_VERIFY_TOKEN"),
            "app_secret_set": has("META_APP_SECRET"),
        },
        "database": {"type": "PostgreSQL", "url": db_url, "total_records": total},
        "env_vars": [
            {"name": n, "set": has(*alts)} for n, alts in (
                ("WHATSAPP_ACCESS_TOKEN", ("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_TOKEN")),
                ("WHATSAPP_PHONE_NUMBER_ID", ("WHATSAPP_PHONE_NUMBER_ID", "PHONE_NUMBER_ID")),
                ("WEBHOOK_VERIFY_TOKEN", ("WEBHOOK_VERIFY_TOKEN",)),
                ("META_APP_SECRET", ("META_APP_SECRET",)),
                ("WHATSAPP_API_VERSION", ("WHATSAPP_API_VERSION", "GRAPH_API_VERSION")),
                ("NVIDIA_API_KEY", ("NVIDIA_API_KEY", "NIM_API_KEY")),
                ("OPENROUTER_API_KEY", ("OPENROUTER_API_KEY",)),
                ("GEMINI_API_KEY", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
                ("DATABASE_URL", ("DATABASE_URL",)),
            )
        ],
        "ai_providers": [
            {"name": "NVIDIA NIM", "configured": has("NVIDIA_API_KEY", "NIM_API_KEY")},
            {"name": "OpenRouter", "configured": has("OPENROUTER_API_KEY")},
            {"name": "Google Gemini", "configured": has("GEMINI_API_KEY", "GOOGLE_API_KEY")},
        ],
    }
