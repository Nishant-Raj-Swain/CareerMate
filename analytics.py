# analytics.py  (PostgreSQL / Supabase edition)
# Everything the admin dashboard needs, computed from your Supabase Postgres
# database (DATABASE_URL), the same database database.py uses.
#
# Source tables (all created by database.initialize(), nothing new is needed):
#   analytics_events   one row per bot event (event_details holds a JSON "meta" blob)
#   quiz_analytics     finished quizzes   -> exposed to the dashboard as `quiz_complete` events
#   resume_analytics   ATS scores         -> exposed to the dashboard as `ats_score` events
#   system_errors      bot errors         -> merged into the Errors panel (recent_errors)
#   users              registration date / phone list
#
# The dashboard functions below query ONE read-only view, `v_events`, that
# merges those tables into a single event stream. init_analytics_db() creates
# or refreshes that view on startup (CREATE OR REPLACE VIEW, safe to repeat).
# Dedicated-table rows are skipped when the bot ALSO logged the same event in
# analytics_events within 2 minutes, so nothing is double counted.

import os
import re
import json
import threading
from datetime import datetime, timedelta
from decimal import Decimal

import psycopg2
import psycopg2.extras
import psycopg2.extensions
import psycopg2.pool

# Kept so older imports (`from analytics import DB_NAME`) don't crash. Unused.
DB_NAME = "postgres"

# Commands considered "a feature" vs plumbing commands, used by the funnel
CAREER_FEATURE_COMMANDS = {
    "/news", "/competitions", "/competition", "/internships", "/internship",
    "/prepare", "/job_search", "/roadmap", "/send", "/assistance", "/ats", "/ats_score",  # "/ats_score" = old rows logged before the rename
}

# ---------------------------------------------------------------------------
# connection layer
# ---------------------------------------------------------------------------
# numeric/decimal -> float so Python arithmetic and JSON behave exactly like before
_DEC2FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values, "DEC2FLOAT",
    lambda v, cur: float(v) if v is not None else None)
psycopg2.extensions.register_type(_DEC2FLOAT)


class Row(dict):
    """dict row that also supports row[0] style access (like sqlite3.Row)."""
    def __getitem__(self, k):
        if isinstance(k, int):
            return list(self.values())[k]
        return dict.__getitem__(self, k)


class _RowCursor(psycopg2.extras.RealDictCursor):
    def fetchone(self):
        r = super().fetchone()
        return Row(r) if r is not None else None

    def fetchall(self):
        return [Row(r) for r in super().fetchall()]

    def fetchmany(self, size=None):
        return [Row(r) for r in super().fetchmany(size)]

    def __iter__(self):
        return iter(self.fetchall())


_POOL = None
_POOL_LOCK = threading.Lock()


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                url = os.getenv("DATABASE_URL")
                if not url:
                    raise RuntimeError("DATABASE_URL environment variable is not set.")
                _POOL = psycopg2.pool.ThreadedConnectionPool(
                    1, int(os.getenv("ANALYTICS_DB_POOL", "5")), url,
                    connect_timeout=10)
    return _POOL


_FMT_SUBSTR = {  # sqlite strftime formats actually used by the queries
    "%Y-%m-%d %H:%M:%S": (1, 19), "%Y-%m-%d %H": (1, 13), "%Y-%m-%d": (1, 10),
    "%Y-%m": (1, 7), "%H": (12, 2), "%H:%M": (12, 5), "%w": None,
}
_NOW_UTC = "(now() AT TIME ZONE 'UTC')"
_TS_FMT = "'YYYY-MM-DD HH24:MI:SS'"


def _strftime(m):
    fmt, expr = m.group(1), m.group(2).strip()
    if fmt == "%w":  # weekday 0=Sunday, like sqlite
        return f"EXTRACT(DOW FROM ({expr})::timestamp)::int::text"
    rng = _FMT_SUBSTR.get(fmt)
    if rng is None:
        raise ValueError(f"unsupported strftime format {fmt}")
    return f"substr({expr}, {rng[0]}, {rng[1]})"


def _dt_now(m):
    n, unit = int(m.group(1)), m.group(2)
    unit = unit.rstrip("s")
    return f"to_char({_NOW_UTC} - interval '{abs(n)} {unit}', {_TS_FMT})" if n < 0 else \
           f"to_char({_NOW_UTC} + interval '{n} {unit}', {_TS_FMT})"


def _translate(sql: str, has_params: bool) -> str:
    """Rewrites the few SQLite-isms the dashboard queries use into Postgres SQL."""
    sql = re.sub(r"strftime\(\s*'([^']*)'\s*,\s*((?:[^()]|\([^()]*\))*)\)", _strftime, sql)
    sql = re.sub(r"datetime\('now'\s*,\s*'([+-]?\d+)\s*(day|days|hour|hours|minute|minutes)'\)", _dt_now, sql)
    sql = re.sub(r"datetime\('now'\)", f"to_char({_NOW_UTC}, {_TS_FMT})", sql)
    sql = re.sub(r"date\('now'\s*,\s*'([+-]?\d+)\s*(day|days)'\)",
                 lambda m: f"to_char({_NOW_UTC} + interval '{int(m.group(1))} day', 'YYYY-MM-DD')", sql)
    sql = re.sub(r"date\('now'\)", f"to_char({_NOW_UTC}, 'YYYY-MM-DD')", sql)
    # sqlite dates are plain 'YYYY-MM-DD' strings; keep them text so comparisons behave identically
    sql = re.sub(r"\bdate\(((?:[^()]|\([^()]*\))*)\)", r"substr(\1, 1, 10)", sql, flags=re.I)
    sql = re.sub(r"julianday\('now'\)", f"(EXTRACT(EPOCH FROM {_NOW_UTC}) / 86400.0)", sql)
    sql = re.sub(r"julianday\(((?:[^()]|\([^()]*\))*)\)", r"(EXTRACT(EPOCH FROM (\1)::timestamp) / 86400.0)", sql)
    sql = re.sub(r"\bIFNULL\(", "COALESCE(", sql, flags=re.I)
    sql = re.sub(r"\banalytics_events\b", "v_events", sql)
    if has_params:
        sql = sql.replace("%", "%%").replace("?", "%s")
    return sql


class _Conn:
    """sqlite3-style facade over a pooled psycopg2 connection:
       `with _conn() as conn: conn.execute(sql, params).fetchone()`."""
    def __init__(self):
        self._pool = _pool()
        self._c = self._pool.getconn()
        self._c.autocommit = False

    def execute(self, sql, params=None):
        cur = self._c.cursor(cursor_factory=_RowCursor)
        params = tuple(params) if params else None
        cur.execute(_translate(sql, params is not None), params)
        return cur

    def raw(self, sql, params=None):  # no translation (DDL / writes)
        cur = self._c.cursor(cursor_factory=_RowCursor)
        cur.execute(sql, params)
        return cur

    def commit(self):
        self._c.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type:
                self._c.rollback()
            else:
                self._c.commit()
        finally:
            self._pool.putconn(self._c)
        return False


def _conn():
    return _Conn()


_VIEW_SQL = """
CREATE OR REPLACE VIEW v_events AS
SELECT e.id::bigint AS id, e.user_phone, e.event_type, e.command,
       CASE WHEN e.event_details IS NULL OR btrim(e.event_details) = '' THEN '{}'
            WHEN btrim(e.event_details) LIKE '{%' THEN e.event_details
            ELSE json_build_object('message', e.event_details)::text END AS meta,
       CASE WHEN e.success IS FALSE THEN 0 ELSE 1 END AS success,
       e.response_time::float8 AS response_time_ms,
       to_char(e.timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') AS "timestamp"
  FROM analytics_events e
 WHERE e.user_phone IS NOT NULL
UNION ALL
SELECT 1000000000 + q.id, q.user_phone, 'quiz_complete', '/quiz',
       json_build_object('topic', q.topic, 'score', COALESCE(q.score, 0),
                         'total', COALESCE(q.attempted, 0))::text,
       1, NULL::float8, to_char(q.timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
  FROM quiz_analytics q
 WHERE q.user_phone IS NOT NULL AND q.completed IS TRUE
   AND NOT EXISTS (SELECT 1 FROM analytics_events x
                    WHERE x.user_phone = q.user_phone AND x.event_type = 'quiz_complete'
                      AND abs(extract(epoch FROM (x.timestamp - q.timestamp))) < 120)
UNION ALL
SELECT 2000000000 + r.id, r.user_phone, 'ats_score', '/ats',
       json_build_object('score', r.ats_score, 'analysis_type', r.analysis_type)::text,
       1, NULL::float8, to_char(r.timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
  FROM resume_analytics r
 WHERE r.user_phone IS NOT NULL AND r.ats_score IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM analytics_events x
                    WHERE x.user_phone = r.user_phone AND x.event_type = 'ats_score'
                      AND abs(extract(epoch FROM (x.timestamp - r.timestamp))) < 120)
"""


def init_analytics_db():
    """Makes sure the columns the dashboard needs exist and (re)creates the
    v_events view. Safe to call on every startup. Never drops or alters data."""
    with _conn() as conn:
        conn.raw("""CREATE TABLE IF NOT EXISTS users (
                        phone TEXT PRIMARY KEY, resume TEXT, state TEXT, doc_text TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        conn.raw("""CREATE TABLE IF NOT EXISTS analytics_events (
                        id SERIAL PRIMARY KEY,
                        user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                        event_type TEXT NOT NULL, command TEXT, success BOOLEAN DEFAULT TRUE,
                        response_time DOUBLE PRECISION,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        conn.raw("""CREATE TABLE IF NOT EXISTS resume_analytics (
                        id SERIAL PRIMARY KEY, user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                        ats_score INTEGER, analysis_type TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        conn.raw("""CREATE TABLE IF NOT EXISTS quiz_analytics (
                        id SERIAL PRIMARY KEY, user_phone TEXT REFERENCES users(phone) ON DELETE SET NULL,
                        topic TEXT, score INTEGER, completed BOOLEAN DEFAULT FALSE,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        conn.raw("""CREATE TABLE IF NOT EXISTS system_errors (
                        id SERIAL PRIMARY KEY, error_type TEXT, details TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        # columns database.py's log functions already write but initialize() doesn't create
        conn.raw("ALTER TABLE analytics_events ADD COLUMN IF NOT EXISTS event_details TEXT")
        conn.raw("ALTER TABLE quiz_analytics ADD COLUMN IF NOT EXISTS attempted INTEGER")
        conn.raw("CREATE INDEX IF NOT EXISTS idx_events_phone ON analytics_events(user_phone)")
        conn.raw("CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type)")
        conn.raw("CREATE INDEX IF NOT EXISTS idx_events_ts ON analytics_events(timestamp)")
        conn.raw(_VIEW_SQL)


def record_quiz(phone: str, topic: str, score: int, attempted: int, completed: bool) -> None:
    """One row per finished OR abandoned quiz in quiz_analytics (score so far + how many questions were answered).
    Never raises."""
    try:
        with _conn() as conn:
            conn.raw("INSERT INTO users (phone, resume, state, doc_text) VALUES (%s, '', '{}', '') "
                     "ON CONFLICT (phone) DO NOTHING", (phone,))
            conn.raw("INSERT INTO quiz_analytics (user_phone, topic, score, attempted, completed) VALUES (%s,%s,%s,%s,%s)",
                     (phone, topic, int(score or 0), int(attempted or 0), bool(completed)))
    except Exception:
        pass


def log_event(phone: str, event_type: str, command: str = None, meta: dict = None,
              success: bool = True, response_time_ms: int = None):
    """Records one analytics event. Never raises — a logging failure should
    never break the bot's actual reply to the user. meta is stored as JSON in
    analytics_events.event_details; response_time stores milliseconds."""
    try:
        with _conn() as conn:
            # analytics_events.user_phone is a foreign key to users(phone)
            conn.raw("INSERT INTO users (phone, resume, state, doc_text) VALUES (%s, '', '{}', '') "
                     "ON CONFLICT (phone) DO NOTHING", (phone,))
            conn.raw("""INSERT INTO analytics_events
                        (user_phone, event_type, command, event_details, success, response_time)
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                     (phone, event_type, command, json.dumps(meta or {}), bool(success), response_time_ms))
    except Exception:
        pass


def extract_score(text: str):
    """Best-effort pull of a 0-100 score out of a free-text AI report,
    matching patterns like 'Score: 82', '82%', '82/100'."""
    if not text:
        return None
    patterns = [
        r'(?:score|ats score)[^\d]{0,10}(\d{1,3})\s*%',
        r'(?:score|ats score)[^\d]{0,10}(\d{1,3})\s*/\s*100',
        r'(\d{1,3})\s*%\s*(?:ats|match)',
        r'(\d{1,3})\s*/\s*100',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
    return None


# ---------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------
def _range_clause(start: str = None, end: str = None, date: str = None):
    """Returns (sql_clause, params) for filtering analytics_events.timestamp."""
    if date:
        return "date(timestamp) = ?", (date,)
    if start and end:
        return "date(timestamp) BETWEEN ? AND ?", (start, end)
    if start:
        return "date(timestamp) >= ?", (start,)
    return "1=1", ()


# ---------------------------------------------------------------
# dashboard sections
# ---------------------------------------------------------------
def overview(date: str = None):
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    with _conn() as conn:
        total_users = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events"
        ).fetchone()["c"]

        today_users = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE date(timestamp) = ?",
            (date or today,)
        ).fetchone()["c"]

        active_users = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE date(timestamp) >= ?",
            (week_ago,)
        ).fetchone()["c"]

        messages_processed = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events"
        ).fetchone()["c"]

        ai_requests = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE event_type IN "
            "('ats_score','ai_assistance','roadmap','quiz_start')"
        ).fetchone()["c"]

    return {
        "total_users": total_users,
        "today_users": today_users,
        "active_users_7d": active_users,
        "messages_processed": messages_processed,
        "ai_requests": ai_requests,
    }


def new_vs_returning(date: str = None):
    day = date or datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        active_today = {r["user_phone"] for r in conn.execute(
            "SELECT DISTINCT user_phone FROM analytics_events WHERE date(timestamp) = ?", (day,)
        )}
        first_seen_today = {r["user_phone"] for r in conn.execute(
            '''SELECT user_phone FROM analytics_events
               GROUP BY user_phone HAVING date(MIN(timestamp)) = ?''', (day,)
        )}
    new_users = len(active_today & first_seen_today)
    returning = len(active_today) - new_users
    return {"date": day, "new_users": new_users, "returning_users": max(returning, 0)}


def monthly_growth(months: int = 6):
    with _conn() as conn:
        rows = conn.execute(
            '''SELECT strftime('%Y-%m', timestamp) ym, COUNT(DISTINCT user_phone) c
               FROM analytics_events
               GROUP BY ym ORDER BY ym DESC LIMIT ?''', (months,)
        ).fetchall()
    data = [{"month": r["ym"], "users": r["c"]} for r in rows]
    return list(reversed(data))


def daily_activity(start: str = None, end: str = None):
    clause, params = _range_clause(start, end)
    with _conn() as conn:
        rows = conn.execute(
            f'''SELECT date(timestamp) d, COUNT(DISTINCT user_phone) users, COUNT(*) messages
                FROM analytics_events WHERE {clause}
                GROUP BY d ORDER BY d''', params
        ).fetchall()
    return [{"date": r["d"], "users": r["users"], "messages": r["messages"]} for r in rows]


def hourly_activity(date: str = None, start: str = None, end: str = None):
    clause, params = _range_clause(start, end, date)
    with _conn() as conn:
        rows = conn.execute(
            f'''SELECT strftime('%H', timestamp) h, COUNT(*) c
                FROM analytics_events WHERE {clause}
                GROUP BY h ORDER BY h''', params
        ).fetchall()
    by_hour = {f"{h:02d}": 0 for h in range(24)}
    for r in rows:
        by_hour[r["h"]] = r["c"]
    return [{"hour": h, "count": c} for h, c in by_hour.items()]


def feature_usage(date: str = None, start: str = None, end: str = None):
    clause, params = _range_clause(start, end, date)
    with _conn() as conn:
        rows = conn.execute(
            f'''SELECT command, COUNT(*) c FROM analytics_events
                WHERE event_type = 'command' AND command IS NOT NULL AND {clause}
                GROUP BY command ORDER BY c DESC, command''', params
        ).fetchall()
    total = sum(r["c"] for r in rows) or 1
    usage = [{"command": r["command"], "count": r["c"], "pct": round(r["c"] * 100 / total, 1)} for r in rows]
    return {"usage": usage, "most_used": usage[:5]}


def resume_analytics():
    with _conn() as conn:
        uploaded = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE event_type = 'resume_upload'"
        ).fetchone()["c"]
        scores = [
            json.loads(r["meta"] or "{}").get("score")
            for r in conn.execute(
                "SELECT meta FROM analytics_events WHERE event_type = 'ats_score' AND success = 1"
            )
        ]
        rt = conn.execute(
            "SELECT AVG(response_time_ms) a FROM analytics_events "
            "WHERE event_type = 'ats_score' AND response_time_ms IS NOT NULL"
        ).fetchone()["a"]
    scores = [s for s in scores if isinstance(s, (int, float))]
    buckets = {"0-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for s in scores:
        if s <= 40:
            buckets["0-40"] += 1
        elif s <= 60:
            buckets["41-60"] += 1
        elif s <= 80:
            buckets["61-80"] += 1
        else:
            buckets["81-100"] += 1
    return {
        "resumes_uploaded": uploaded,
        "ats_analyses": len(scores),
        "average_score": round(sum(scores) / len(scores), 1) if scores else None,
        "highest_score": max(scores) if scores else None,
        "score_distribution": buckets,
        "avg_response_ms": round(rt) if rt else 0,
    }


def job_search_analytics():
    with _conn() as conn:
        rows = conn.execute(
            "SELECT user_phone, meta FROM analytics_events WHERE event_type = 'job_search'"
        ).fetchall()
    topics = {}
    users = set()
    for r in rows:
        users.add(r["user_phone"])
        topic = json.loads(r["meta"] or "{}").get("topic")
        if topic:
            topics[topic] = topics.get(topic, 0) + 1
    popular = sorted(topics.items(), key=lambda x: -x[1])[:5]
    return {
        "searches": len(rows),
        "unique_users": len(users),
        "popular_topics": [{"topic": t, "count": c} for t, c in popular],
        "other_count": max(len(rows) - sum(c for _, c in popular), 0),
    }


def quiz_analytics():
    with _conn() as conn:
        started = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE event_type = 'quiz_start'"
        ).fetchone()["c"]
        completed_rows = conn.execute(
            "SELECT meta FROM analytics_events WHERE event_type = 'quiz_complete'"
        ).fetchall()

    scores_pct, topics, questions_answered = [], {}, 0
    for r in completed_rows:
        m = json.loads(r["meta"] or "{}")
        score, total, topic = m.get("score"), m.get("total"), m.get("topic")
        if isinstance(score, (int, float)) and isinstance(total, (int, float)) and total > 0:
            scores_pct.append(score * 100 / total)
            questions_answered += total
        if topic:
            topics[topic] = topics.get(topic, 0) + 1

    popular = sorted(topics.items(), key=lambda x: -x[1])[:5]
    return {
        "quizzes_started": started,
        "quizzes_completed": len(completed_rows),
        "questions_answered": questions_answered,
        "average_score_pct": round(sum(scores_pct) / len(scores_pct), 1) if scores_pct else None,
        "completion_rate": min(100.0, round(len(completed_rows) * 100 / started, 1)) if started else None,
        "popular_topics": [{"topic": t, "count": c} for t, c in popular],
    }


def performance():
    with _conn() as conn:
        received = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE event_type = 'command'"
        ).fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE event_type = 'command' AND success = 0"
        ).fetchone()["c"]
        avg_rt = conn.execute(
            "SELECT AVG(response_time_ms) a FROM analytics_events "
            "WHERE event_type = 'command' AND response_time_ms IS NOT NULL"
        ).fetchone()["a"]

    processed = received - failed
    return {
        "messages_received": received,
        "messages_processed": processed,
        "failed_messages": failed,
        "success_rate": round(processed * 100 / received, 1) if received else None,
        "avg_response_time_ms": round(avg_rt, 0) if avg_rt else None,
    }


def ai_usage():
    with _conn() as conn:
        def count(event_type):
            return conn.execute(
                "SELECT COUNT(*) c FROM analytics_events WHERE event_type = ?", (event_type,)
            ).fetchone()["c"]

        return {
            "ai_requests": count("ats_score") + count("ai_assistance") + count("roadmap") + count("quiz_start"),
            "resume_analyses": count("ats_score"),
            "ai_assistance_chats": count("ai_assistance"),
            "roadmap_generations": count("roadmap"),
            "quiz_generations": count("quiz_start"),
        }


def recent_errors(limit: int = 10):
    _ERR = """(SELECT event_type, meta, timestamp FROM analytics_events WHERE success = 0
               UNION ALL
               SELECT 'error', json_build_object('message', COALESCE(error_type,'') ||
                      CASE WHEN details IS NULL THEN '' ELSE ': ' || details END)::text,
                      to_char(timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') FROM system_errors) e"""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT event_type, meta, timestamp FROM {_ERR} ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        errors_today = conn.execute(
            f"SELECT COUNT(*) c FROM {_ERR} WHERE date(timestamp) = date('now')"
        ).fetchone()["c"]

    items = []
    for r in rows:
        m = json.loads(r["meta"] or "{}")
        items.append({
            "time": r["timestamp"],
            "stage": r["event_type"],
            "message": m.get("message", ""),
        })
    return {"errors_today": errors_today, "recent": items}


def users_summary(start: str = None, end: str = None):
    """Total / new / returning / active users for the Users page KPI row."""
    clause, params = _range_clause(start, end)
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    with _conn() as conn:
        total_users = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events"
        ).fetchone()["c"]

        active_in_range = {r["user_phone"] for r in conn.execute(
            f"SELECT DISTINCT user_phone FROM analytics_events WHERE {clause}", params
        )}

        first_seen_in_range = {r["user_phone"] for r in conn.execute(
            f'''SELECT user_phone FROM analytics_events
                GROUP BY user_phone HAVING date(MIN(timestamp)) BETWEEN
                coalesce(?, '0001-01-01') AND coalesce(?, date('now'))''',
            (start, end or start)
        )} if (start or end) else {r["user_phone"] for r in conn.execute(
            '''SELECT user_phone FROM analytics_events
               GROUP BY user_phone HAVING date(MIN(timestamp)) >= ?''', (week_ago,)
        )}

        active_7d = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE date(timestamp) >= ?",
            (week_ago,)
        ).fetchone()["c"]

    new_users = len(active_in_range & first_seen_in_range) if (start or end) else len(first_seen_in_range)
    returning = max(len(active_in_range) - new_users, 0)

    return {
        "total_users": total_users,
        "new_users": new_users,
        "returning_users": returning,
        "active_users_7d": active_7d,
    }


def users_timeseries(start: str = None, end: str = None):
    """Per-day series for the User Growth (line) and User Activity (bar)
    charts: cumulative total users, new users that day, and active users
    that day."""
    clause, params = _range_clause(start, end)
    with _conn() as conn:
        days = [r["d"] for r in conn.execute(
            f"SELECT DISTINCT date(timestamp) d FROM analytics_events WHERE {clause} ORDER BY d", params
        )]
        first_seen = {}
        for r in conn.execute(
            "SELECT user_phone, date(MIN(timestamp)) d FROM analytics_events GROUP BY user_phone"
        ):
            first_seen.setdefault(r["d"], 0)
            first_seen[r["d"]] += 1

        active_by_day = {}
        for r in conn.execute(
            f'''SELECT date(timestamp) d, COUNT(DISTINCT user_phone) c
                FROM analytics_events WHERE {clause} GROUP BY d''', params
        ):
            active_by_day[r["d"]] = r["c"]

        total_before = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE date(timestamp) < ?",
            (days[0] if days else datetime.now().strftime("%Y-%m-%d"),)
        ).fetchone()["c"] if days else 0

    running_total = total_before
    out = []
    for d in days:
        new_today = first_seen.get(d, 0)
        running_total += new_today
        out.append({
            "date": d,
            "total_users": running_total,
            "new_users": new_today,
            "active_users": active_by_day.get(d, 0),
        })
    return out


def users_source():
    """CareerMate is a WhatsApp-only bot today, so every user's source is
    WhatsApp — reported honestly rather than inventing channels that
    don't exist yet."""
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(DISTINCT user_phone) c FROM analytics_events").fetchone()["c"]
    return {
        "total_users": total,
        "sources": [
            {"label": "WhatsApp", "count": total, "pct": 100.0 if total else 0.0},
            {"label": "Direct", "count": 0, "pct": 0.0},
            {"label": "Others", "count": 0, "pct": 0.0},
        ],
    }


def users_activity_stats():
    with _conn() as conn:
        total_messages = conn.execute("SELECT COUNT(*) c FROM analytics_events").fetchone()["c"]
        total_users = conn.execute("SELECT COUNT(DISTINCT user_phone) c FROM analytics_events").fetchone()["c"]
        active_sessions = conn.execute(
            "SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE timestamp >= datetime('now','-30 minutes')"
        ).fetchone()["c"]
        recency_rows = conn.execute(
            '''SELECT (julianday('now') - julianday(MAX(timestamp))) d
               FROM analytics_events GROUP BY user_phone'''
        ).fetchall()

    recencies = [r["d"] for r in recency_rows if r["d"] is not None]
    avg_recent_days = round(sum(recencies) / len(recencies), 1) if recencies else None

    return {
        "total_messages": total_messages,
        "avg_messages_per_user": round(total_messages / total_users, 1) if total_users else 0,
        "active_sessions": active_sessions,
        "avg_recent_activity_days": avg_recent_days,
    }


def list_users(limit: int = 20, offset: int = 0, search: str = None):
    where = "WHERE user_phone LIKE ?" if search else ""
    params = (f"%{search}%",) if search else ()
    with _conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(DISTINCT user_phone) c FROM analytics_events {where}", params
        ).fetchone()["c"]

        rows = conn.execute(
            f'''SELECT user_phone,
                       MIN(timestamp) joined_on,
                       MAX(timestamp) last_active,
                       COUNT(*) total_messages
                FROM analytics_events
                {where}
                GROUP BY user_phone
                ORDER BY last_active DESC, user_phone
                LIMIT ? OFFSET ?''',
            params + (limit, offset)
        ).fetchall()

    items = []
    for i, r in enumerate(rows, start=offset + 1):
        items.append({
            "index": i,
            "phone": r["user_phone"],
            "label": f"User {i}",
            "joined_on": r["joined_on"],
            "last_active": r["last_active"],
            "total_messages": r["total_messages"],
            "status": "Active",
        })
    return {"total": total, "limit": limit, "offset": offset, "users": items}


def user_engagement(limit: int = 6):
    with _conn() as conn:
        rows = conn.execute(
            '''SELECT user_phone,
                      COUNT(*) messages,
                      SUM(CASE WHEN event_type = 'command' THEN 1 ELSE 0 END) commands
               FROM analytics_events
               GROUP BY user_phone
               ORDER BY messages DESC, user_phone
               LIMIT ?''', (limit,)
        ).fetchall()
    out = []
    for i, r in enumerate(rows, start=1):
        out.append({
            "label": f"User {i}",
            "phone": r["user_phone"],
            "messages": r["messages"],
            "commands": r["commands"] or 0,
        })
    return out


def popular_commands_by_users(limit: int = 5):
    """Like feature_usage(), but ranked by how many distinct users used
    each command rather than total uses — 'Popular Commands (by Users)'."""
    with _conn() as conn:
        rows = conn.execute(
            '''SELECT command, COUNT(DISTINCT user_phone) users
               FROM analytics_events
               WHERE event_type = 'command' AND command IS NOT NULL
               GROUP BY command ORDER BY users DESC, command LIMIT ?''', (limit,)
        ).fetchall()
        total_users = conn.execute("SELECT COUNT(DISTINCT user_phone) c FROM analytics_events").fetchone()["c"] or 1
    return [
        {"command": r["command"], "users": r["users"], "pct": round(r["users"] * 100 / total_users, 1)}
        for r in rows
    ]


def user_detail(phone: str, days: int = 30):
    with _conn() as conn:
        profile = conn.execute(
            '''SELECT MIN(timestamp) joined_on, MAX(timestamp) last_active, COUNT(*) total_messages
               FROM analytics_events WHERE user_phone = ?''', (phone,)
        ).fetchone()
        if not profile or profile["total_messages"] == 0:
            return None

        job_searches = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE user_phone = ? AND event_type = 'job_search'", (phone,)
        ).fetchone()["c"]
        quiz_attempts = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE user_phone = ? AND event_type IN ('quiz_start')", (phone,)
        ).fetchone()["c"]
        resume_analyses = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE user_phone = ? AND event_type = 'ats_score' AND success = 1",
            (phone,)
        ).fetchone()["c"]
        avg_rt = conn.execute(
            "SELECT AVG(response_time_ms) a FROM analytics_events WHERE user_phone = ? AND response_time_ms IS NOT NULL",
            (phone,)
        ).fetchone()["a"]

        activity_rows = conn.execute(
            '''SELECT date(timestamp) d, COUNT(*) messages,
                      SUM(CASE WHEN event_type='command' THEN 1 ELSE 0 END) commands
               FROM analytics_events WHERE user_phone = ?
               GROUP BY d ORDER BY d''', (phone,)
        ).fetchall()

        feature_rows = conn.execute(
            '''SELECT command, COUNT(*) c FROM analytics_events
               WHERE user_phone = ? AND event_type = 'command' AND command IS NOT NULL
               GROUP BY command ORDER BY c DESC, command''', (phone,)
        ).fetchall()

        funnel_used_feature = conn.execute(
            f'''SELECT COUNT(*) c FROM analytics_events
                WHERE user_phone = ? AND event_type = 'command' AND command IN
                ({",".join("?" * len(CAREER_FEATURE_COMMANDS))})''',
            (phone, *CAREER_FEATURE_COMMANDS)
        ).fetchone()["c"]
        funnel_job_search = job_searches
        funnel_applied = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE user_phone = ? AND event_type = 'job_apply'", (phone,)
        ).fetchone()["c"]

        job_rows = conn.execute(
            "SELECT meta, timestamp FROM analytics_events WHERE user_phone = ? AND event_type = 'job_search' ORDER BY timestamp DESC, id DESC",
            (phone,)
        ).fetchall()

        quiz_rows = conn.execute(
            "SELECT meta, timestamp, event_type FROM analytics_events WHERE user_phone = ? AND event_type IN "
            "('quiz_complete','quiz_exit_early') ORDER BY timestamp DESC, id DESC", (phone,)
        ).fetchall()

        resume_rows = conn.execute(
            "SELECT meta, timestamp FROM analytics_events WHERE user_phone = ? AND event_type = 'ats_score' ORDER BY timestamp DESC, id DESC",
            (phone,)
        ).fetchall()

        recent_events = conn.execute(
            '''SELECT event_type, command, meta, success, timestamp FROM analytics_events
               WHERE user_phone = ? ORDER BY timestamp DESC, id DESC LIMIT 25''', (phone,)
        ).fetchall()

        peak_row = conn.execute(
            '''SELECT strftime('%H', timestamp) h, COUNT(*) c FROM analytics_events
               WHERE user_phone = ? GROUP BY h ORDER BY c DESC LIMIT 1''', (phone,)
        ).fetchone()
        active_days = len(activity_rows)

    total_msgs = profile["total_messages"]

    quiz_scores = []
    for r in quiz_rows:
        m = json.loads(r["meta"] or "{}")
        score, total = m.get("score"), m.get("total")
        if isinstance(score, (int, float)) and isinstance(total, (int, float)) and total > 0:
            quiz_scores.append(score * 100 / total)

    resume_scores = []
    for r in resume_rows:
        m = json.loads(r["meta"] or "{}")
        if isinstance(m.get("score"), (int, float)):
            resume_scores.append(m["score"])

    # --- aggregated views for the Job / Quiz / Resume tabs ---
    all_job_topics, job_last = {}, {}
    for r in job_rows:
        t = json.loads(r["meta"] or "{}").get("topic") or "—"
        all_job_topics[t] = all_job_topics.get(t, 0) + 1
        job_last.setdefault(t, r["timestamp"])   # rows are newest-first
    job_total = sum(all_job_topics.values()) or 1
    job_domains = [
        {"topic": t, "count": c, "pct": round(c * 100 / job_total, 1), "last": job_last[t]}
        for t, c in sorted(all_job_topics.items(), key=lambda x: -x[1])
    ]

    quiz_topic_pcts, quiz_completed = {}, 0
    quiz_topic_raw = {}
    for r in quiz_rows:
        m = json.loads(r["meta"] or "{}")
        if r["event_type"] == "quiz_complete":
            quiz_completed += 1
        sc, tt = m.get("score"), m.get("total")
        if isinstance(sc, (int, float)) and isinstance(tt, (int, float)) and tt > 0:
            quiz_topic_pcts.setdefault(m.get("topic") or "General", []).append(sc * 100 / tt)
            raw = quiz_topic_raw.setdefault(m.get("topic") or "General", [0, 0])
            raw[0] += sc
            raw[1] += tt
    quiz_topics = sorted(
        [{"topic": t, "attempts": len(v), "avg_pct": round(sum(v) / len(v), 1),
          "correct": round(quiz_topic_raw[t][0]), "questions": round(quiz_topic_raw[t][1])} for t, v in quiz_topic_pcts.items()],
        key=lambda x: -x["attempts"])

    r_dist = {"0-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for sc in resume_scores:
        r_dist["0-40" if sc <= 40 else "41-60" if sc <= 60 else "61-80" if sc <= 80 else "81-100"] += 1

    r_dist5 = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for sc in resume_scores:
        r_dist5["0-20" if sc <= 20 else "21-40" if sc <= 40 else "41-60" if sc <= 60 else "61-80" if sc <= 80 else "81-100"] += 1

    # resume file formats (only if the bot records filename/format in event meta)
    formats = {}
    with _conn() as conn:
        for r in conn.execute("SELECT meta FROM analytics_events WHERE user_phone = ? AND event_type IN ('resume_upload','ats_score')", (phone,)):
            m = json.loads(r["meta"] or "{}")
            fmt_ = m.get("format") or m.get("file_type")
            if not fmt_ and isinstance(m.get("filename"), str) and "." in m["filename"]:
                fmt_ = m["filename"].rsplit(".", 1)[-1]
            if fmt_:
                formats[str(fmt_).upper().lstrip(".")] = formats.get(str(fmt_).upper().lstrip("."), 0) + 1

    # period-over-period deltas (last `days` vs the `days` before)
    days = max(1, min(int(days), 365))
    now = datetime.now()
    cur_start = (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    prev_start = (now - timedelta(days=2 * days)).strftime("%Y-%m-%d %H:%M:%S")

    def _period(where, extra=""):
        with _conn() as conn:
            q = f"SELECT COUNT(*) c, AVG(response_time_ms) a FROM analytics_events WHERE user_phone = ? {where} AND timestamp >= ? AND timestamp < ?"
            cur = conn.execute(q, (phone, cur_start, "9999-12-31 00:00:00")).fetchone()
            prev = conn.execute(q, (phone, prev_start, cur_start)).fetchone()
        return cur, prev

    deltas = {"days": days}
    for key, where in (("total_messages", ""), ("job_searches", "AND event_type='job_search'"),
                       ("quiz_attempts", "AND event_type='quiz_start'"),
                       ("resume_analyses", "AND event_type='ats_score' AND success=1")):
        cur, prev = _period(where)
        deltas[key] = cur["c"] - prev["c"]
    cur, prev = _period("AND response_time_ms IS NOT NULL")
    deltas["avg_response_time_s"] = (round((cur["a"] - prev["a"]) / 1000, 1)
                                     if cur["a"] is not None and prev["a"] is not None else None)

    peak_day = max(activity_rows, key=lambda r: r["messages"]) if activity_rows else None
    quiz_correct = sum(v[0] for v in quiz_topic_raw.values())
    quiz_questions = sum(v[1] for v in quiz_topic_raw.values())

    def _describe(r):
        m = json.loads(r["meta"] or "{}")
        et, cmd = r["event_type"], r["command"]
        if et == "job_search":
            return f"Searched for {m.get('topic') or 'jobs'}", "Job Search"
        if et in ("quiz_start", "quiz_complete", "quiz_exit_early"):
            return f"{'Completed' if et == 'quiz_complete' else 'Attempted'} {m.get('topic') or 'General'} Quiz", "Quiz"
        if et in ("resume_upload", "ats_score"):
            fn = m.get("filename")
            return (f"Uploaded resume ({fn})" if fn else ("Analyzed resume" + (f" (score {m['score']}%)" if isinstance(m.get("score"), (int, float)) else ""))), "Resume"
        if et == "command":
            return f"Used {cmd or 'command'}", "Command"
        return (cmd or et).replace("_", " ").capitalize(), "Event"

    return {
        "phone": phone,
        "deltas": deltas,
        "peak_day": {"date": peak_day["d"], "messages": peak_day["messages"]} if peak_day else None,
        "quiz_summary": {"correct": quiz_correct, "questions": quiz_questions,
                         "accuracy_pct": round(quiz_correct * 100 / quiz_questions, 1) if quiz_questions else None},
        "resume_formats": formats,
        "label": "User",
        "status": "Active",
        "joined_on": profile["joined_on"],
        "last_active": profile["last_active"],
        "overview": {
            "total_messages": total_msgs,
            "job_searches": job_searches,
            "quiz_attempts": quiz_attempts,
            "resume_analyses": resume_analyses,
            "avg_response_time_s": round(avg_rt / 1000, 1) if avg_rt else None,
        },
        "activity": [
            {"date": r["d"], "messages": r["messages"], "commands": r["commands"] or 0}
            for r in activity_rows
        ],
        "feature_usage": [
            {"command": r["command"], "count": r["c"]} for r in feature_rows
        ],
        "funnel": {
            "steps": [
                {"label": "Started Bot", "value": total_msgs},
                {"label": "Explored Features", "value": funnel_used_feature},
                {"label": "Used Job Search", "value": funnel_job_search},
                {"label": "Attempted Quiz", "value": quiz_attempts},
                {"label": "Applied Jobs", "value": funnel_applied},
            ]
        },
        "quick_stats": {
            "quiz_accuracy_pct": round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else None,
            "resume_best_score": max(resume_scores) if resume_scores else None,
            "peak_hour": int(peak_row["h"]) if peak_row else None,
            "most_used_command": feature_rows[0]["command"] if feature_rows else None,
            "active_days": active_days,
            "quiz_completion_rate": min(100.0, round(quiz_completed * 100 / quiz_attempts, 1)) if quiz_attempts else None,
        },
        "job_domains": job_domains,
        "quiz_topics": quiz_topics,
        "resume_stats": {
            "total": len(resume_scores),
            "avg": round(sum(resume_scores) / len(resume_scores), 1) if resume_scores else None,
            "best": max(resume_scores) if resume_scores else None,
            "distribution": r_dist,
            "distribution5": r_dist5,
        },
        "job_searches_list": [
            {"topic": json.loads(r["meta"] or "{}").get("topic", "—"), "timestamp": r["timestamp"]}
            for r in job_rows[:20]
        ],
        "quiz_list": [
            {
                "topic": json.loads(r["meta"] or "{}").get("topic", "—"),
                "score": json.loads(r["meta"] or "{}").get("score"),
                "total": json.loads(r["meta"] or "{}").get("total"),
                "status": "Passed" if r["event_type"] == "quiz_complete" else "Exited early",
                "timestamp": r["timestamp"],
            }
            for r in quiz_rows[:20]
        ],
        "resume_list": [
            {"score": json.loads(r["meta"] or "{}").get("score"), "timestamp": r["timestamp"]}
            for r in resume_rows[:20]
        ],
        "messages": [
            {
                "event_type": r["event_type"],
                "command": r["command"],
                "success": bool(r["success"]),
                "timestamp": r["timestamp"],
                "desc": _describe(r)[0],
                "tag": _describe(r)[1],
            }
            for r in recent_events
        ],
    }


# ---------------------------------------------------------------
# system health
# ---------------------------------------------------------------
import time as _time

_PROCESS_STARTED_AT = _time.time()
_HISTORY = []  # ring buffer of {t, cpu, mem, net} sampled every 30 s by a background thread
_HISTORY_MAX = 20  # 20 samples x 30 s = last 10 minutes
_SAMPLE_EVERY = 30
_NET_LAST = {"t": None, "bytes": None, "mbps": 0.0}
_NET_CAPACITY_MBPS = 12.5  # 100 Mbit/s link, used only to turn MB/s into the gauge percentage
_SAMPLER = {"started": False}


def _sample_once():
    """Take one CPU / memory / network sample (non-blocking) and append it to _HISTORY."""
    try:
        import psutil
    except ImportError:
        return None
    now = _time.time()
    net = psutil.net_io_counters()
    total = net.bytes_sent + net.bytes_recv
    if _NET_LAST["t"] is not None and now > _NET_LAST["t"]:
        _NET_LAST["mbps"] = max(0.0, (total - _NET_LAST["bytes"]) / (1024 ** 2) / (now - _NET_LAST["t"]))
    _NET_LAST["t"], _NET_LAST["bytes"] = now, total
    point = {
        "t": datetime.now().strftime("%H:%M"),
        "cpu": psutil.cpu_percent(interval=None),
        "mem": psutil.virtual_memory().percent,
        "net": round(min(100.0, _NET_LAST["mbps"] / _NET_CAPACITY_MBPS * 100), 1),
    }
    _HISTORY.append(point)
    del _HISTORY[:-_HISTORY_MAX]
    return point


def _ensure_sampler():
    if _SAMPLER["started"]:
        return
    _SAMPLER["started"] = True
    import threading

    def loop():
        while True:
            try:
                _sample_once()
            except Exception:
                pass
            _time.sleep(_SAMPLE_EVERY)

    _sample_once()
    threading.Thread(target=loop, daemon=True, name="system-sampler").start()



def system_metrics(log_limit: int = 5):
    import shutil
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
    _ensure_sampler()
    log_limit = max(1, min(int(log_limit or 5), 100))

    t0 = _time.perf_counter()
    with _conn() as conn:
        conn.execute("SELECT 1").fetchone()
    db_response_ms = round((_time.perf_counter() - t0) * 1000, 1)

    uptime_seconds = _time.time() - _PROCESS_STARTED_AT

    if has_psutil:
        cpu_pct = psutil.cpu_percent(interval=0.15)
        cores = psutil.cpu_count() or 1
        mem = psutil.virtual_memory()
        mem_pct = mem.percent
        mem_used_gb = round(mem.used / (1024 ** 3), 1)
        mem_total_gb = round(mem.total / (1024 ** 3), 1)
        net_mbps = round(_NET_LAST["mbps"], 2)
        net_pct = round(min(100.0, _NET_LAST["mbps"] / _NET_CAPACITY_MBPS * 100), 1)
        cpu_cores_used = round(cpu_pct * cores / 100, 1)
    else:
        cpu_pct = mem_pct = mem_used_gb = mem_total_gb = net_mbps = net_pct = cores = cpu_cores_used = None

    disk_total, disk_used, disk_free = shutil.disk_usage(".")
    disk_pct = round(disk_used * 100 / disk_total, 1)

    AI_TYPES = ("ats_score", "ai_assistance", "roadmap")
    with _conn() as conn:
        def stats(where, params=(), days=7):
            r = conn.execute(
                f"""SELECT COUNT(*) n, SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) ok,
                           AVG(response_time_ms) ms FROM analytics_events
                    WHERE timestamp >= datetime('now','-{days} day') AND {where}""", params).fetchone()
            n = r["n"] or 0
            return {"n": n, "rate": round((r["ok"] or 0) * 100 / n, 1) if n else None,
                    "ms": round(r["ms"]) if r["ms"] is not None else None}

        allev = stats("1=1")
        cmd = stats("event_type='command'")
        ai = stats("event_type IN (?,?,?)", AI_TYPES)
        jobs = stats("event_type='job_search'")
        recent_webhook = conn.execute(
            "SELECT COUNT(*) c FROM analytics_events WHERE timestamp >= datetime('now','-1 hour')").fetchone()["c"]
        hours_active = conn.execute(
            """SELECT COUNT(DISTINCT strftime('%Y-%m-%d %H', timestamp)) c FROM analytics_events
               WHERE timestamp >= datetime('now','-24 hour')""").fetchone()["c"]
        recent_logs = conn.execute(
            """SELECT event_type, command, meta, success, timestamp FROM analytics_events
               ORDER BY timestamp DESC, id DESC LIMIT ?""", (log_limit,)).fetchall()

    def up(rate):  # uptime proxy: share of successful events in the last 7 days
        return rate if rate is not None else 100.0

    ai_ok = ai["rate"] is None or ai["rate"] >= 80
    job_ok = jobs["rate"] is None or jobs["rate"] >= 80
    api_ms = round(db_response_ms + 1.0, 1)
    services = [
        {"name": "Bot API (FastAPI)", "key": "api", "status": "Healthy", "response_ms": api_ms, "uptime_pct": up(allev["rate"])},
        {"name": "Database (PostgreSQL)", "key": "db", "status": "Healthy" if db_response_ms < 500 else "Slow",
         "response_ms": db_response_ms, "uptime_pct": 100.0},
        {"name": "WhatsApp Webhook", "key": "whatsapp", "status": "Healthy" if recent_webhook > 0 else "Idle",
         "response_ms": cmd["ms"], "uptime_pct": min(100.0, round(hours_active * 100 / 24, 1))},
        {"name": "NVIDIA NIM (Kimi)", "key": "ai", "status": "Healthy" if ai_ok else "Degraded",
         "response_ms": ai["ms"], "uptime_pct": up(ai["rate"])},
        {"name": "External APIs", "key": "external", "status": "Healthy" if job_ok else "Degraded",
         "response_ms": jobs["ms"], "uptime_pct": up(jobs["rate"])},
    ]
    all_healthy = all(x["status"] in ("Healthy", "Idle") for x in services)

    logs = []
    for r in recent_logs:
        m = json.loads(r["meta"] or "{}")
        logs.append({
            "time": r["timestamp"],
            "success": bool(r["success"]),
            "level": "WARNING" if not r["success"] else ("SUCCESS" if r["event_type"] in
                     ("command", "job_search", "ats_score") else "INFO"),
            "message": m.get("message") or f"{r['event_type']}" + (f" — {r['command']}" if r["command"] else ""),
            "source": r["command"] or r["event_type"],
        })

    return {
        "all_operational": all_healthy,
        "bot_status": "Online",
        "uptime_pct": allev["rate"],
        "uptime_seconds": int(uptime_seconds),
        "database_status": "Healthy",
        "database_response_ms": db_response_ms,
        "web_server_status": "Healthy",
        "web_server_response_ms": api_ms,
        "whatsapp_status": "Connected" if recent_webhook > 0 else "Idle",
        "services": services,
        "resources": {
            "cpu_pct": cpu_pct,
            "cpu_cores": cores,
            "cpu_cores_used": cpu_cores_used,
            "memory_pct": mem_pct,
            "memory_used_gb": mem_used_gb,
            "memory_total_gb": mem_total_gb,
            "disk_pct": disk_pct,
            "disk_used_gb": round(disk_used / (1024 ** 3), 1),
            "disk_total_gb": round(disk_total / (1024 ** 3), 1),
            "network_pct": net_pct,
            "network_mbps": net_mbps,
        },
        "history": list(_HISTORY),
        "recent_logs": logs,
    }


DASHBOARD_CAREER_COMMANDS = (
    "/competitions", "/competition", "/internships", "/internship",
    "/prepare", "/job_search", "/roadmap", "/ats", "/ats_score",  # old name kept for history
)


def funnel():
    """Dashboard 'User Funnel': Users -> Used Features -> Career Features ->
    Job Search -> Applied Jobs (all distinct-user counts)."""
    with _conn() as conn:
        def distinct(where, params=()):
            return conn.execute(
                f"SELECT COUNT(DISTINCT user_phone) c FROM analytics_events WHERE {where}", params
            ).fetchone()["c"]

        total_users = distinct("1=1")
        used_feature = distinct("event_type = 'command'")
        career = distinct(
            f"event_type = 'command' AND lower(command) IN ({','.join('?' * len(DASHBOARD_CAREER_COMMANDS))})",
            DASHBOARD_CAREER_COMMANDS,
        )
        job_search = distinct("event_type = 'job_search'")
        applied = distinct("event_type = 'job_apply'")

    return {
        "steps": [
            {"label": "Users", "value": total_users},
            {"label": "Used Features", "value": used_feature},
            {"label": "Career Features", "value": career},
            {"label": "Job Search", "value": job_search},
            {"label": "Applied Jobs", "value": applied},
        ]
    }


# ---------------------------------------------------------------
# week-over-week / period-over-period trend (powers every "+12.3% vs last
# 7 days"-style delta chip across the dashboard, users, resume and quiz
# pages). Every number here comes from a real SQL count over a real date
# range — nothing is invented.
# ---------------------------------------------------------------
_TREND_EVENT_TYPE = {
    "messages": None,                 # count of all events
    "job_searches": "job_search",
    "quizzes": "quiz_complete",
    "resumes": "resume_upload",
    "feature_usage": "command",
}


def _cumulative_users_at(conn, end_date: str) -> int:
    """Distinct users whose first-ever event is on/before end_date."""
    return conn.execute(
        '''SELECT COUNT(*) c FROM (
               SELECT user_phone FROM analytics_events
               GROUP BY user_phone HAVING date(MIN(timestamp)) <= ?
           )''', (end_date,)
    ).fetchone()["c"]


def _metric_value(conn, metric: str, start: str, end: str):
    if metric == "total_users":
        return _cumulative_users_at(conn, end)
    if metric == "avg_response_ms":
        row = conn.execute(
            '''SELECT AVG(response_time_ms) v FROM analytics_events
               WHERE event_type = 'command' AND response_time_ms IS NOT NULL
                 AND date(timestamp) BETWEEN ? AND ?''', (start, end)
        ).fetchone()
        return round(row["v"], 0) if row["v"] is not None else 0
    if metric == "success_rate":
        row = conn.execute(
            '''SELECT
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) ok,
                   COUNT(*) total
               FROM analytics_events
               WHERE event_type IN ('command','ats_score','resume_upload','job_search','roadmap','quiz_start','ai_assistance')
                 AND date(timestamp) BETWEEN ? AND ?''', (start, end)
        ).fetchone()
        return round((row["ok"] / row["total"]) * 100, 1) if row["total"] else 100.0

    if metric in ("new_users", "active_users", "returning_users"):
        active = conn.execute(
            '''SELECT COUNT(DISTINCT user_phone) c FROM analytics_events
               WHERE date(timestamp) BETWEEN ? AND ?''', (start, end)).fetchone()["c"]
        new = conn.execute(
            '''SELECT COUNT(*) c FROM (SELECT user_phone FROM analytics_events
               GROUP BY user_phone HAVING date(MIN(timestamp)) BETWEEN ? AND ?)''', (start, end)).fetchone()["c"]
        return {"active_users": active, "new_users": new, "returning_users": max(active - new, 0)}[metric]

    if metric in ("cmd_internships", "cmd_competitions"):
        like = "/internship%" if metric == "cmd_internships" else "/competition%"
        return conn.execute(
            '''SELECT COUNT(*) c FROM analytics_events
               WHERE event_type = 'command' AND lower(command) LIKE ?
                 AND date(timestamp) BETWEEN ? AND ?''', (like, start, end)
        ).fetchone()["c"]

    event_type = _TREND_EVENT_TYPE.get(metric)
    if event_type is None and metric != "messages":
        raise ValueError(f"unknown trend metric: {metric}")
    if event_type:
        return conn.execute(
            '''SELECT COUNT(*) c FROM analytics_events
               WHERE event_type = ? AND date(timestamp) BETWEEN ? AND ?''',
            (event_type, start, end)
        ).fetchone()["c"]
    return conn.execute(
        '''SELECT COUNT(*) c FROM analytics_events WHERE date(timestamp) BETWEEN ? AND ?''',
        (start, end)
    ).fetchone()["c"]


def trend(metric: str, start: str, end: str):
    """Current value for [start, end] vs. the immediately preceding period
    of equal length, plus a %% delta. Used for every WoW/period delta chip."""
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end, "%Y-%m-%d").date()
    days = (d2 - d1).days + 1
    prev_end = d1 - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)

    with _conn() as conn:
        current = _metric_value(conn, metric, start, end)
        previous = _metric_value(conn, metric, prev_start.isoformat(), prev_end.isoformat())

    delta_pct = None
    direction = "flat"
    if previous not in (None, 0):
        delta_pct = round((current - previous) * 100 / previous, 1)
    elif current:
        delta_pct = 100.0
    if delta_pct is not None:
        direction = "up" if delta_pct > 0 else ("down" if delta_pct < 0 else "flat")

    return {
        "metric": metric,
        "current": current,
        "previous": previous,
        "delta_pct": delta_pct,
        "direction": direction,
    }


# ---------------------------------------------------------------
# Resume & Quiz detail views (ATS trend / recent analyses, quiz score
# distribution / topic table / recent attempts / per-user table).
# Everything is derived from the same analytics_events rows.
# ---------------------------------------------------------------
def _mask(phone: str) -> str:
    return ("••••" + phone[-4:]) if phone and len(phone) > 4 else (phone or "")


def resume_detail(start: str = None, end: str = None):
    clause, params = _range_clause(start=start, end=end)
    with _conn() as conn:
        rows = conn.execute(
            f'''SELECT user_phone, meta, success, timestamp FROM analytics_events
                WHERE event_type = 'ats_score' AND {clause}
                ORDER BY timestamp DESC, id DESC''', params
        ).fetchall()
        uploads = conn.execute(
            f"SELECT COUNT(*) c FROM analytics_events WHERE event_type='resume_upload' AND {clause}", params
        ).fetchone()["c"]

    by_day, recent = {}, []
    for r in rows:
        score = json.loads(r["meta"] or "{}").get("score")
        score = score if isinstance(score, (int, float)) else None
        if score is not None and r["success"]:
            by_day.setdefault(r["timestamp"][:10], []).append(score)
        if len(recent) < 10:
            recent.append({
                "phone": r["user_phone"], "masked": _mask(r["user_phone"]),
                "score": score, "success": bool(r["success"]), "time": r["timestamp"],
            })
    trend_pts = [{"date": d, "avg_score": round(sum(v) / len(v), 1), "count": len(v)}
                 for d, v in sorted(by_day.items())]
    return {"uploads_in_range": uploads, "ats_trend": trend_pts, "recent": recent}


def quiz_detail(start: str = None, end: str = None):
    clause, params = _range_clause(start=start, end=end)
    with _conn() as conn:
        rows = conn.execute(
            f'''SELECT user_phone, meta, timestamp FROM analytics_events
                WHERE event_type = 'quiz_complete' AND {clause}
                ORDER BY timestamp DESC, id DESC''', params
        ).fetchall()
        started_by_user = {r["user_phone"]: r["c"] for r in conn.execute(
            f'''SELECT user_phone, COUNT(*) c FROM analytics_events
                WHERE event_type = 'quiz_start' AND {clause} GROUP BY user_phone''', params)}
        topic_started = {}
        for r in conn.execute(
            f'''SELECT meta FROM analytics_events WHERE event_type='quiz_start' AND {clause}''', params):
            t = json.loads(r["meta"] or "{}").get("topic")
            if t:
                topic_started[t] = topic_started.get(t, 0) + 1

    buckets = {"0-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90-100": 0}
    topics, users, recent = {}, {}, []
    for r in rows:
        m = json.loads(r["meta"] or "{}")
        s, t, topic = m.get("score"), m.get("total"), m.get("topic") or "General"
        if not (isinstance(s, (int, float)) and isinstance(t, (int, float)) and t > 0):
            continue
        pct = s * 100 / t
        key = "0-59" if pct < 60 else "60-69" if pct < 70 else "70-79" if pct < 80 else "80-89" if pct < 90 else "90-100"
        buckets[key] += 1
        tp = topics.setdefault(topic, {"attempts": 0, "pcts": []})
        tp["attempts"] += 1
        tp["pcts"].append(pct)
        u = users.setdefault(r["user_phone"], {"pcts": [], "last": r["timestamp"]})
        u["pcts"].append(pct)
        if len(recent) < 10:
            recent.append({"phone": r["user_phone"], "masked": _mask(r["user_phone"]), "topic": topic,
                           "score": s, "total": t, "pct": round(pct, 1), "time": r["timestamp"]})

    topic_rows = []
    for name, v in topics.items():
        started = max(topic_started.get(name, 0), v["attempts"])
        topic_rows.append({
            "topic": name, "attempts": v["attempts"],
            "avg_score_pct": round(sum(v["pcts"]) / len(v["pcts"]), 1),
            "completion_rate": round(v["attempts"] * 100 / started, 1) if started else None,
        })
    topic_rows.sort(key=lambda x: (-x["attempts"], -x["avg_score_pct"]))

    user_rows = []
    for phone, v in users.items():
        started = max(started_by_user.get(phone, 0), len(v["pcts"]))
        user_rows.append({
            "phone": phone, "masked": _mask(phone), "attempts": len(v["pcts"]),
            "avg_score_pct": round(sum(v["pcts"]) / len(v["pcts"]), 1),
            "best_pct": round(max(v["pcts"]), 1),
            "completion_rate": round(len(v["pcts"]) * 100 / started, 1) if started else None,
            "last_attempt": v["last"],
        })
    user_rows.sort(key=lambda x: -x["attempts"])
    for i, u in enumerate(user_rows, 1):
        u["label"] = f"User {i}"

    return {"score_distribution": buckets, "topics": topic_rows[:10],
            "recent_attempts": recent, "users": user_rows[:50]}


# ---------------------------------------------------------------
# Job Search page: /job_search, /internships and /competitions usage
# ---------------------------------------------------------------
def opportunities_activity(start: str = None, end: str = None):
    clause, params = _range_clause(start=start, end=end)
    with _conn() as conn:
        jobs = conn.execute(
            f"SELECT COUNT(*) c FROM analytics_events WHERE event_type='job_search' AND {clause}", params
        ).fetchone()["c"]
        interns = conn.execute(
            f"SELECT COUNT(*) c FROM analytics_events WHERE event_type='command' AND lower(command) LIKE '/internship%' AND {clause}", params
        ).fetchone()["c"]
        comps = conn.execute(
            f"SELECT COUNT(*) c FROM analytics_events WHERE event_type='command' AND lower(command) LIKE '/competition%' AND {clause}", params
        ).fetchone()["c"]
        users = conn.execute(
            f'''SELECT COUNT(DISTINCT user_phone) c FROM analytics_events
               WHERE (event_type='job_search'
                      OR (event_type='command' AND (lower(command) LIKE '/internship%' OR lower(command) LIKE '/competition%')))
                 AND {clause}''', params
        ).fetchone()["c"]
        rows = conn.execute(
            f'''SELECT user_phone, event_type, command, meta, timestamp FROM analytics_events
               WHERE (event_type='job_search'
                      OR (event_type='command' AND (lower(command) LIKE '/internship%' OR lower(command) LIKE '/competition%')))
                 AND {clause}
               ORDER BY timestamp DESC, id DESC LIMIT 10''', params
        ).fetchall()
        topic_rows = conn.execute(
            f"SELECT meta FROM analytics_events WHERE event_type='job_search' AND {clause}", params
        ).fetchall()

    total = (jobs + interns + comps) or 1
    functions = [
        {"key": "job_search", "label": "/job_search", "count": jobs, "share_pct": round(jobs * 100 / total, 1)},
        {"key": "internships", "label": "/internships", "count": interns, "share_pct": round(interns * 100 / total, 1)},
        {"key": "competitions", "label": "/competitions", "count": comps, "share_pct": round(comps * 100 / total, 1)},
    ]

    topics = {}
    for r in topic_rows:
        t = json.loads(r["meta"] or "{}").get("topic")
        if t:
            topics[t] = topics.get(t, 0) + 1
    top_topics = [{"topic": t, "count": c} for t, c in sorted(topics.items(), key=lambda x: -x[1])[:5]]

    # stable "User N" labels shared with the Users page (ordered by last activity)
    activity = []
    for r in rows:
        is_job = r["event_type"] == "job_search"
        fn = "/job_search" if is_job else ("/internships" if "internship" in (r["command"] or "").lower() else "/competitions")
        activity.append({
            "phone": r["user_phone"], "masked": _mask(r["user_phone"]), "function": fn,
            "domain": json.loads(r["meta"] or "{}").get("topic") if is_job else None,
            "tag": "Search" if is_job else "Browse", "time": r["timestamp"],
        })
    return {"functions": functions, "unique_users": users, "top_topics": top_topics, "recent": activity}


# ---------------------------------------------------------------
# Feature Analytics page
# ---------------------------------------------------------------
FEATURE_META = [
    ("job_search", "Job Search", "Search for jobs by domain"),
    ("quiz", "Quiz", "Take career quizzes"),
    ("resume", "Resume & ATS", "Get resume analysis"),
    ("roadmap", "Career Roadmap", "Get learning roadmap"),
    ("internships", "Internships", "Find internships"),
    ("competitions", "Competitions", "Find competitions"),
    ("news", "News", "Get latest news"),
    ("other", "Other", "Miscellaneous commands"),
]
_FEATURE_LABELS = {k: (l, d) for k, l, d in FEATURE_META}
_DEDICATED_EVENT = {"job_search": "job_search", "quiz_start": "quiz", "ats_score": "resume", "roadmap": "roadmap"}
_COMMAND_FEATURE = {
    "/news": "news", "news": "news",
    "/competitions": "competitions", "/competition": "competitions", "competitions": "competitions",
    "/internships": "internships", "/internship": "internships", "internships": "internships",
    "/roadmap": "roadmap", "/prepare": "roadmap", "roadmap": "roadmap", "prepare": "roadmap",
    "/resume": "resume", "resume": "resume",
}
# commands whose use is already logged as a dedicated event (avoid double counting)
_COVERED_COMMANDS = {"/job_search", "job_search", "/quiz", "quiz", "/ats_score", "ats_score", "/ats"}
_FEATURE_CMD_LABEL = {"job_search": "/job_search", "quiz": "/quiz", "resume": "/resume", "roadmap": "/roadmap"}


def _feature_of(event_type, command):
    """Maps one analytics row to a feature key, or None if it isn't a feature use."""
    if event_type in _DEDICATED_EVENT:
        return _DEDICATED_EVENT[event_type]
    if event_type == "command" and command:
        c = command.strip().lower()
        if c in _COVERED_COMMANDS:
            return None
        if c in ("/start", "/help", "start", "help", "/menu"):
            return None
        return _COMMAND_FEATURE.get(c, "other")
    return None


def _feature_counts(conn, start, end):
    rows = conn.execute(
        '''SELECT date(timestamp) d, event_type, command FROM analytics_events
           WHERE date(timestamp) BETWEEN ? AND ? AND event_type IN
           ('command','job_search','quiz_start','ats_score','roadmap')''', (start, end)
    ).fetchall()
    total, by_day = {}, {}
    for r in rows:
        f = _feature_of(r["event_type"], r["command"])
        if not f:
            continue
        total[f] = total.get(f, 0) + 1
        by_day.setdefault(r["d"], {}).setdefault(f, 0)
        by_day[r["d"]][f] += 1
    return total, by_day


def features_detail(start: str = None, end: str = None):
    end = end or datetime.now().strftime("%Y-%m-%d")
    start = start or (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end, "%Y-%m-%d").date()
    n = (d2 - d1).days + 1
    prev_end = d1 - timedelta(days=1)
    prev_start = prev_end - timedelta(days=n - 1)

    with _conn() as conn:
        cur, by_day = _feature_counts(conn, start, end)
        prev, _ = _feature_counts(conn, prev_start.isoformat(), prev_end.isoformat())
        recent_rows = conn.execute(
            '''SELECT event_type, command, success, timestamp FROM analytics_events
               WHERE event_type IN ('command','job_search','quiz_start','ats_score','roadmap')
               ORDER BY timestamp DESC, id DESC LIMIT 60'''
        ).fetchall()

    total = sum(cur.values())
    features = []
    for key, label, desc in FEATURE_META:
        c, p = cur.get(key, 0), prev.get(key, 0)
        if c == 0 and p == 0 and key == "other":
            continue
        pct = None
        if p:
            pct = round((c - p) * 100 / p)
        elif c:
            pct = 100
        features.append({
            "key": key, "label": label, "description": desc, "count": c,
            "pct": round(c * 100 / total, 1) if total else 0.0,
            "trend_pct": pct,
            "trend": "flat" if not pct else ("up" if pct > 0 else "down"),
        })
    features.sort(key=lambda f: (-f["count"], f["key"] == "other"))

    labels, series = [], {"job_search": [], "quiz": [], "resume": [], "others": []}
    day = d1
    while day <= d2:
        iso = day.isoformat()
        labels.append(iso)
        row = by_day.get(iso, {})
        series["job_search"].append(row.get("job_search", 0))
        series["quiz"].append(row.get("quiz", 0))
        series["resume"].append(row.get("resume", 0))
        series["others"].append(sum(v for k, v in row.items() if k not in ("job_search", "quiz", "resume")))
        day += timedelta(days=1)

    recent = []
    for r in recent_rows:
        f = _feature_of(r["event_type"], r["command"])
        if not f:
            continue
        cmd = r["command"] if r["event_type"] == "command" and r["command"] else _FEATURE_CMD_LABEL.get(f, f)
        if not cmd.startswith("/"):
            cmd = "/" + cmd
        recent.append({"feature": f, "command": cmd, "success": bool(r["success"]), "timestamp": r["timestamp"]})
        if len(recent) == 5:
            break

    return {
        "total_usage": total,
        "available_commands": len(CAREER_FEATURE_COMMANDS - {"/competition", "/internship"}),
        "features": features,
        "series": {"labels": labels, **series},
        "recent": recent,
    }


# ---------------------------------------------------------------
# Resume & ATS page
# ---------------------------------------------------------------
def _ats_rows(conn, start, end):
    out = []
    for r in conn.execute(
        '''SELECT user_phone, meta, success, timestamp FROM analytics_events
           WHERE event_type = 'ats_score' AND date(timestamp) BETWEEN ? AND ?
           ORDER BY timestamp DESC, id DESC''', (start, end)):
        m = json.loads(r["meta"] or "{}")
        sc = m.get("score")
        out.append({"phone": r["user_phone"], "meta": m, "success": bool(r["success"]), "time": r["timestamp"],
                    "score": sc if isinstance(sc, (int, float)) else None})
    return out


def _ats_status(score):
    if score is None:
        return "Failed"
    return "Excellent" if score >= 85 else "Good" if score >= 70 else "Average" if score >= 60 else "Needs Improvement"


def resume_page(start: str = None, end: str = None):
    end = end or datetime.now().strftime("%Y-%m-%d")
    start = start or (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end, "%Y-%m-%d").date()
    n = (d2 - d1).days + 1
    pe = d1 - timedelta(days=1)
    ps = pe - timedelta(days=n - 1)

    with _conn() as conn:
        cur = _ats_rows(conn, start, end)
        prev = _ats_rows(conn, ps.isoformat(), pe.isoformat())

    def scored(rows):
        return [r for r in rows if r["success"] and r["score"] is not None]

    cs, pv = scored(cur), scored(prev)

    def avg(rows):
        return round(sum(r["score"] for r in rows) / len(rows), 1) if rows else None

    ca, pa = avg(cs), avg(pv)
    avg_delta = None
    if ca is not None and pa:
        avg_delta = round((ca - pa) * 100 / pa, 1)
    elif ca is not None:
        avg_delta = 100.0

    high = sum(1 for r in cs if r["score"] >= 80)
    low = sum(1 for r in cs if r["score"] < 60)
    phigh = sum(1 for r in pv if r["score"] >= 80)
    plow = sum(1 for r in pv if r["score"] < 60)

    bands = [("80 - 100", lambda s: s >= 80), ("70 - 79", lambda s: 70 <= s < 80),
             ("60 - 69", lambda s: 60 <= s < 70), ("Below 60", lambda s: s < 60)]
    distribution = []
    for label, fn in bands:
        c = sum(1 for r in cs if fn(r["score"]))
        distribution.append({"label": label, "count": c, "pct": round(c * 100 / len(cs), 1) if cs else 0.0})

    by_day = {}
    for r in cs:
        by_day.setdefault(r["time"][:10], []).append(r["score"])
    labels, avg_series, best_series = [], [], []
    day = d1
    while day <= d2:
        iso = day.isoformat()
        v = by_day.get(iso)
        labels.append(iso)
        avg_series.append(round(sum(v) / len(v), 1) if v else None)
        best_series.append(max(v) if v else None)
        day += timedelta(days=1)

    # optional detail the bot may log in meta: breakdown{content,keywords,skills,format}, skills[], filename, file_size_mb
    bsum, bcnt, skills = {}, {}, {}
    for r in cs:
        b = r["meta"].get("breakdown")
        if isinstance(b, dict):
            for k in ("content", "keywords", "skills", "format"):
                if isinstance(b.get(k), (int, float)):
                    bsum[k] = bsum.get(k, 0) + b[k]
                    bcnt[k] = bcnt.get(k, 0) + 1
        sk = r["meta"].get("skills") or r["meta"].get("key_skills")
        if isinstance(sk, list):
            for s in set(str(x).strip() for x in sk if str(x).strip()):
                skills[s] = skills.get(s, 0) + 1
    breakdown = [{"label": k.capitalize(), "value": round(bsum[k] / bcnt[k])} for k in ("content", "keywords", "skills", "format") if k in bsum]
    top_skills = [{"skill": s, "count": c, "pct": round(c * 100 / len(cs))} for s, c in sorted(skills.items(), key=lambda x: -x[1])[:5]]

    recent = []
    for r in cur[:20]:
        m = r["meta"]
        sk = m.get("skills") or m.get("key_skills")
        fn = m.get("filename")
        ext = fn.rsplit(".", 1)[-1].lower() if isinstance(fn, str) and "." in fn else None
        recent.append({
            "phone": r["phone"], "score": r["score"], "status": _ats_status(r["score"] if r["success"] else None),
            "filename": fn, "ext": ext, "size_mb": m.get("file_size_mb"),
            "skills": ", ".join(str(x) for x in sk[:3]) if isinstance(sk, list) else None,
            "time": r["time"],
        })

    return {
        "total": len(cs), "total_delta": len(cs) - len(pv),
        "avg_score": ca, "avg_delta_pct": avg_delta,
        "high": high, "high_delta": high - phigh,
        "low": low, "low_delta": low - plow,
        "distribution": distribution,
        "trend": {"labels": labels, "avg": avg_series, "best": best_series},
        "breakdown": breakdown, "top_skills": top_skills,
        "top_skill": top_skills[0]["skill"] if top_skills else None,
        "recent": recent,
    }


# ---------------------------------------------------------------
# Quiz Analytics page
# ---------------------------------------------------------------
def _user_label_map(conn):
    rows = conn.execute(
        "SELECT user_phone FROM analytics_events GROUP BY user_phone ORDER BY MAX(timestamp) DESC, user_phone"
    ).fetchall()
    return {r["user_phone"]: f"User {i}" for i, r in enumerate(rows, 1)}


def _quiz_period(conn, start, end, topic):
    rows = conn.execute(
        '''SELECT user_phone, event_type, meta, timestamp FROM analytics_events
           WHERE event_type IN ('quiz_start','quiz_complete','quiz_exit_early')
             AND date(timestamp) BETWEEN ? AND ? ORDER BY timestamp, id''', (start, end)
    ).fetchall()
    starts, completes, exits = [], [], []
    for r in rows:
        m = json.loads(r["meta"] or "{}")
        t = m.get("topic") or "General"
        if topic and t != topic:
            continue
        item = {"phone": r["user_phone"], "topic": t, "ts": r["timestamp"], "meta": m}
        if r["event_type"] == "quiz_start":
            starts.append(item)
        elif r["event_type"] == "quiz_complete":
            sc, tt = m.get("score"), m.get("total")
            item["pct"] = sc * 100 / tt if isinstance(sc, (int, float)) and isinstance(tt, (int, float)) and tt > 0 else None
            item["score"], item["total"] = sc, tt
            completes.append(item)
        else:
            exits.append(item)
    # time taken: meta.duration_s, else gap to the latest earlier unmatched start by the same user
    used, durations = set(), []
    for c in completes:
        d = c["meta"].get("duration_s")
        if isinstance(d, (int, float)):
            durations.append(d / 60)
            c["minutes"] = d / 60
            continue
        best = None
        for i, s in enumerate(starts):
            if i in used or s["phone"] != c["phone"] or s["ts"] > c["ts"]:
                continue
            best = i
        if best is not None:
            gap = (datetime.fromisoformat(c["ts"]) - datetime.fromisoformat(starts[best]["ts"])).total_seconds() / 60
            if 0 < gap <= 180:
                used.add(best)
                durations.append(gap)
                c["minutes"] = gap
    attempts = max(len(starts), len(completes) + len(exits))           # a quiz that was left also counts as an attempt
    pcts = [c["pct"] for c in completes if c["pct"] is not None]
    return {
        "starts": starts, "completes": completes, "exits": exits,
        "attempts": attempts, "completed": len(completes), "left": max(attempts - len(completes), 0),
        "avg_pct": round(sum(pcts) / len(pcts), 1) if pcts else None,
        "avg_min": round(sum(durations) / len(durations), 1) if durations else None,
    }


def _pct_change(cur, prev):
    if cur is None:
        return None
    if prev:
        return round((cur - prev) * 100 / prev, 1)
    return 100.0 if cur else None


def quiz_page(start: str = None, end: str = None, topic: str = None):
    end = end or datetime.now().strftime("%Y-%m-%d")
    start = start or (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end, "%Y-%m-%d").date()
    n = (d2 - d1).days + 1
    pe = d1 - timedelta(days=1)
    ps = pe - timedelta(days=n - 1)
    topic = topic or None

    with _conn() as conn:
        labels = _user_label_map(conn)
        cur = _quiz_period(conn, start, end, topic)
        prev = _quiz_period(conn, ps.isoformat(), pe.isoformat(), topic)
        opts = {}
        for r in conn.execute("SELECT meta FROM analytics_events WHERE event_type IN ('quiz_start','quiz_complete')"):
            m = json.loads(r["meta"] or "{}")
            t = m.get("topic") or "General"
            if isinstance(m.get("total"), (int, float)):
                opts[t] = int(m["total"])
            else:
                opts.setdefault(t, None)

    ac, ap = cur["attempts"], prev["attempts"]
    comp_rate = round(cur["completed"] * 100 / ac, 1) if ac else None
    left_rate = round(cur["left"] * 100 / ac, 1) if ac else None

    # daily trend
    by_day = {}
    for s in cur["starts"]:
        by_day.setdefault(s["ts"][:10], {"s": 0, "c": 0})["s"] += 1
    for c in cur["completes"]:
        by_day.setdefault(c["ts"][:10], {"s": 0, "c": 0})["c"] += 1
    t_labels, t_done, t_left = [], [], []
    day = d1
    while day <= d2:
        iso = day.isoformat()
        v = by_day.get(iso, {"s": 0, "c": 0})
        t_labels.append(iso)
        t_done.append(v["c"])
        t_left.append(max(max(v["s"], v["c"]) - v["c"], 0))
        day += timedelta(days=1)

    # per-user aggregates
    users = {}
    for s in cur["starts"]:
        users.setdefault(s["phone"], {"starts": 0, "done": [], "recent": []})["starts"] += 1
    for c in cur["completes"]:
        u = users.setdefault(c["phone"], {"starts": 0, "done": [], "recent": []})
        u["done"].append(c)
    prev_users = {}
    for s in prev["starts"]:
        prev_users[s["phone"]] = prev_users.get(s["phone"], 0) + 1

    # score distribution = users bucketed by their average score
    bucket_keys = ["0-59", "60-69", "70-79", "80-89", "90-100"]
    dist = {k: 0 for k in bucket_keys}
    user_rows = []
    for phone, u in users.items():
        pcts = [c["pct"] for c in u["done"] if c["pct"] is not None]
        avg = round(sum(pcts) / len(pcts), 1) if pcts else None
        if avg is not None:
            dist["0-59" if avg < 60 else "60-69" if avg < 70 else "70-79" if avg < 80 else "80-89" if avg < 90 else "90-100"] += 1
        attempts = max(u["starts"], len(u["done"]))
        mins = [c["minutes"] for c in u["done"] if "minutes" in c]
        recent = sorted(u["done"], key=lambda c: c["ts"], reverse=True)[:5]
        topics_pct = {}
        for c in u["done"]:
            if c["pct"] is not None:
                topics_pct.setdefault(c["topic"], []).append(c["pct"])
        user_rows.append({
            "phone": phone, "masked": _mask(phone), "label": labels.get(phone, "User"),
            "attempts": attempts, "attempts_delta": attempts - prev_users.get(phone, 0),
            "completed": len(u["done"]),
            "completed_pct": round(len(u["done"]) * 100 / attempts) if attempts else 0,
            "avg_pct": avg, "avg_min": round(sum(mins) / len(mins), 1) if mins else None,
            "weak_topics": sorted([{"topic": t, "avg_pct": round(sum(v) / len(v), 1)} for t, v in topics_pct.items()], key=lambda x: x["avg_pct"])[:5],
            "recent": [{"topic": c["topic"], "pct": round(c["pct"]) if c["pct"] is not None else None,
                        "status": "Passed" if (c["pct"] or 0) >= 60 else "Failed", "time": c["ts"], "phone": phone} for c in recent],
        })
    user_rows.sort(key=lambda x: (-x["attempts"], -(x["avg_pct"] or 0)))
    for i, u in enumerate(user_rows, 1):
        u["rank"] = i

    # topics table
    tp = {}
    for s in cur["starts"]:
        tp.setdefault(s["topic"], {"starts": 0, "pcts": []})["starts"] += 1
    for c in cur["completes"]:
        t = tp.setdefault(c["topic"], {"starts": 0, "pcts": []})
        if c["pct"] is not None:
            t["pcts"].append(c["pct"])
        t["done"] = t.get("done", 0) + 1
    topic_rows = []
    for name, v in tp.items():
        done = v.get("done", 0)
        att = max(v["starts"], done)
        topic_rows.append({"topic": name, "attempts": att,
                           "avg_pct": round(sum(v["pcts"]) / len(v["pcts"]), 1) if v["pcts"] else None,
                           "completion_pct": round(done * 100 / att, 1) if att else None})
    topic_rows.sort(key=lambda x: (-x["attempts"], -(x["avg_pct"] or 0)))

    # recent attempts: completed + exited early
    events = [dict(c, kind="done") for c in cur["completes"]] + [dict(e, kind="left") for e in cur["exits"]]
    events.sort(key=lambda e: e["ts"], reverse=True)
    recent = []
    for e in events[:20]:
        pct = e.get("pct")
        detail = ""
        if e["kind"] == "left":
            # the quiz was left before the end: show the score the user had so far (correct answers / questions answered)
            sc, att, tot = e["meta"].get("score"), e["meta"].get("attempted"), e["meta"].get("total")
            ok_ = isinstance(sc, (int, float)) and isinstance(att, (int, float))
            pct = sc * 100 / att if ok_ and att else (0 if ok_ else None)
            detail = (f"{int(sc)} correct of {int(att)} answered" + (f" ({int(tot)} questions)" if isinstance(tot, (int, float)) and tot else "")) if ok_ else ""
            status = "Left"
        else:
            status = "Passed" if (pct or 0) >= 60 else "Failed"
        recent.append({"phone": e["phone"], "label": labels.get(e["phone"], "User"), "topic": e["topic"],
                       "pct": round(pct) if pct is not None else None, "status": status, "detail": detail, "time": e["ts"]})

    return {
        "options": [{"topic": t, "questions": q} for t, q in sorted(opts.items())],
        "selected": topic,
        "attempts": ac, "attempts_delta_pct": _pct_change(ac, ap),
        "completed": cur["completed"], "completed_rate": comp_rate,
        "completed_delta_pct": _pct_change(cur["completed"], prev["completed"]),
        "left": cur["left"], "left_rate": left_rate, "left_delta_pct": _pct_change(cur["left"], prev["left"]),
        "avg_score": cur["avg_pct"], "avg_score_delta_pct": _pct_change(cur["avg_pct"], prev["avg_pct"]),
        "avg_min": cur["avg_min"], "avg_min_delta_pct": _pct_change(cur["avg_min"], prev["avg_min"]),
        "trend": {"labels": t_labels, "completed": t_done, "left": t_left},
        "distribution": dist,
        "topics": topic_rows[:10],
        "recent": recent,
        "users": user_rows[:50],
    }


# ---------------------------------------------------------------
# Job Search page (/job_search, /internships, /competitions)
# ---------------------------------------------------------------
def _opp_counts(conn, start, end):
    r = conn.execute(
        '''SELECT
             SUM(CASE WHEN event_type='job_search' THEN 1 ELSE 0 END) jobs,
             SUM(CASE WHEN event_type='command' AND lower(command) LIKE '%internship%' THEN 1 ELSE 0 END) interns,
             SUM(CASE WHEN event_type='command' AND lower(command) LIKE '%competition%' THEN 1 ELSE 0 END) comps
           FROM analytics_events WHERE date(timestamp) BETWEEN ? AND ?''', (start, end)).fetchone()
    users = conn.execute(
        '''SELECT COUNT(DISTINCT user_phone) c FROM analytics_events
           WHERE (event_type='job_search' OR (event_type='command' AND (lower(command) LIKE '%internship%' OR lower(command) LIKE '%competition%')))
             AND date(timestamp) BETWEEN ? AND ?''', (start, end)).fetchone()["c"]
    return {"jobs": r["jobs"] or 0, "interns": r["interns"] or 0, "comps": r["comps"] or 0, "users": users}


def _domain_of(meta):
    return meta.get("topic") or meta.get("domain") or meta.get("category") or meta.get("query")


def opportunities_page(start: str = None, end: str = None):
    end = end or datetime.now().strftime("%Y-%m-%d")
    start = start or (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
    d1 = datetime.strptime(start, "%Y-%m-%d").date()
    d2 = datetime.strptime(end, "%Y-%m-%d").date()
    n = (d2 - d1).days + 1
    pe = d1 - timedelta(days=1)
    ps = pe - timedelta(days=n - 1)

    with _conn() as conn:
        cur = _opp_counts(conn, start, end)
        prev = _opp_counts(conn, ps.isoformat(), pe.isoformat())
        labels = _user_label_map(conn)
        rows = conn.execute(
            '''SELECT user_phone, event_type, command, meta, timestamp FROM analytics_events
               WHERE (event_type='job_search' OR (event_type='command' AND (lower(command) LIKE '%internship%' OR lower(command) LIKE '%competition%')))
                 AND date(timestamp) BETWEEN ? AND ? ORDER BY timestamp DESC, id DESC''', (start, end)).fetchall()

    total = (cur["jobs"] + cur["interns"] + cur["comps"]) or 1
    def fn(key, label, c, p):
        return {"key": key, "label": label, "count": c, "share_pct": round(c * 100 / total, 1),
                "delta_pct": _pct_change(c, p)}
    functions = [fn("job_search", "/job_search", cur["jobs"], prev["jobs"]),
                 fn("internships", "/internships", cur["interns"], prev["interns"]),
                 fn("competitions", "/competitions", cur["comps"], prev["comps"])]

    job_t, int_t, comps, recent = {}, {}, [], []
    for r in rows:
        m = json.loads(r["meta"] or "{}")
        cmd = (r["command"] or "").lower()
        if r["event_type"] == "job_search":
            kind, dom = "job_search", _domain_of(m)
            if dom:
                job_t[dom] = job_t.get(dom, 0) + 1
        elif "internship" in cmd:
            kind, dom = "internships", _domain_of(m)
            if dom:
                int_t[dom] = int_t.get(dom, 0) + 1
        else:
            kind, dom = "competitions", _domain_of(m)
            items = m.get("competitions") if isinstance(m.get("competitions"), list) else [m]
            for it in items:
                if isinstance(it, dict) and (it.get("name") or it.get("title")) and len(comps) < 5:
                    comps.append({"name": it.get("name") or it.get("title"),
                                  "domain": it.get("domain") or it.get("category") or dom, "date": r["timestamp"]})
        if len(recent) < 10:
            recent.append({"phone": r["user_phone"], "label": labels.get(r["user_phone"], "User"),
                           "function": "/" + kind if kind != "job_search" else "/job_search",
                           "domain": dom, "tag": "Search" if kind != "competitions" else "Browse", "time": r["timestamp"]})

    def top5(d, denom):
        return [{"topic": t, "count": c, "pct": round(c * 100 / denom, 1) if denom else 0.0}
                for t, c in sorted(d.items(), key=lambda x: -x[1])[:5]]

    return {
        "functions": functions,
        "users": cur["users"], "users_delta_pct": _pct_change(cur["users"], prev["users"]),
        "job_total": cur["jobs"], "job_domains": top5(job_t, cur["jobs"]),
        "intern_total": cur["interns"], "intern_domains": top5(int_t, cur["interns"]),
        "competitions": comps, "recent": recent,
    }
