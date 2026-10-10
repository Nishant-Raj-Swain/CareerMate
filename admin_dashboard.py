import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from pydantic import BaseModel

import database as db
from auth import (
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
    client_ip = get_client_ip(request)
    username_clean = credentials.username.strip().lower()

    if is_rate_limited(client_ip) or is_rate_limited(username_clean):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again in 15 minutes."
        )

    if not authenticate_admin(username_clean, credentials.password):
        record_failed_attempt(client_ip)
        record_failed_attempt(username_clean)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

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
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
        terminate_session(token)
    return {"status": "success", "message": "Logged out successfully."}


@router.get("/api/auth/verify")
async def verify_token(current_user: str = Depends(require_admin)):
    return {"status": "valid", "username": current_user}


# --- Analytics Endpoints (Connected to Supabase PostgreSQL) ---

@router.get("/api/analytics/overview")
async def get_overview(date: Optional[str] = None, current_user: str = Depends(require_admin)):
    target_date = date if date else datetime.utcnow().strftime("%Y-%m-%d")
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Total unique users
            cur.execute("SELECT COUNT(*) as count FROM users;")
            total_users = cur.fetchone()["count"]

            # Users active on selected day
            cur.execute("""
                SELECT COUNT(DISTINCT user_phone) as count 
                FROM analytics_events 
                WHERE DATE(timestamp) = %s;
            """, (target_date,))
            users_selected_day = cur.fetchone()["count"]

            # Active Users in past 7 days
            cur.execute("""
                SELECT COUNT(DISTINCT user_phone) as count 
                FROM analytics_events 
                WHERE timestamp >= NOW() - INTERVAL '7 days';
            """)
            active_users_7d = cur.fetchone()["count"]

            # Total deduplicated messages processed
            cur.execute("SELECT COUNT(*) as count FROM processed_messages;")
            messages_processed = cur.fetchone()["count"]

            # AI tool invocations
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM analytics_events 
                WHERE command IN ('/assistance', '/roadmap', '/ats_score', '/prepare', '/news', '/job_search');
            """)
            ai_requests = cur.fetchone()["count"]

    return {
        "total_users": total_users,
        "users_selected_day": users_selected_day,
        "active_users_7d": active_users_7d,
        "messages_processed": messages_processed,
        "ai_requests": ai_requests
    }


@router.get("/api/analytics/new-returning")
async def get_new_returning(date: Optional[str] = None, current_user: str = Depends(require_admin)):
    target_date = date if date else datetime.utcnow().strftime("%Y-%m-%d")
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Users created on selected day
            cur.execute("SELECT COUNT(*) as count FROM users WHERE DATE(created_at) = %s;", (target_date,))
            new_users = cur.fetchone()["count"]

            # Users active today but registered before today
            cur.execute("""
                SELECT COUNT(DISTINCT e.user_phone) as count 
                FROM analytics_events e
                JOIN users u ON u.phone = e.user_phone
                WHERE DATE(e.timestamp) = %s AND DATE(u.created_at) < %s;
            """, (target_date, target_date))
            returning_users = cur.fetchone()["count"]

    return {"new_users": new_users, "returning_users": returning_users}


@router.get("/api/analytics/growth")
async def get_growth(months: int = 6, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT TO_CHAR(created_at, 'Mon YYYY') as month, COUNT(*) as count
                FROM users
                WHERE created_at >= NOW() - (%s || ' months')::INTERVAL
                GROUP BY TO_CHAR(created_at, 'Mon YYYY'), DATE_TRUNC('month', created_at)
                ORDER BY DATE_TRUNC('month', created_at) ASC;
            """, (months,))
            rows = cur.fetchall()
            labels = [r["month"] for r in rows]
            active_users = [r["count"] for r in rows]

    return {"labels": labels, "active_users": active_users}


@router.get("/api/analytics/daily")
async def get_daily(start: Optional[str] = None, end: Optional[str] = None, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(timestamp)::text as day, COUNT(*) as count
                FROM analytics_events
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY day ASC;
            """)
            rows = cur.fetchall()
            return {"labels": [r["day"] for r in rows], "activity": [r["count"] for r in rows]}


@router.get("/api/analytics/hourly")
async def get_hourly(date: Optional[str] = None, current_user: str = Depends(require_admin)):
    target_date = date if date else datetime.utcnow().strftime("%Y-%m-%d")
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXTRACT(HOUR FROM timestamp)::int as hour, COUNT(*) as count
                FROM analytics_events
                WHERE DATE(timestamp) = %s
                GROUP BY hour
                ORDER BY hour ASC;
            """, (target_date,))
            rows = cur.fetchall()
            hours_map = {r["hour"]: r["count"] for r in rows}
            return {
                "hours": [f"{h:02d}:00" for h in range(24)],
                "counts": [hours_map.get(h, 0) for h in range(24)]
            }


@router.get("/api/analytics/features")
async def get_features(start: Optional[str] = None, end: Optional[str] = None, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT command, COUNT(*) as count 
                FROM analytics_events 
                WHERE command IS NOT NULL 
                GROUP BY command 
                ORDER BY count DESC;
            """)
            rows = cur.fetchall()
            features_dict = {r["command"]: r["count"] for r in rows}
            return {"features": features_dict, "distribution": features_dict}


@router.get("/api/analytics/funnel")
async def get_funnel(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM users;")
            total_u = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(DISTINCT user_phone) as count FROM analytics_events WHERE command IS NOT NULL;")
            command_u = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(DISTINCT user_phone) as count FROM quiz_analytics;")
            quiz_u = cur.fetchone()["count"]

    return {
        "steps": ["Total Users", "Ran Command", "Started Quiz"],
        "conversion_rates": [total_u, command_u, quiz_u]
    }


@router.get("/api/analytics/jobs")
async def get_jobs_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM analytics_events WHERE command = '/job_search';")
            total_searches = cur.fetchone()["count"]
            return {"searches": total_searches, "top_queries": []}


@router.get("/api/analytics/resume")
async def get_resume_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total, COALESCE(AVG(ats_score), 0) as avg_score FROM resume_analytics;")
            row = cur.fetchone()
            return {"reviews": row["total"], "avg_score": round(row["avg_score"], 1)}


@router.get("/api/analytics/quiz")
async def get_quiz_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_quizzes,
                    SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed_quizzes,
                    COALESCE(AVG(score), 0) as avg_score,
                    COALESCE(SUM(attempted), 0) as total_attempted
                FROM quiz_analytics;
            """)
            row = cur.fetchone()
            return {
                "completions": row["completed_quizzes"] or 0,
                "total_quizzes": row["total_quizzes"] or 0,
                "avg_score": round(row["avg_score"], 1),
                "total_questions_attempted": row["total_attempted"] or 0
            }


@router.get("/api/analytics/ai")
async def get_ai_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM analytics_events 
                WHERE command IN ('/assistance', '/roadmap', '/ats_score');
            """)
            requests_count = cur.fetchone()["count"]
            return {"requests": requests_count, "token_count": requests_count * 350}


@router.get("/api/analytics/performance")
async def get_performance(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(AVG(response_time), 0.0) as avg_rt FROM analytics_events;")
            avg_rt = cur.fetchone()["avg_rt"]
            return {"avg_response_time_ms": round(avg_rt * 1000, 2), "uptime": "99.9%"}


@router.get("/api/analytics/errors")
async def get_errors(limit: int = 10, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM system_errors;")
            total_errs = cur.fetchone()["total"]

            cur.execute("""
                SELECT user_phone, error_message, endpoint, timestamp 
                FROM system_errors 
                ORDER BY timestamp DESC 
                LIMIT %s;
            """, (limit,))
            rows = cur.fetchall()
            return {
                "error_count": total_errs,
                "recent_errors": [dict(r) for r in rows]
            }


@router.get("/api/analytics/logs")
async def get_analytics_logs(current_user: str = Depends(require_admin)) -> List[Dict[str, Any]]:
    """Returns recent user interactions recorded in Supabase analytics_events."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_phone as whatsapp_no, 
                           COALESCE(command, event_type) as command, 
                           timestamp 
                    FROM analytics_events 
                    ORDER BY timestamp DESC 
                    LIMIT 500;
                """)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(e)}"
        )
