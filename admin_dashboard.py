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

class LoginRequest(BaseModel):
    username: str
    password: str

def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

async def require_admin(authorization: Optional[str] = Header(None)) -> str:
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

@router.post("/api/auth/login")
async def login(credentials: LoginRequest, request: Request):
    client_ip = get_client_ip(request)
    username_clean = credentials.username.strip().lower()

    if is_rate_limited(client_ip) or is_rate_limited(username_clean):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many failed login attempts.")

    if not authenticate_admin(username_clean, credentials.password):
        record_failed_attempt(client_ip)
        record_failed_attempt(username_clean)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    clear_failed_attempts(client_ip)
    clear_failed_attempts(username_clean)

    session_token = create_session(username_clean, duration_hours=24)
    return {"status": "success", "token": session_token, "token_type": "bearer", "username": username_clean}

@router.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
        terminate_session(token)
    return {"status": "success", "message": "Logged out successfully."}

@router.get("/api/auth/verify")
async def verify_token(current_user: str = Depends(require_admin)):
    return {"status": "valid", "username": current_user}

# --- Analytics Endpoints ---


@router.get("/api/analytics/overview")
async def get_overview(date: Optional[str] = None, current_user: str = Depends(require_admin)):
    target_date = date if date else datetime.utcnow().strftime("%Y-%m-%d")
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM users;")
            total_users = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(DISTINCT user_phone) as count FROM analytics_events WHERE DATE(timestamp) = %s;", (target_date,))
            today_users = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(DISTINCT user_phone) as count FROM analytics_events WHERE timestamp >= NOW() - INTERVAL '7 days';")
            active_users_7d = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM processed_messages;")
            messages_processed = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM analytics_events WHERE command IN ('/assistance', '/roadmap', '/ats_score', '/prepare', '/news', '/job_search');")
            ai_requests = cur.fetchone()["count"]

    return {
        "total_users": total_users,
        "today_users": today_users,
        "active_users_7d": active_users_7d,
        "messages_processed": messages_processed,
        "ai_requests": ai_requests
    }

@router.get("/api/analytics/new-returning")
async def get_new_returning(date: Optional[str] = None, current_user: str = Depends(require_admin)):
    target_date = date if date else datetime.utcnow().strftime("%Y-%m-%d")
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM users WHERE DATE(created_at) = %s;", (target_date,))
            new_users = cur.fetchone()["count"]

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
                SELECT TO_CHAR(created_at, 'Mon YYYY') as month, COUNT(*) as users
                FROM users
                WHERE created_at >= NOW() - (%s || ' months')::INTERVAL
                GROUP BY TO_CHAR(created_at, 'Mon YYYY'), DATE_TRUNC('month', created_at)
                ORDER BY DATE_TRUNC('month', created_at) ASC;
            """, (months,))
            rows = [dict(r) for r in cur.fetchall()]
    return {"data": rows}

@router.get("/api/analytics/daily")
async def get_daily(start: Optional[str] = None, end: Optional[str] = None, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(timestamp)::text as date, COUNT(DISTINCT user_phone) as users
                FROM analytics_events
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(timestamp)
                ORDER BY date ASC;
            """)
            rows = [dict(r) for r in cur.fetchall()]
    return {"data": rows}

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
            rows = [dict(r) for r in cur.fetchall()]
    return {"data": rows}

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
            rows = [dict(r) for r in cur.fetchall()]
    return {"usage": rows, "most_used": rows}

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
        "steps": [
            {"label": "Total Users", "value": total_u},
            {"label": "Executed Command", "value": command_u},
            {"label": "Attempted Quiz", "value": quiz_u}
        ]
    }



@router.get("/api/analytics/jobs")
async def get_jobs_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            # Count total job search executions
            cur.execute("SELECT COUNT(*) as count FROM analytics_events WHERE command = '/job_search';")
            searches = cur.fetchone()["count"]

            # Count unique users who searched
            cur.execute("SELECT COUNT(DISTINCT user_phone) as count FROM analytics_events WHERE command = '/job_search';")
            unique_users = cur.fetchone()["count"]

            # Query most frequent domains/topics recorded in event_details
            cur.execute("""
                SELECT 
                    INITCAP(TRIM(event_details)) as topic,
                    COUNT(*) as count
                FROM analytics_events
                WHERE command = '/job_search' 
                  AND event_details IS NOT NULL 
                  AND TRIM(event_details) != ''
                GROUP BY INITCAP(TRIM(event_details))
                ORDER BY count DESC
                LIMIT 5;
            """)
            rows = cur.fetchall()

            # Map results to popular_topics expected by dashboard.js
            popular_topics = [
                {"topic": r["topic"], "count": r["count"]} for r in rows
            ]

    return {
        "searches": searches,
        "unique_users": unique_users,
        "popular_topics": popular_topics
    }



@router.get("/api/analytics/resume")
async def get_resume_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total, COALESCE(AVG(ats_score), 0) as avg_score, COALESCE(MAX(ats_score), 0) as highest_score FROM resume_analytics;")
            row = cur.fetchone()
            total = row["total"] or 0
            avg_score = round(float(row["avg_score"]), 1) if total > 0 else None
            highest_score = round(float(row["highest_score"]), 1) if total > 0 else None
    return {
        "resumes_uploaded": total,
        "ats_analyses": total,
        "average_score": avg_score,
        "highest_score": highest_score,
        "score_distribution": {"0-50%": 0, "50-80%": total, "80-100%": 0}
    }

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
            started = row["total_quizzes"] or 0
            completed = row["completed_quizzes"] or 0
            avg_score = round(float(row["avg_score"]), 1) if started > 0 else None
            completion_rate = round((completed / started * 100), 1) if started > 0 else 0
            attempted = row["total_attempted"] or 0

    return {
        "quizzes_started": started,
        "quizzes_completed": completed,
        "questions_answered": attempted,
        "average_score_pct": avg_score,
        "completion_rate": completion_rate,
        "popular_topics": [{"topic": "Aptitude / General", "count": started}]
    }

@router.get("/api/analytics/ai")
async def get_ai_analytics(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM analytics_events WHERE command IN ('/assistance', '/roadmap', '/ats_score');")
            requests_count = cur.fetchone()["count"]
    return {
        "ai_requests": requests_count,
        "resume_analyses": 0,
        "ai_assistance_chats": requests_count,
        "roadmap_generations": 0,
        "quiz_generations": 0
    }

@router.get("/api/analytics/performance")
async def get_performance(current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total, COALESCE(AVG(response_time), 0.0) as avg_rt FROM analytics_events;")
            row = cur.fetchone()
            total = row["total"] or 0
            avg_rt = round(float(row["avg_rt"]) * 1000, 2)
    return {
        "messages_received": total,
        "messages_processed": total,
        "failed_messages": 0,
        "success_rate": 100.0,
        "avg_response_time_ms": avg_rt
    }

@router.get("/api/analytics/errors")
async def get_errors(limit: int = 10, current_user: str = Depends(require_admin)):
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM system_errors;")
            total_errs = cur.fetchone()["total"]

            cur.execute("SELECT timestamp::text as time, endpoint as stage, error_message as message FROM system_errors ORDER BY timestamp DESC LIMIT %s;", (limit,))
            rows = [dict(r) for r in cur.fetchall()]
    return {
        "errors_today": total_errs,
        "recent": rows
    }

@router.get("/api/analytics/logs")
async def get_analytics_logs(current_user: str = Depends(require_admin)) -> List[Dict[str, Any]]:
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
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database query error: {str(e)}")
