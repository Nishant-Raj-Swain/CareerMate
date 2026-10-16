# admin_dashboard.py
#
# API surface consumed by the separate Flask admin panel. Two things
# changed from the original version:
#
# 1. Authentication now goes through auth.py's real PBKDF2 + session-table
#    implementation (rate-limited, 24h sessions) instead of comparing
#    plaintext env vars and returning a hardcoded token.
# 2. A full set of analytics endpoints backed by analytics.py, matching
#    every section in the admin dashboard spec (overview, growth, feature
#    usage, resume/ATS, job search, quiz, performance, AI usage, errors,
#    funnel).

from fastapi import APIRouter, HTTPException, Header, Request, Depends
from pydantic import BaseModel, Field

import auth
import analytics
import settings_store

router = APIRouter()


# --- Request Models ---
class LoginRequest(BaseModel):
    username: str
    password: str


class SettingsPatch(BaseModel):
    bot_name: str | None = None
    welcome_message: str | None = None
    notify_new_user: bool | None = None
    notify_new_job: bool | None = None
    notify_system_errors: bool | None = None
    notify_weekly_reports: bool | None = None
    theme: str | None = None
    primary_color: str | None = None
    sidebar_style: str | None = None
    bot_active: bool | None = None
    bot_avatar: str | None = Field(default=None, max_length=8)


# --- Auth dependency ---
def require_admin(authorization: str = Header(None)):
    """Raises 401 unless the Authorization header carries a valid,
    non-expired session token minted by /api/auth/login."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not auth.validate_session(token):
        raise HTTPException(status_code=401, detail="Session invalid or expired")
    return token


# --- Authentication endpoints ---
@router.post("/api/auth/login")
async def login(credentials: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    if auth.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again in 15 minutes."
        )

    if not auth.authenticate_admin(credentials.username, credentials.password):
        auth.record_failed_attempt(client_ip)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    auth.clear_failed_attempts(client_ip)
    token = auth.create_session(credentials.username)
    return {"token": token, "username": credentials.username.lower(), "expires_in_hours": 24}


@router.post("/api/auth/logout")
async def logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        auth.terminate_session(authorization.split(" ", 1)[1].strip())
    return {"status": "Logged out successfully"}


@router.get("/api/auth/verify")
async def verify(_: str = Depends(require_admin)):
    return {"valid": True}


# --- Analytics endpoints (all require a valid session) ---
@router.get("/api/analytics/overview")
async def get_overview(date: str = None, _: str = Depends(require_admin)):
    return analytics.overview(date=date)


@router.get("/api/analytics/new-returning")
async def get_new_returning(date: str = None, _: str = Depends(require_admin)):
    return analytics.new_vs_returning(date=date)


@router.get("/api/analytics/growth")
async def get_growth(months: int = 6, _: str = Depends(require_admin)):
    return {"data": analytics.monthly_growth(months=months)}


@router.get("/api/analytics/daily")
async def get_daily(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return {"data": analytics.daily_activity(start=start, end=end)}


@router.get("/api/analytics/hourly")
async def get_hourly(date: str = None, start: str = None, end: str = None, _: str = Depends(require_admin)):
    return {"data": analytics.hourly_activity(date=date, start=start, end=end)}


@router.get("/api/analytics/features/detail")
async def get_features_detail(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.features_detail(start, end)


@router.get("/api/analytics/features")
async def get_features(date: str = None, start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.feature_usage(date=date, start=start, end=end)


@router.get("/api/analytics/resume")
async def get_resume(_: str = Depends(require_admin)):
    return analytics.resume_analytics()


@router.get("/api/analytics/jobs")
async def get_jobs(_: str = Depends(require_admin)):
    return analytics.job_search_analytics()


@router.get("/api/analytics/quiz")
async def get_quiz(_: str = Depends(require_admin)):
    return analytics.quiz_analytics()


@router.get("/api/analytics/performance")
async def get_performance(_: str = Depends(require_admin)):
    return analytics.performance()


@router.get("/api/analytics/ai")
async def get_ai_usage(_: str = Depends(require_admin)):
    return analytics.ai_usage()


@router.get("/api/analytics/errors")
async def get_errors(limit: int = 10, _: str = Depends(require_admin)):
    return analytics.recent_errors(limit=limit)


@router.get("/api/analytics/funnel")
async def get_funnel(_: str = Depends(require_admin)):
    return analytics.funnel()


@router.get("/api/analytics/trend")
async def get_trend(metric: str, start: str, end: str, _: str = Depends(require_admin)):
    """Powers every '+12.3% vs last period' delta chip. metric is one of:
    total_users, messages, job_searches, quizzes, resumes, feature_usage,
    avg_response_ms, success_rate."""
    try:
        return analytics.trend(metric, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/analytics/opportunities/page")
async def get_opportunities_page(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.opportunities_page(start, end)


@router.get("/api/analytics/opportunities")
async def get_opportunities(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.opportunities_activity(start=start, end=end)


@router.get("/api/analytics/config")
async def get_config(request: Request, _: str = Depends(require_admin)):
    return settings_store.config_info(str(request.base_url))


@router.get("/api/analytics/resume/page")
async def get_resume_page(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.resume_page(start, end)


@router.get("/api/analytics/resume/detail")
async def get_resume_detail(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.resume_detail(start=start, end=end)


@router.get("/api/analytics/quiz/page")
async def get_quiz_page(start: str = None, end: str = None, topic: str = None, _: str = Depends(require_admin)):
    return analytics.quiz_page(start, end, topic)


@router.get("/api/analytics/quiz/detail")
async def get_quiz_detail(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.quiz_detail(start=start, end=end)


# --- Users page ---
@router.get("/api/analytics/users/summary")
async def get_users_summary(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return analytics.users_summary(start=start, end=end)


@router.get("/api/analytics/users/timeseries")
async def get_users_timeseries(start: str = None, end: str = None, _: str = Depends(require_admin)):
    return {"data": analytics.users_timeseries(start=start, end=end)}


@router.get("/api/analytics/users/source")
async def get_users_source(_: str = Depends(require_admin)):
    return analytics.users_source()


@router.get("/api/analytics/users/activity-stats")
async def get_users_activity_stats(_: str = Depends(require_admin)):
    return analytics.users_activity_stats()


@router.get("/api/analytics/users/engagement")
async def get_users_engagement(limit: int = 6, _: str = Depends(require_admin)):
    return {"data": analytics.user_engagement(limit=limit)}


@router.get("/api/analytics/users/commands")
async def get_users_commands(limit: int = 5, _: str = Depends(require_admin)):
    return {"data": analytics.popular_commands_by_users(limit=limit)}


@router.get("/api/analytics/users/list")
async def get_users_list(limit: int = 20, offset: int = 0, search: str = None, _: str = Depends(require_admin)):
    return analytics.list_users(limit=limit, offset=offset, search=search)


@router.get("/api/analytics/users/{phone}")
async def get_user_detail(phone: str, days: int = 30, _: str = Depends(require_admin)):
    detail = analytics.user_detail(phone, days)
    if not detail:
        raise HTTPException(status_code=404, detail="User not found")
    return detail


# --- System health ---
@router.get("/api/analytics/system")
async def get_system(logs: int = 5, _: str = Depends(require_admin)):
    return analytics.system_metrics(log_limit=logs)


# --- Admin panel settings (display preferences only — never real secrets) ---
@router.get("/api/analytics/settings")
async def get_settings(_: str = Depends(require_admin)):
    return settings_store.get_settings()


@router.post("/api/analytics/settings")
async def post_settings(patch: SettingsPatch, _: str = Depends(require_admin)):
    return settings_store.save_settings(patch.dict(exclude_unset=True))


# --- Raw log viewer (kept from the original version, now auth-protected) ---
@router.get("/api/analytics/logs")
async def get_analytics_logs(limit: int = 200, _: str = Depends(require_admin)):
    """Returns the raw user_logs rows (one row per command sent)."""
    try:
        with analytics._conn() as conn:
            rows = conn.raw(
                """SELECT user_phone AS whatsapp_no, command,
                          to_char(timestamp AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') AS timestamp
                   FROM analytics_events WHERE event_type = 'command' AND user_phone IS NOT NULL
                   ORDER BY timestamp DESC, id DESC LIMIT %s""", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
