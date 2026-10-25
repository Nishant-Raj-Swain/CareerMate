# learning_analytics.py  -  data for the "Learning Analytics" pages of the admin panel
# ---------------------------------------------------------------------------
#   notes_page(start, end)   ->  YouTube Notes Analytics   (/utube_notes events)
#   chat_pdf_page(start, end) ->  Chat with PDF Analytics   (/chat_pdf events)
#   pyq_page(start, end)       ->  PYQ Analytics            (/pyq events, event_type 'pyq')
# (The Books page is added here next.)
#
# Source: the analytics_events table. /utube_notes logs one event per request:
#       event_type 'ai_assistance', event_details {"feature": "notes_utube", "video": id, "title": ..., "duration": sec, "ms": ...}
#       success TRUE / FALSE.   Older rows that lack title / duration are still counted (shown as "YouTube video").
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

# subject of a video, guessed from its title / channel (first rule that matches wins)
SUBJECT_RULES = [
    ("Machine Learning", r"machine learning|deep learning|neural|\bai\b|artificial intelligence|\bllm|tensorflow|pytorch|nlp|gen ?ai"),
    ("Data Science", r"data science|data analy|pandas|numpy|power ?bi|tableau|statistics|data visual|big data"),
    ("Web Development", r"web dev|html|css|javascript|react|node\.?js|frontend|front-end|backend|full ?stack|django|flask|angular|next\.?js|php|bootstrap"),
    ("Python", r"python"),
    ("Java", r"\bjava\b|spring boot|jdbc|servlet"),
    ("C / C++", r"\bc\+\+|\bc programming|\bc language|\bcpp\b"),
    ("DSA", r"data structure|algorithm|\bdsa\b|leetcode|sorting|dynamic programming"),
    ("Database / SQL", r"\bsql\b|mysql|dbms|database|mongodb|postgres"),
    ("Aptitude / Placement", r"aptitude|placement|interview|reasoning|resume"),
    ("Cloud / DevOps", r"aws|azure|docker|kubernetes|devops|cloud|linux|git\b|github"),
]
LENGTH_BUCKETS = [("<10 min", 0, 600), ("10-20 min", 600, 1200), ("20-30 min", 1200, 1800), ("30-60 min", 1800, 3600),
                  ("1-2 hr", 3600, 7200), ("2-5 hr", 7200, 18000), (">5 hr", 18000, 10 ** 9)]


def subject_of(title: str, channel: str = "") -> str:
    text = f"{title} {channel}".lower()
    for name, pat in SUBJECT_RULES:
        if re.search(pat, text):
            return name
    return "Others"


def _conn():
    import analytics
    return analytics._conn()


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


def _delta(cur, prev):
    if not prev:
        return None if not cur else 100.0
    return round(100.0 * (cur - prev) / prev, 1)


def _rows(conn, start: str, end: str) -> list[dict]:
    """All /utube_notes events with date in [start, end] (UTC dates), oldest first, details parsed."""
    cur = conn.raw("""SELECT user_phone, success, response_time, event_details,
                             to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                        FROM analytics_events
                       WHERE event_type = 'ai_assistance' AND event_details LIKE %s
                         AND to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s
                       ORDER BY timestamp, id""", ('%"notes_utube"%', start, end))
    out = []
    for r in cur.fetchall():
        try:
            d = json.loads(r["event_details"] or "{}")
        except ValueError:
            d = {}
        if d.get("feature") != "notes_utube" or not r["user_phone"]:
            continue
        ms = d.get("ms") if isinstance(d.get("ms"), (int, float)) else None
        out.append({"phone": r["user_phone"], "ok": bool(r["success"]), "ts": r["ts"], "video": d.get("video") or "",
                    "title": d.get("title") or "", "channel": d.get("channel") or "", "duration": int(d.get("duration") or 0),
                    "secs": round(ms / 1000.0, 1) if ms is not None else None, "message": d.get("message") or ""})
    return out


def _summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    times = [r["secs"] for r in ok if r["secs"] is not None]
    return {"total": len(rows), "users": len({r["phone"] for r in rows}), "success": len(ok), "failed": len(rows) - len(ok),
            "avg_secs": round(sum(times) / len(times), 1) if times else 0.0}


def notes_page(start: str | None = None, end: str | None = None) -> dict:
    """Everything the YouTube Notes Analytics page shows for [start, end] (inclusive 'YYYY-MM-DD'), vs the previous period of equal length."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    start = start or (e - timedelta(days=29)).strftime("%Y-%m-%d")
    s = datetime.strptime(start, "%Y-%m-%d")
    n_days = (e - s).days + 1
    p_end, p_start = s - timedelta(days=1), s - timedelta(days=n_days)
    with _conn() as conn:
        rows = _rows(conn, start, end)
        prev = _rows(conn, p_start.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d"))
        names = {r["user_phone"]: f"User {i}" for i, r in enumerate(
            conn.raw("SELECT user_phone FROM analytics_events WHERE user_phone IS NOT NULL GROUP BY user_phone ORDER BY MIN(timestamp), user_phone").fetchall(), 1)}
    cur_s, prev_s = _summary(rows), _summary(prev)
    ok_rate = _pct(cur_s["success"], cur_s["total"])
    prev_rate = _pct(prev_s["success"], prev_s["total"])
    for r in rows:
        r["subject"] = subject_of(r["title"], r["channel"])

    days, d = [], s
    while d <= e and len(days) < 400:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    per = {k: {x: 0 for x in days} for k in ("total", "ok", "failed")}
    for r in rows:
        day = r["ts"][:10]
        per["total"][day] += 1
        per["ok" if r["ok"] else "failed"][day] += 1

    subj: dict[str, int] = {}
    for r in rows:
        subj[r["subject"]] = subj.get(r["subject"], 0) + 1
    top = sorted(((k, v) for k, v in subj.items() if k != "Others"), key=lambda x: -x[1])
    shown = top[:4]
    others = sum(v for k, v in subj.items()) - sum(v for _, v in shown)
    subjects = [{"name": k, "count": v, "pct": _pct(v, len(rows))} for k, v in shown]
    if others:
        subjects.append({"name": "Others", "count": others, "pct": _pct(others, len(rows))})

    lengths = []
    for label, lo, hi in LENGTH_BUCKETS:
        c = sum(1 for r in rows if r["duration"] and lo <= r["duration"] < hi)
        lengths.append({"label": label, "count": c})
    with_len = sum(x["count"] for x in lengths)
    for x in lengths:
        x["pct"] = _pct(x["count"], with_len)
    best_len = max(lengths, key=lambda x: x["count"]) if with_len else None

    ok_times = [r["secs"] for r in rows if r["ok"] and r["secs"] is not None]
    perf = {"success_rate": ok_rate, "failure_rate": round(100 - ok_rate, 1) if rows else 0.0, "avg_secs": cur_s["avg_secs"],
            "max_secs": round(max(ok_times), 1) if ok_times else 0.0}

    by_video: dict[str, dict] = {}
    for r in rows:
        key = r["video"] or r["title"] or "unknown"
        v = by_video.setdefault(key, {"video": r["video"], "title": r["title"] or "YouTube video", "subject": r["subject"], "requests": 0, "users": set()})
        v["requests"] += 1; v["users"].add(r["phone"])
        if r["title"]:
            v["title"], v["subject"] = r["title"], r["subject"]
    vids = sorted(by_video.values(), key=lambda v: (-v["requests"], v["title"]))
    top_videos = [{"video": v["video"], "title": v["title"], "subject": v["subject"], "requests": v["requests"]} for v in vids[:5]]
    repeated = [{"video": v["video"], "title": v["title"], "users": len(v["users"]), "requests": v["requests"], "pct": _pct(v["requests"], len(rows))}
                for v in vids if v["requests"] > 1][:5]

    recent = [{"user": names.get(r["phone"], "User"), "phone": r["phone"], "video": r["video"], "title": r["title"] or "YouTube video",
               "subject": r["subject"], "ok": r["ok"], "time": r["ts"], "error": r["message"] if not r["ok"] else ""} for r in reversed(rows[-100:])]

    hours = [0] * 24
    for r in rows:
        hours[int(r["ts"][11:13])] += 1
    peak_hour = hours.index(max(hours)) if rows else None

    insights = []
    if best_len:
        insights.append(f"{best_len['label']} videos receive the highest requests ({fmtp(best_len['pct'])} of requests with a known length).")
    if len(subjects) >= 2 and subjects[0]["name"] != "Others":
        insights.append(f"{subjects[0]['name']} and {subjects[1]['name']} are the top subjects ({fmtp(subjects[0]['pct'] + subjects[1]['pct'])} combined).")
    elif subjects:
        insights.append(f"{subjects[0]['name']} is the top subject ({fmtp(subjects[0]['pct'])}).")
    if rows and prev_s["total"]:
        diff = round(ok_rate - prev_rate, 1)
        insights.append(f"Success rate is {fmtp(ok_rate)}, which is {abs(diff)}% {'higher' if diff >= 0 else 'lower'} than the previous {n_days} days.")
        fd = _delta(cur_s["failed"], prev_s["failed"])
        if fd is not None:
            insights.append(f"Failed requests have {'decreased' if fd <= 0 else 'increased'} by {abs(fd)}% compared to the previous {n_days} days.")
    elif rows:
        insights.append(f"Success rate is {fmtp(ok_rate)} over the last {n_days} days.")

    return {"start": start, "end": end, "days": n_days,
            "kpis": {"total": {"value": cur_s["total"], "delta": _delta(cur_s["total"], prev_s["total"])},
                     "users": {"value": cur_s["users"], "delta": _delta(cur_s["users"], prev_s["users"])},
                     "success": {"value": cur_s["success"], "delta": _delta(cur_s["success"], prev_s["success"])},
                     "failed": {"value": cur_s["failed"], "delta": _delta(cur_s["failed"], prev_s["failed"])},
                     "avg_secs": {"value": cur_s["avg_secs"], "delta": _delta(cur_s["avg_secs"], prev_s["avg_secs"]) if prev_s["avg_secs"] else None}},
            "trend": {"labels": days, "total": [per["total"][x] for x in days], "success": [per["ok"][x] for x in days], "failed": [per["failed"][x] for x in days]},
            "subjects": subjects, "lengths": lengths, "best_length": best_len["label"] if best_len else None,
            "best_length_pct": best_len["pct"] if best_len else 0, "performance": perf,
            "top_videos": top_videos, "repeated": repeated, "recent": recent, "peak_hour_utc": peak_hour, "insights": insights}


def fmtp(x: float) -> str:
    return f"{x:.1f}%"


# =============================================================================== Chat with PDF
DOC_TYPES = [
    ("Research Papers", r"abstract|et al\.?|\bdoi\b|arxiv|journal|proceedings|\breferences\b|keywords?:|ieee|springer|elsevier|paper"),
    ("Notes / Handouts", r"notes|handout|lecture|\bunit[ -]?\d|\bmodule|slides|worksheet|assignment"),
    ("Guides / Manuals", r"guide|manual|handbook|tutorial|cheat ?sheet|documentation|how to|reference card"),
    ("PDF (Study Material)", r"textbook|syllabus|question paper|\bpyq\b|exam|study|course|\bbook\b|edition|chapter|semester|previous year|curriculum"),
]
DOC_SUBJECTS = [
    ("Data Science", r"data science|machine learning|analytics|pandas|statistic|neural|deep learning|\bai\b|artificial intelligence|big data"),
    ("Computer Science", r"programming|algorithm|python|java\b|c\+\+|software|computer|database|dbms|operating system|network|html|web dev|compiler|data structure|\bdsa\b|\bsql\b"),
    ("Engineering", r"engineering|mechanical|electrical|civil|electronics|thermodynamics|circuit|structural|physics|mathematics|calculus"),
    ("Business & Management", r"management|marketing|finance|accounting|business|economics|entrepreneur|\bhr\b|supply chain"),
]


def classify_doc(filename: str, text: str = "") -> tuple[str, str]:
    """(document type, subject) of an uploaded PDF from its file name + the start of its text."""
    name, body = re.sub(r"[_\-.]+", " ", (filename or "").lower()), (text or "").lower()[:3000]
    dtype = "Others"
    for label, pat in DOC_TYPES:
        hits = len(re.findall(pat, name)) * 2 + len(re.findall(pat, body))
        if hits >= (3 if label == "Research Papers" else 1) or (label == "Research Papers" and re.search(r"abstract", body) and re.search(r"references|et al", body)):
            dtype = label
            break
    subject = "Others"
    for label, pat in DOC_SUBJECTS:
        if re.search(pat, name) or len(re.findall(pat, body)) >= 2:
            subject = label
            break
    return dtype, subject


def _events(conn, feature: str, start: str, end: str) -> list[dict]:
    cur = conn.raw("""SELECT user_phone, success, event_details,
                             to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                        FROM analytics_events
                       WHERE event_type = 'ai_assistance' AND event_details LIKE %s
                         AND to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s
                       ORDER BY timestamp, id""", (f'%"{feature}"%', start, end))
    out = []
    for r in cur.fetchall():
        try:
            d = json.loads(r["event_details"] or "{}")
        except ValueError:
            d = {}
        if d.get("feature") == feature and r["user_phone"]:
            out.append({"phone": r["user_phone"], "ok": bool(r["success"]), "ts": r["ts"], "d": d})
    return out


def _pdf_sessions(events: list[dict]) -> list[dict]:
    """Groups chat_pdf events into sessions: one per ingest attempt; answers/summaries/exit attach to the user's latest ingest."""
    sessions, cur = [], {}
    for e in events:
        st, p, d = e["d"].get("stage"), e["phone"], e["d"]
        if st == "ingest":
            s = {"phone": p, "ts": e["ts"], "file": d.get("filename") or "", "ingest_ok": e["ok"], "doc_type": d.get("doc_type") or "Others",
                 "subject": d.get("subject") or "Others", "pages": d.get("pages") or 0, "q_ok": 0, "q_fail": 0, "summaries": 0, "exited": False,
                 "times": [], "error": d.get("message") or ""}
            sessions.append(s); cur[p] = s
        elif p in cur and st in ("answer", "summary"):
            s = cur[p]
            if e["ok"]:
                s["q_ok"] += 1
                if st == "summary":
                    s["summaries"] += 1
                if isinstance(d.get("ms"), (int, float)):
                    s["times"].append(d["ms"] / 1000.0)
            else:
                s["q_fail"] += 1
        elif p in cur and st == "exit":
            cur[p]["exited"] = True
    for s in sessions:
        if not s["ingest_ok"] or (s["q_ok"] == 0 and s["q_fail"] > 0):
            s["status"] = "failed"
        elif s["q_ok"] == 0:
            s["status"] = "interrupted"
        else:
            s["status"] = "successful"
    return sessions


def _pdf_summary(sessions: list[dict]) -> dict:
    n = len(sessions)
    times = [t for s in sessions for t in s["times"]]
    return {"sessions": n, "users": len({s["phone"] for s in sessions}), "questions": sum(s["q_ok"] for s in sessions),
            "success": sum(1 for s in sessions if s["status"] == "successful"),
            "rate": _pct(sum(1 for s in sessions if s["status"] == "successful"), n),
            "avg_secs": round(sum(times) / len(times), 1) if times else 0.0}


def chat_pdf_page(start: str | None = None, end: str | None = None) -> dict:
    """Everything the Chat with PDF Analytics page shows for [start, end], vs the previous period of equal length."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    start = start or (e - timedelta(days=29)).strftime("%Y-%m-%d")
    s0 = datetime.strptime(start, "%Y-%m-%d")
    n_days = (e - s0).days + 1
    p_end, p_start = s0 - timedelta(days=1), s0 - timedelta(days=n_days)
    with _conn() as conn:
        # sessions need their answers even when the ingest was just before the range start -> look back one extra day
        look = (s0 - timedelta(days=1)).strftime("%Y-%m-%d")
        ev = _events(conn, "chat_pdf", look, end)
        prev_ev = _events(conn, "chat_pdf", (p_start - timedelta(days=1)).strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d"))
        names = {r["user_phone"]: f"User {i}" for i, r in enumerate(
            conn.raw("SELECT user_phone FROM analytics_events WHERE user_phone IS NOT NULL GROUP BY user_phone ORDER BY MIN(timestamp), user_phone").fetchall(), 1)}
    sessions = [x for x in _pdf_sessions(ev) if start <= x["ts"][:10] <= end]
    prev = [x for x in _pdf_sessions(prev_ev) if p_start.strftime("%Y-%m-%d") <= x["ts"][:10] <= p_end.strftime("%Y-%m-%d")]
    cur_s, prev_s = _pdf_summary(sessions), _pdf_summary(prev)
    n = len(sessions)

    days, d = [], s0
    while d <= e and len(days) < 400:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    per = {k: {x: 0 for x in days} for k in ("sessions", "questions", "failed")}
    for x in sessions:
        per["sessions"][x["ts"][:10]] += 1
        if x["status"] == "failed":
            per["failed"][x["ts"][:10]] += 1
    for x in ev:                                              # questions on the day they were asked
        if x["d"].get("stage") in ("answer", "summary") and x["ok"] and start <= x["ts"][:10] <= end:
            per["questions"][x["ts"][:10]] += 1

    status = [{"name": k.capitalize(), "key": k, "count": sum(1 for x in sessions if x["status"] == k), "pct": 0.0} for k in ("successful", "failed", "interrupted")]
    for st in status:
        st["pct"] = _pct(st["count"], n)

    qbuckets = [("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-10", 6, 10), (">10", 11, 10 ** 6)]
    qps = [{"label": lbl, "count": sum(1 for x in sessions if lo <= x["q_ok"] <= hi)} for lbl, lo, hi in qbuckets]

    def breakdown(key, order):
        cnt: dict[str, int] = {}
        for x in sessions:
            cnt[x[key]] = cnt.get(x[key], 0) + 1
        ordered = [(k, cnt[k]) for k in order if k in cnt] + [(k, v) for k, v in cnt.items() if k not in order]
        return [{"name": k, "count": v, "pct": _pct(v, n)} for k, v in sorted(ordered, key=lambda kv: (kv[0] == "Others", -kv[1]))]
    doc_types = breakdown("doc_type", [])
    subj_all = breakdown("subject", [])
    subjects = subj_all                                       # at most 5 labels (4 subjects + Others)

    by_file: dict[str, dict] = {}
    for x in sessions:
        if x["file"]:
            f = by_file.setdefault(x["file"].lower(), {"name": x["file"], "sessions": 0})
            f["sessions"] += 1
    popular = sorted(by_file.values(), key=lambda f: (-f["sessions"], f["name"]))[:5]

    recent = [{"user": names.get(x["phone"], "User"), "phone": x["phone"], "file": x["file"] or "(unnamed PDF)", "questions": x["q_ok"],
               "status": x["status"], "time": x["ts"], "error": x["error"]} for x in reversed(sessions[-100:])]

    exited = sum(1 for x in sessions if x["exited"] and x["ingest_ok"])
    summaries = sum(x["summaries"] for x in sessions)
    summary_sessions = sum(1 for x in sessions if x["summaries"])
    actions = [{"key": "upload", "label": "Uploaded PDF", "count": n, "pct": 100.0 if n else 0.0},
               {"key": "ask", "label": "Asked Questions", "count": cur_s["questions"], "pct": _pct(cur_s["questions"], n)},
               {"key": "summary", "label": "Viewed Summary", "count": summaries, "pct": _pct(summary_sessions, n)},
               {"key": "done", "label": "Session Completed", "count": exited, "pct": _pct(exited, n)}]

    return {"start": start, "end": end, "days": n_days,
            "kpis": {"sessions": {"value": n, "delta": _delta(n, prev_s["sessions"])},
                     "users": {"value": cur_s["users"], "delta": _delta(cur_s["users"], prev_s["users"])},
                     "questions": {"value": cur_s["questions"], "delta": _delta(cur_s["questions"], prev_s["questions"])},
                     "rate": {"value": cur_s["rate"], "delta": round(cur_s["rate"] - prev_s["rate"], 1) if prev_s["sessions"] and n else None}},
            "trend": {"labels": days, "sessions": [per["sessions"][x] for x in days], "questions": [per["questions"][x] for x in days], "failed": [per["failed"][x] for x in days]},
            "status": status, "questions_per_session": qps,
            "avg_response": {"value": cur_s["avg_secs"], "delta": _delta(cur_s["avg_secs"], prev_s["avg_secs"]) if prev_s["avg_secs"] else None},
            "doc_types": doc_types, "subjects": subjects, "popular": popular, "recent": recent, "actions": actions,
            "note": "Total questions can be more than total sessions as users may ask multiple questions in a single session."}


# =============================================================================== PYQ
# /pyq logs event_type 'pyq' with event_details:
#   {"stage": "start",   "section": "university" | "company" | "interview"}                      (user chose a section)
#   {"stage": "deliver", "section": "university", "uni":..., "course":..., "branch":..., "count": n}      success = files sent
#   {"stage": "deliver", "section": "company",    "company": "TCS", "count": n}
#   {"stage": "deliver", "section": "interview",  "kind": "hr" | "gd" | "tech", "topic": "...", "count": n}
# Old rows written by the first version (no "stage", but with "uni") are read as university deliveries.
PYQ_KINDS = {"hr": "HR Interview", "gd": "Group Discussion (GD)", "tech": "Technical Interview"}
SESSION_GAP = timedelta(hours=6)


def _pyq_events(conn, start: str, end: str) -> list[dict]:
    cur = conn.raw("""SELECT user_phone, success, event_details,
                             to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                        FROM analytics_events
                       WHERE event_type = 'pyq' AND user_phone IS NOT NULL
                         AND to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s
                       ORDER BY timestamp, id""", (start, end))
    out = []
    for r in cur.fetchall():
        try:
            d = json.loads(r["event_details"] or "{}")
        except ValueError:
            d = {}
        stage = d.get("stage") or ("deliver" if d.get("uni") is not None else None)
        section = d.get("section") or ("university" if d.get("uni") is not None else None)
        if stage in ("start", "deliver") and section in ("university", "company", "interview"):
            out.append({"phone": r["user_phone"], "ok": bool(r["success"]), "ts": r["ts"], "stage": stage, "section": section, "d": d})
    return out


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _pyq_label(section: str, d: dict) -> str:
    if section == "university":
        parts = [d.get("uni"), d.get("course"), d.get("branch")]
        extra = d.get("filter")
        label = " \u2192 ".join(str(p) for p in parts if p)
        return label + (f" \u2192 {extra}" if extra and isinstance(extra, (str, int)) else "")
    if section == "company":
        return str(d.get("company") or "")
    kind = PYQ_KINDS.get(d.get("kind"), "Interview")
    return f"{kind} \u2192 {d['topic']}" if d.get("topic") else kind


def _pyq_sessions(events: list[dict]) -> list[dict]:
    """A session = the user choosing a section, plus what was delivered after it. A delivery without a start event makes its own session."""
    sessions, cur = [], {}
    for e in events:
        p, d = e["phone"], e["d"]
        if e["stage"] == "start":
            s = {"phone": p, "ts": e["ts"], "section": e["section"], "deliveries": [], "label": ""}
            sessions.append(s); cur[p] = s
            continue
        s = cur.get(p)
        if not s or s["section"] != e["section"] or _parse_ts(e["ts"]) - _parse_ts(s["ts"]) > SESSION_GAP:
            s = {"phone": p, "ts": e["ts"], "section": e["section"], "deliveries": [], "label": ""}
            sessions.append(s); cur[p] = s
        n = d.get("count")
        s["deliveries"].append({"ok": e["ok"] and (n is None or n > 0), "count": int(n) if isinstance(n, (int, float)) else 1, "d": d, "ts": e["ts"]})
        s["label"] = _pyq_label(e["section"], d) or s["label"]
    for s in sessions:
        ok = [x for x in s["deliveries"] if x["ok"]]
        s["status"] = "completed" if ok else ("failed" if s["deliveries"] else "incomplete")
        s["papers"] = sum(x["count"] for x in ok)
    return sessions


def _pyq_summary(sessions: list[dict]) -> dict:
    n = len(sessions)
    dels = [x for s in sessions for x in s["deliveries"]]
    done = sum(1 for s in sessions if s["status"] == "completed")
    return {"sessions": n, "users": len({s["phone"] for s in sessions}), "papers": sum(s["papers"] for s in sessions),
            "completion": _pct(done, n), "success": _pct(sum(1 for x in dels if x["ok"]), len(dels)), "attempts": len(dels)}


def pyq_page(start: str | None = None, end: str | None = None) -> dict:
    """Everything the PYQ Analytics page shows for [start, end], vs the previous period of equal length."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    start = start or (e - timedelta(days=29)).strftime("%Y-%m-%d")
    s0 = datetime.strptime(start, "%Y-%m-%d")
    n_days = (e - s0).days + 1
    p_end, p_start = s0 - timedelta(days=1), s0 - timedelta(days=n_days)
    with _conn() as conn:
        ev = _pyq_events(conn, (s0 - timedelta(days=1)).strftime("%Y-%m-%d"), end)
        prev_ev = _pyq_events(conn, (p_start - timedelta(days=1)).strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d"))
        names = {r["user_phone"]: f"User {i}" for i, r in enumerate(
            conn.raw("SELECT user_phone FROM analytics_events WHERE user_phone IS NOT NULL GROUP BY user_phone ORDER BY MIN(timestamp), user_phone").fetchall(), 1)}
    sessions = [x for x in _pyq_sessions(ev) if start <= x["ts"][:10] <= end]
    prev = [x for x in _pyq_sessions(prev_ev) if p_start.strftime("%Y-%m-%d") <= x["ts"][:10] <= p_end.strftime("%Y-%m-%d")]
    cur_s, prev_s = _pyq_summary(sessions), _pyq_summary(prev)
    n = len(sessions)

    days, d = [], s0
    while d <= e and len(days) < 400:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    per = {k: {x: 0 for x in days} for k in ("sessions", "papers", "done")}
    for x in sessions:
        per["sessions"][x["ts"][:10]] += 1
        per["papers"][x["ts"][:10]] += x["papers"]
        per["done"][x["ts"][:10]] += 1 if x["status"] == "completed" else 0
    completion = [round(100.0 * per["done"][x] / per["sessions"][x], 1) if per["sessions"][x] else None for x in days]

    def tally(items):
        c: dict[str, dict] = {}
        for key, papers in items:
            if not key:
                continue
            row = c.setdefault(key.strip().lower(), {"name": key.strip(), "sessions": 0, "papers": 0})
            row["sessions"] += 1; row["papers"] += papers
        return sorted(c.values(), key=lambda r: (-r["sessions"], r["name"]))

    def first(s, field):
        for x in s["deliveries"]:
            if x["ok"] and x["d"].get(field):
                return str(x["d"][field])
        return ""
    uni_rows = tally((first(s, "uni"), s["papers"]) for s in sessions if s["section"] == "university")
    uni_total = sum(r["sessions"] for r in uni_rows)
    universities = [{**r, "pct": _pct(r["sessions"], uni_total)} for r in uni_rows[:10]]

    comp_rows = tally((first(s, "company"), s["papers"]) for s in sessions if s["section"] == "company")
    comp_total = sum(r["sessions"] for r in comp_rows)
    companies = [{"name": r["name"], "queries": r["sessions"], "papers": r["papers"], "pct": _pct(r["sessions"], comp_total)} for r in comp_rows[:6]]
    rest = sum(r["sessions"] for r in comp_rows[6:])
    if rest:
        companies.append({"name": "Other", "queries": rest, "papers": sum(r["papers"] for r in comp_rows[6:]), "pct": _pct(rest, comp_total)})

    kinds = {k: 0 for k in PYQ_KINDS}
    tech_topics: dict[str, int] = {}
    for s in sessions:
        if s["section"] != "interview":
            continue
        for x in s["deliveries"]:
            if not x["ok"]:
                continue
            k = x["d"].get("kind")
            if k in kinds:
                kinds[k] += 1
                if k == "tech" and x["d"].get("topic"):
                    t = str(x["d"]["topic"]).strip()
                    tech_topics[t] = tech_topics.get(t, 0) + 1
    int_total = sum(kinds.values())
    interview = [{"key": k, "name": PYQ_KINDS[k], "count": v, "pct": _pct(v, int_total)} for k, v in kinds.items()]
    tech_total = sum(tech_topics.values())
    tt = sorted(tech_topics.items(), key=lambda kv: (-kv[1], kv[0]))
    topics = [{"name": k, "count": v, "pct": _pct(v, tech_total)} for k, v in tt[:5]]
    if len(tt) > 5:
        o = sum(v for _, v in tt[5:])
        topics.append({"name": "Others", "count": o, "pct": _pct(o, tech_total)})

    sec_label = {"university": "University", "company": "Company", "interview": "Interview"}
    recent = [{"user": names.get(x["phone"], "User"), "phone": x["phone"], "section": sec_label[x["section"]], "label": x["label"] or "\u2014",
               "time": x["ts"], "status": x["status"]} for x in reversed(sessions[-100:])]
    by_section = {k: sum(1 for x in sessions if x["section"] == k) for k in sec_label}

    return {"start": start, "end": end, "days": n_days,
            "kpis": {"sessions": {"value": n, "delta": _delta(n, prev_s["sessions"])},
                     "papers": {"value": cur_s["papers"], "delta": _delta(cur_s["papers"], prev_s["papers"])},
                     "completion": {"value": cur_s["completion"], "delta": round(cur_s["completion"] - prev_s["completion"], 1) if prev_s["sessions"] and n else None},
                     "success": {"value": cur_s["success"], "delta": round(cur_s["success"] - prev_s["success"], 1) if prev_s["attempts"] and cur_s["attempts"] else None}},
            "trend": {"labels": days, "sessions": [per["sessions"][x] for x in days], "papers": [per["papers"][x] for x in days], "completion": completion},
            "universities": universities, "university_total": uni_total, "companies": companies, "company_total": comp_total,
            "interview": interview, "interview_total": int_total, "tech_topics": topics, "recent": recent, "by_section": by_section}


# ======================================================================= Books Analytics (/books)
BOOK_METHODS = {"topic": "Topic-wise", "name": "Book name / Author"}


def _book_events(conn, start: str, end: str) -> list[dict]:
    cur = conn.raw("""SELECT user_phone, success, event_details,
                             to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                        FROM analytics_events
                       WHERE event_type = 'books' AND user_phone IS NOT NULL
                         AND to_char(timestamp AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s
                       ORDER BY timestamp, id""", (start, end))
    out = []
    for r in cur.fetchall():
        try:
            d = json.loads(r["event_details"] or "{}")
        except ValueError:
            d = {}
        out.append({"phone": r["user_phone"], "ok": bool(r["success"]), "ts": r["ts"], "d": d})
    return out


def _book_sessions(events: list[dict]) -> list[dict]:
    """A session starts at a 'start' event (or after SESSION_GAP of silence); old rows without 'stage' become one session each."""
    sessions, cur = [], {}

    def new(e, method=None):
        s = {"phone": e["phone"], "ts": e["ts"], "last": _parse_ts(e["ts"]), "method": method, "chosen": method is not None, "selected": False, "results": False, "found": 0,
             "sent": 0, "alts": False, "none": False, "category": "", "subtopic": "", "title": "", "query": "", "books": []}
        sessions.append(s); cur[e["phone"]] = s
        return s
    for e in events:
        d, p = e["d"], e["phone"]
        stage = d.get("stage")
        if not stage:                                                           # legacy rows (before stages existed)
            s = new(e, "name" if d.get("query") or d.get("title") and not d.get("subtopic") else "topic")
            s.update(selected=True, query=d.get("query", ""), title=d.get("title", ""), category=d.get("topic", ""), subtopic=d.get("subtopic", ""))
            if e["ok"]:
                s.update(results=True, found=1, sent=1)
            elif d.get("alternatives"):
                s.update(results=True, found=d["alternatives"], alts=True)
            else:
                s["none"] = True
            continue
        s = cur.get(p)
        if stage == "start" or s is None or (_parse_ts(e["ts"]) - s["last"]) > SESSION_GAP:
            s = new(e)
            if stage == "start":
                continue
        s["last"] = _parse_ts(e["ts"])
        m = d.get("method")
        if m in BOOK_METHODS and not s["method"]:
            s["method"] = m
        if stage == "method":
            s["method"] = m if m in BOOK_METHODS else s["method"]; s["chosen"] = True
        elif stage in ("select", "query"):
            s["selected"] = True
            s["category"] = d.get("category") or s["category"]; s["subtopic"] = d.get("subtopic") or s["subtopic"]; s["query"] = d.get("query") or s["query"]
        elif stage == "results":
            f = int(d.get("found") or 0)
            s["selected"] = True
            if f > 0:
                s["results"] = True; s["found"] = max(s["found"], f)
            s["category"] = d.get("category") or s["category"]; s["query"] = d.get("query") or s["query"]
        elif stage == "alts":
            s["alts"] = True; s["results"] = True
            s["category"] = d.get("category") or s["category"]; s["query"] = d.get("query") or s["query"]
        elif stage == "deliver":
            s["sent"] += int(d.get("sent") or 1); s["results"] = True; s["selected"] = True
            s["title"] = d.get("title") or s["title"]; s["category"] = d.get("category") or s["category"]
        elif stage == "none":
            s["none"] = True
        if stage == "book":
            s["books"].append({"title": str(d.get("title") or "").strip(), "located": bool(d.get("located")), "sent": e["ok"]})
    for s in sessions:
        s["status"] = "sent" if s["sent"] else "alts" if s["alts"] else "none" if s["none"] else "incomplete"
        if not s["method"]:
            s["method"] = "name" if s["query"] else "topic"
    return sessions


def _book_label(s: dict) -> str:
    if s["method"] == "name":
        return s["title"] or s["query"] or "—"
    return " → ".join(x for x in (s["category"], s["subtopic"] if s["subtopic"] != s["category"] else "") if x) or "—"


def books_page(start: str | None = None, end: str | None = None) -> dict:
    """Everything the Books Analytics page shows for [start, end], vs the previous period of equal length."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    start = start or (e - timedelta(days=29)).strftime("%Y-%m-%d")
    s0 = datetime.strptime(start, "%Y-%m-%d")
    n_days = (e - s0).days + 1
    p_end, p_start = s0 - timedelta(days=1), s0 - timedelta(days=n_days)
    with _conn() as conn:
        ev = _book_events(conn, (s0 - timedelta(days=1)).strftime("%Y-%m-%d"), end)
        prev_ev = _book_events(conn, (p_start - timedelta(days=1)).strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d"))
        names = {r["user_phone"]: f"User {i}" for i, r in enumerate(
            conn.raw("SELECT user_phone FROM analytics_events WHERE user_phone IS NOT NULL GROUP BY user_phone ORDER BY MIN(timestamp), user_phone").fetchall(), 1)}
    sessions = [x for x in _book_sessions(ev) if start <= x["ts"][:10] <= end]
    prev = [x for x in _book_sessions(prev_ev) if p_start.strftime("%Y-%m-%d") <= x["ts"][:10] <= p_end.strftime("%Y-%m-%d")]
    n = len(sessions)
    users = len({x["phone"] for x in sessions}); pusers = len({x["phone"] for x in prev})
    found = sum(x["found"] for x in sessions); pfound = sum(x["found"] for x in prev)
    failed = sum(1 for x in sessions if x["status"] == "none"); pfailed = sum(1 for x in prev if x["status"] == "none")
    sent_s = sum(1 for x in sessions if x["sent"])

    days, d = [], s0
    while d <= e and len(days) < 400:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    per = {m: {x: 0 for x in days} for m in BOOK_METHODS}
    for x in sessions:
        per[x["method"]][x["ts"][:10]] += 1

    def ranked(key_fn, limit):
        c: dict[str, dict] = {}
        for x in sessions:
            k = (key_fn(x) or "").strip()
            if k:
                c.setdefault(k.lower(), {"name": k, "count": 0})["count"] += 1
        return sorted(c.values(), key=lambda r: (-r["count"], r["name"])), limit
    cats, _ = ranked(lambda x: x["category"], 5)
    cat_total = sum(r["count"] for r in cats)
    categories = [{**r, "pct": _pct(r["count"], cat_total)} for r in cats[:5]]
    if len(cats) > 5:
        o = sum(r["count"] for r in cats[5:]); categories.append({"name": "Others", "count": o, "pct": _pct(o, cat_total)})
    subs, _ = ranked(lambda x: x["subtopic"] or x["category"], 6)
    sub_max = subs[0]["count"] if subs else 0
    subtopics = [{**r, "pct": _pct(r["count"], sub_max)} for r in subs[:6]]

    funnel = [("Started /books", n), ("Selected search method", sum(1 for x in sessions if x["chosen"] or x["selected"])),
              ("Selected category / entered book", sum(1 for x in sessions if x["selected"])),
              ("Received book results", sum(1 for x in sessions if x["results"])), ("PDF successfully sent", sent_s)]
    funnel = [{"name": a, "count": b, "pct": _pct(b, n)} for a, b in funnel]

    bk: dict[str, dict] = {}
    for x in sessions:
        for b in x["books"]:
            if not b["title"]:
                continue
            r = bk.setdefault(b["title"].lower(), {"title": b["title"], "requests": 0, "found": 0, "sent": 0})
            r["requests"] += 1; r["found"] += 1 if b["located"] else 0; r["sent"] += 1 if b["sent"] else 0
    top_books = sorted(bk.values(), key=lambda r: (-r["requests"], r["title"]))[:5]

    by_method = {m: sum(1 for x in sessions if x["method"] == m) for m in BOOK_METHODS}
    mt = sum(by_method.values())
    methods = [{"key": m, "name": BOOK_METHODS[m], "count": c, "pct": _pct(c, mt)} for m, c in by_method.items()]

    recent = [{"user": names.get(x["phone"], "User"), "phone": x["phone"], "method": BOOK_METHODS[x["method"]], "label": _book_label(x),
               "time": x["ts"], "status": x["status"]} for x in reversed(sessions[-100:])]
    return {"start": start, "end": end, "days": n_days,
            "kpis": {"requests": {"value": n, "delta": _delta(n, len(prev))}, "users": {"value": users, "delta": _delta(users, pusers)},
                     "found": {"value": found, "delta": _delta(found, pfound)}, "failed": {"value": failed, "delta": _delta(failed, pfailed)}},
            "categories": categories, "category_total": cat_total, "subtopics": subtopics,
            "trend": {"labels": days, "topic": [per["topic"][x] for x in days], "name": [per["name"][x] for x in days]},
            "funnel": funnel,
            "delivery": {"found": found, "sent": sum(x["sent"] for x in sessions), "alts": sum(1 for x in sessions if x["status"] == "alts"), "none": failed},
            "top_books": top_books, "methods": methods, "method_total": mt, "recent": recent}


# ======================================================================= AI Providers health (chat_pdf fallback chain)
def ai_providers_page(start: str | None = None, end: str | None = None) -> dict:
    """Which AI provider/model works right now, from the attempts chat_pdf logged (table ai_provider_log)."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    start = start or (e - timedelta(days=6)).strftime("%Y-%m-%d")
    rows, recent = [], []
    try:
        import chat_pdf
        conf = chat_pdf.provider_status()
    except Exception:
        conf = {"llm": [], "embed": [], "embed_order": "", "llm_order": ""}
    with _conn() as conn:
        try:
            cur = conn.raw("""SELECT kind, provider, model, ok, ms, error,
                                     to_char(ts AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                                FROM ai_provider_log
                               WHERE to_char(ts AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s
                               ORDER BY ts, id""", (start, end))
            rows = cur.fetchall()
        except Exception:
            rows = []
    now = datetime.utcnow()
    by: dict[tuple, dict] = {}
    for r in rows:
        k = (r["kind"], r["provider"], r["model"] if r["kind"] == "llm" else r["provider"])
        d = by.setdefault(k, {"kind": r["kind"], "provider": r["provider"], "model": k[2], "calls": 0, "ok": 0, "ms": [], "last": [],
                              "last_ok": None, "last_fail": None, "last_error": ""})
        d["calls"] += 1
        d["ok"] += 1 if r["ok"] else 0
        if r["ok"]:
            d["ms"].append(r["ms"] or 0); d["last_ok"] = r["ts"]
        else:
            d["last_fail"] = r["ts"]; d["last_error"] = r["error"] or ""
            recent.append({"time": r["ts"], "kind": r["kind"], "provider": r["provider"], "model": k[2], "error": r["error"] or "", "ms": r["ms"] or 0})
        d["last"].append(bool(r["ok"]))
    cool = {(x["provider"], x["model"]): x["cooldown_s"] for x in conf["llm"]}
    cfg = {(x["provider"], x["model"]): x for x in conf["llm"]}
    cfg_e = {x["provider"]: x for x in conf["embed"]}
    out = []
    for k, d in by.items():
        recent3 = d["last"][-3:]
        status = "working" if d["last"][-1] else "failing" if len(recent3) >= 2 and not any(recent3) else "degraded"
        if d["kind"] == "llm" and cool.get((d["provider"], d["model"]), 0) > 0:
            status = "cooling"
        out.append({"kind": d["kind"], "provider": d["provider"], "model": d["model"], "calls": d["calls"], "ok": d["ok"], "fail": d["calls"] - d["ok"],
                    "success": _pct(d["ok"], d["calls"]), "avg_ms": int(sum(d["ms"]) / len(d["ms"])) if d["ms"] else None,
                    "last_ok": d["last_ok"], "last_fail": d["last_fail"], "last_error": d["last_error"][:160],
                    "status": status, "cooldown_s": cool.get((d["provider"], d["model"]), 0)})
    seen = {(x["kind"], x["provider"], x["model"]) for x in out}
    for x in conf["llm"]:                       # configured but never used (yet)
        if x["configured"] and ("llm", x["provider"], x["model"]) not in seen:
            out.append({"kind": "llm", "provider": x["provider"], "model": x["model"], "calls": 0, "ok": 0, "fail": 0, "success": None, "avg_ms": None,
                        "last_ok": None, "last_fail": None, "last_error": "", "status": "unused", "cooldown_s": x["cooldown_s"]})
    for x in conf["embed"]:
        if x["configured"] and ("embed", x["provider"], x["provider"]) not in seen and not any(o["kind"] == "embed" and o["provider"] == x["provider"] for o in out):
            out.append({"kind": "embed", "provider": x["provider"], "model": x["provider"], "calls": 0, "ok": 0, "fail": 0, "success": None, "avg_ms": None,
                        "last_ok": None, "last_fail": None, "last_error": "", "status": "unused", "cooldown_s": 0})
    order = {n: i for i, n in enumerate(conf["llm_order"].split(","))}
    out.sort(key=lambda x: (x["kind"], order.get(x["provider"], 99), x["model"]))
    recent.sort(key=lambda r: r["time"], reverse=True)
    calls = sum(x["calls"] for x in out); okc = sum(x["ok"] for x in out)
    not_configured = [x["provider"] for x in conf["llm"] if not x["configured"]]
    return {"start": start, "end": end, "providers": out, "recent_errors": recent[:25],
            "kpis": {"configured": len({x["provider"] for x in conf["llm"] if x["configured"]}),
                     "working": sum(1 for x in out if x["status"] == "working"),
                     "failing": sum(1 for x in out if x["status"] in ("failing", "cooling")),
                     "calls": calls, "success": _pct(okc, calls)},
            "llm_order": conf["llm_order"], "embed_order": conf["embed_order"], "not_configured": sorted(set(not_configured))}
