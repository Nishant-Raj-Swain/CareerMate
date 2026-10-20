# feedback.py  -  /feedback : users send feedback, an ML model labels it, the admin panel shows it
# ---------------------------------------------------------------------------
#   /feedback            -> bot asks for the feedback; /feedback <text> works in one message
#   user text            -> classified by a machine-learning model into  POSITIVE / NEGATIVE / SUGGESTION
#                           (TF-IDF word + character n-grams  ->  Logistic Regression, scikit-learn),
#                           saved in the `feedback` table with the model's confidence, then thanked.
#   admin panel          -> GET /api/analytics/feedback/page   (overview, distribution, trend, recent, to-do lists)
#                           POST /api/analytics/feedback/resolve  {id, resolved}
#
# The model is trained at first use from SEED (below, ~190 labelled sentences) - nothing to download, no API key.
# More examples = better accuracy: the admin can correct a label (POST feedback/relabel) and corrected rows are added to the
# training set at the next restart. Sub-categories (Helpful / Bug / Slow response / New feature / UI ...) and the tag
# (Search, Competitions, Performance ...) come from simple keyword rules, not from the ML model.
# If scikit-learn is missing the bot still works with a keyword fallback (add `scikit-learn` to requirements.txt).
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import datetime, timedelta

log = logging.getLogger("feedback")

MODE = "feedback"
LABELS = ("positive", "negative", "suggestion")
MAX_LEN = 1000
EXIT_WORDS = {"exit", "cancel", "stop", "quit", "/exit", "/cancel"}

# ----------------------------------------------------------------------- training data
SEED = {
    "positive": [
        "Great bot! Found the perfect internship through your bot.", "Thanks, the resources were really helpful", "Thank you so much, this bot is amazing",
        "Loved the resume analysis, very useful", "Awesome work, keep it up", "The quiz questions are really good and helped me prepare",
        "Very helpful guidance for my placements", "I got a job interview thanks to the job search feature", "Excellent service, super easy to use",
        "This is the best career assistant I have used", "Really impressed with the ATS score feedback", "Nice bot, works smoothly",
        "Good experience overall", "Everything works fine and the replies are quick", "The notes from YouTube videos saved me hours, thank you",
        "Very useful for students like me", "Great job team, I love it", "It helped me improve my resume a lot",
        "Smooth and simple to use, I am satisfied", "Brilliant idea, this bot is a lifesaver", "Perfect, exactly what I needed",
        "The practice tests are fantastic", "I like the way it explains answers, very clear", "Superb, thanks for the free books feature",
        "Happy with the results, recommended it to my friends", "Wonderful bot, got great internship options", "Really good, the roadmap helped me a lot",
        "Loved it, thank you for building this", "Amazing, the PDF chat feature is so handy", "Very nice and informative",
        "The bot is fast and accurate, good work", "Super helpful, cleared my doubts quickly", "Best bot for career help",
        "Thanks a lot, it was really useful for my exam preparation", "Excellent features, I am enjoying it", "Good bot, easy and quick",
        "It works great, no complaints", "Thank you, found my dream company in the list", "I am very satisfied with this service",
        "Fantastic experience, highly recommend", "The job alerts are spot on, great", "Nice work, the interface in chat is simple and clean",
        "Really appreciate this free tool", "It is very helpful for freshers", "Impressive, the answers are correct and detailed",
        "Good job, keep going", "My friends and I use it daily and love it", "Super bot, saved my time",
        "great app very helpful thanks", "bahut accha bot hai, bahut helpful", "bohot badiya, thank you", "mast hai yaar, kaafi useful",
        "The mock quizzes boosted my confidence", "Wow, the notes PDF is neat and well organised", "Helpful bot, got all the info I wanted",
        "Appreciate the quick replies", "Cool bot, I like it", "Good work guys, loving the career guidance", "Thanks for the amazing resume tips",
        "Everything is perfect", "I found it very useful, thanks", "This bot made my job hunt easier",
    ],
    "negative": [
        "The job search results are sometimes not relevant", "App is slow sometimes, especially in the evening", "The bot is not working",
        "It keeps giving me an error", "Very slow response, I waited for minutes", "The /competitions command sometimes shows old events",
        "Resume score is wrong, it gave me 20 for a good resume", "The bot crashed when I sent my PDF", "Quiz questions are repeated again and again",
        "Bad experience, nothing worked", "I did not get any reply", "The notes feature failed and sent no PDF",
        "Useless results, nothing matches my domain", "The bot stopped responding in the middle of the quiz", "Too many errors, very frustrating",
        "Not helpful at all", "The answers are incorrect", "It takes too long to reply", "The books command could not find any book",
        "Links are broken and do not open", "PDF upload failed again", "Worst bot, wasted my time", "Internship results are outdated",
        "It says something went wrong every time", "Bot is lagging and hangs", "The responses are very late",
        "I am disappointed, it never understands my question", "Doesn't work for me", "The job links are expired",
        "Error while generating the roadmap", "Not satisfied with the quality of answers", "The quiz gave me wrong answers",
        "It is not able to read my resume", "The reply was irrelevant to what I asked", "Too slow and sometimes no response",
        "The bot ignores my messages", "Search keeps showing the same jobs", "The ATS score is not accurate", "Bug: menu does not open",
        "I faced a lot of issues, it is buggy", "Poor performance today", "The PDF is blank", "The video notes are wrong and incomplete",
        "Failed to download the book", "It is confusing and does not work properly", "Stuck on loading forever", "Terrible, it deleted my progress",
        "bot slow hai bahut", "kaam nahi kar raha", "bakwas bot, kuch nahi milta", "reply nahi aa raha", "error aa raha hai baar baar",
        "The responses take ages", "Got no results for my city", "Wrong information about the company", "Keeps timing out",
        "The chat froze and I had to restart", "I don't like the results, they are poor", "It is not good, many mistakes",
        "Very bad, the bot hangs on every command", "The quiz is broken", "Not working since morning", "The internship list is empty",
    ],
    "suggestion": [
        "Please add a Dark Mode option, it will be really helpful", "Can you add more data science competitions?", "It would be great if you can add a resume builder section",
        "Add more filters in internship search", "Please add a mobile app version", "I suggest adding a Hindi language option",
        "Would be nice to have interview practice with voice", "You should add more quiz topics like Machine Learning", "Please include a cover letter generator",
        "Can you add previous year question papers?", "It would be better if the bot remembers my last search", "Add an option to save jobs for later",
        "Suggestion: add salary details in job results", "Please add notifications for new internships", "I think you should add a mock interview feature",
        "Could you add support for more languages?", "Add more books on programming", "Please make the menu easier with buttons",
        "It would be helpful to have a daily quiz reminder", "Can you include company wise interview questions?", "Add GATE preparation material please",
        "I wish there was an option to download my quiz results", "Please add a feature to compare two resumes", "Maybe add a leaderboard for quizzes",
        "You could improve the UI by adding more buttons", "Make the PDF notes look better with colours", "The design could be better, please improve the layout",
        "Please add a feature to share the notes with friends", "Consider adding an option to filter jobs by salary", "Add remote jobs filter",
        "It would be nice if the bot could suggest courses", "Can you add LinkedIn profile review?", "Please add video lectures links also",
        "Need an option to choose the number of quiz questions", "Add support for Word files in resume upload", "Would love to see a progress tracker",
        "Please add scholarships information", "Include more cities in the job search", "Why not add a study planner feature?",
        "Add a way to give ratings to each feature", "Please add an option to export the chat as PDF", "Add more difficulty levels in the quiz",
        "The bot should send weekly job updates", "It would be great if I could search books by author", "Please add hackathon updates",
        "My suggestion is to add a dashboard for my own progress", "Add voice message support", "please add more competitions for students",
        "Kindly add a resume template download option", "Can there be an option to set my preferred domain once?", "Add free certification course list",
        "you should add telugu language", "kya aap internship alerts add kar sakte ho", "please add dark mode", "Add placement papers for companies",
        "Improve the UI, buttons would be better than typing numbers", "It would be good to have a back button in every menu", "Suggest adding GD topics",
        "I'd like to see more filters like experience and location", "Add an option to delete my data", "Maybe you can add a feature for aptitude practice",
        "Thanks, but please add more internships in Orissa", "Good bot, it would be even better with a resume builder", "Nice work, can you add a dark theme",
        "Great, one more thing: add a notes option for PDFs too", "Helpful, however add more quiz categories",
    ],
}

# ----------------------------------------------------------------------- the model
_model = None
_model_lock = threading.Lock()
_extra_rows: list[tuple[str, str]] = []                    # admin-corrected rows added to training at start-up


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:MAX_LEN]


def _training_set() -> tuple[list[str], list[str]]:
    xs, ys = [], []
    for label, rows in SEED.items():
        xs += rows; ys += [label] * len(rows)
    for text, label in _extra_rows:
        if label in LABELS:
            xs.append(text); ys.append(label)
    return xs, ys


def _build_model(xs: list[str], ys: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("clf", LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced")),
    ]).fit(xs, ys)


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    _model = _build_model(*_training_set())
                except ImportError:
                    log.warning("scikit-learn is not installed - using the keyword fallback classifier (pip install scikit-learn)")
                    _model = False
    return _model


# --- keyword fallback + sub-categories -------------------------------------------------------
_SUGGEST = re.compile(r"\b(please add|pls add|add (a|an|more|the|some)?|can you add|could you add|should add|you should|would be (great|nice|better|good|helpful)|"
                      r"it would|i suggest|suggestion|suggest|i wish|wish there|why not|maybe add|consider|include|need an option|kindly add|feature request|"
                      r"improve the|should have|kya aap .* add)\b", re.I)
_NEG = re.compile(r"\b(not working|doesn'?t work|does not work|didn'?t work|error|bug|buggy|crash|slow|lag|hang|stuck|wrong|fail|failed|broken|useless|worst|bad|"
                  r"terrible|poor|not relevant|irrelevant|disappoint|no reply|no response|nahi|bakwas|issue|problem|frustrat|waste)\b", re.I)
_POS = re.compile(r"\b(great|good|thanks|thank you|awesome|amazing|love|loved|helpful|useful|excellent|perfect|nice|best|fantastic|wonderful|superb|brilliant|"
                  r"satisfied|happy|accha|badiya|mast)\b", re.I)


def _rule_label(text: str) -> tuple[str, float]:
    s, n, p = bool(_SUGGEST.search(text)), len(_NEG.findall(text)), len(_POS.findall(text))
    if s and n <= p + 1:
        return "suggestion", 0.6
    if n > p:
        return "negative", 0.6
    if p > 0:
        return "positive", 0.6
    return ("suggestion", 0.4) if s else ("positive", 0.34)


def classify(text: str) -> tuple[str, float]:
    """-> (label, confidence 0..1). ML model; keyword fallback if scikit-learn is unavailable."""
    text = _tidy(text)
    model = get_model()
    if not model:
        return _rule_label(text)
    probs = model.predict_proba([text])[0]
    i = int(probs.argmax())
    return str(model.classes_[i]), round(float(probs[i]), 3)


_SUBCATS = {
    "positive": [("Appreciation", r"thank|great|awesome|amazing|love|excellent|fantastic|wonderful|superb|brilliant|best|badiya|mast|appreciate"),
                 ("Helpful", r"help|useful|guidance|resource|saved|improv|clear|got (a|the|my)|prepare")],
    "negative": [("Slow Response", r"slow|late|delay|lag|takes? (too )?long|loading|timeout|timing out|hang|froze|freez|ages|wait"),
                 ("Feature Missing", r"missing|no option|can'?t find|lacks?|not available|doesn'?t have|empty|no results|not able to|could not find|couldn'?t find"),
                 ("Bug / Error", r"error|bug|crash|not working|doesn'?t work|fail|broken|stuck|blank|wrong|nahi|expired|stopped|issue|incorrect")],
    "suggestion": [("UI/UX Improvements", r"\bui\b|\bux\b|design|dark|theme|layout|button|menu|look|colou?r|interface|back button"),
                   ("New Features", r"add|feature|option|include|support|integrat|builder|filter|alert|notification|language|tracker|template|mock|leaderboard")],
}
_DEFAULT_SUB = {"positive": "Good Experience", "negative": "Other Issue", "suggestion": "Other Ideas"}
_TAGS = [("Performance", r"slow|lag|late|delay|hang|loading|timeout|crash|froze"), ("Competitions", r"competition|hackathon|contest|event"),
         ("Search", r"job|internship|search|filter|city|domain|salary|remote"), ("UI/UX", r"\bui\b|\bux\b|design|dark|theme|layout|button|menu"),
         ("Resume", r"resume|\bcv\b|ats"), ("Quiz", r"quiz|question|practice|mock|test"), ("Books", r"\bbooks?\b|pdf"),
         ("Notes", r"notes|youtube|video")]


def subcategory(label: str, text: str) -> str:
    for name, pat in _SUBCATS.get(label, []):
        if re.search(pat, text, re.I):
            return name
    return _DEFAULT_SUB.get(label, "Other")


def tag_of(label: str, text: str) -> str:
    for name, pat in _TAGS:
        if re.search(pat, text, re.I):
            return name
    return "Feature" if label == "suggestion" else "General"


# ----------------------------------------------------------------------- storage
def _conn():
    import analytics
    return analytics._conn()


def init_db():
    """Creates the `feedback` table (safe on every start) and loads admin-corrected rows for training."""
    with _conn() as conn:
        conn.raw("""CREATE TABLE IF NOT EXISTS feedback (
                        id SERIAL PRIMARY KEY, user_phone TEXT, message TEXT NOT NULL, label TEXT NOT NULL,
                        confidence REAL, category TEXT, tag TEXT, resolved BOOLEAN DEFAULT FALSE,
                        corrected BOOLEAN DEFAULT FALSE, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW())""")
        conn.raw("CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)")
        rows = conn.raw("SELECT message, label FROM feedback WHERE corrected IS TRUE").fetchall()
    _extra_rows[:] = [(r["message"], r["label"]) for r in rows]


def save(phone: str, text: str) -> dict:
    """Classifies and stores one feedback message. Returns {id, label, confidence, category, tag}."""
    text = _tidy(text)
    label, conf = classify(text)
    cat, tag = subcategory(label, text), tag_of(label, text)
    with _conn() as conn:
        conn.raw("INSERT INTO users (phone, resume, state, doc_text) VALUES (%s, '', '{}', '') ON CONFLICT (phone) DO NOTHING", (phone,))
        row = conn.raw("INSERT INTO feedback (user_phone, message, label, confidence, category, tag) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                       (phone, text, label, conf, cat, tag)).fetchone()
    return {"id": row["id"], "label": label, "confidence": conf, "category": cat, "tag": tag}


def set_resolved(fid: int, resolved: bool) -> bool:
    with _conn() as conn:
        return conn.raw("UPDATE feedback SET resolved=%s WHERE id=%s", (bool(resolved), int(fid))).rowcount > 0


def relabel(fid: int, label: str) -> bool:
    """Admin correction: fixes the label (and sub-category) and marks the row as training data for the next restart."""
    if label not in LABELS:
        raise ValueError("label must be positive, negative or suggestion")
    with _conn() as conn:
        row = conn.raw("SELECT message FROM feedback WHERE id=%s", (int(fid),)).fetchone()
        if not row:
            return False
        conn.raw("UPDATE feedback SET label=%s, category=%s, tag=%s, corrected=TRUE, confidence=1 WHERE id=%s",
                 (label, subcategory(label, row["message"]), tag_of(label, row["message"]), int(fid)))
    _extra_rows.append((row["message"], label))
    return True


# ----------------------------------------------------------------------- admin data
def _pct(n, total):
    return round(100.0 * n / total, 1) if total else 0.0


def page(start: str | None = None, end: str | None = None) -> dict:
    """Everything the Feedback Analytics page shows, for the date range (inclusive, 'YYYY-MM-DD')."""
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    start = start or (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=29)).strftime("%Y-%m-%d")
    where, params = "to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD') BETWEEN %s AND %s", (start, end)
    with _conn() as conn:
        rows = conn.raw(f"""SELECT id, user_phone, message, label, confidence, category, tag, resolved,
                                   to_char(created_at AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI:SS') AS ts
                              FROM feedback WHERE {where} ORDER BY created_at DESC, id DESC""", params).fetchall()
        users = {r["user_phone"]: f"User {i}" for i, r in enumerate(
            conn.raw("SELECT user_phone FROM feedback GROUP BY user_phone ORDER BY MIN(created_at)").fetchall(), 1)}
    total = len(rows)
    counts = {k: sum(1 for r in rows if r["label"] == k) for k in LABELS}
    dist = {}
    for k in LABELS:
        sub: dict[str, int] = {}
        for r in rows:
            if r["label"] == k:
                sub[r["category"] or _DEFAULT_SUB[k]] = sub.get(r["category"] or _DEFAULT_SUB[k], 0) + 1
        dist[k] = {"total": counts[k], "pct": _pct(counts[k], total),
                   "items": [{"name": n, "count": c, "pct": _pct(c, counts[k])} for n, c in sorted(sub.items(), key=lambda x: -x[1])]}
    days, d = [], datetime.strptime(start, "%Y-%m-%d")
    while d.strftime("%Y-%m-%d") <= end and len(days) < 400:
        days.append(d.strftime("%Y-%m-%d")); d += timedelta(days=1)
    by_day = {k: {x: 0 for x in days} for k in LABELS}
    for r in rows:
        day = r["ts"][:10]
        if day in by_day[r["label"]]:
            by_day[r["label"]][day] += 1

    def item(r):
        return {"id": r["id"], "phone": r["user_phone"], "user": users.get(r["user_phone"], "User"), "message": r["message"], "type": r["label"],
                "category": r["category"], "tag": r["tag"], "confidence": r["confidence"], "resolved": bool(r["resolved"]), "time": r["ts"]}
    recent = [item(r) for r in rows[:100]]
    high = [item(r) for r in rows if r["label"] == "negative"]
    sugg = [item(r) for r in rows if r["label"] == "suggestion"]
    return {"start": start, "end": end, "total": total, "counts": counts,
            "pct": {k: _pct(counts[k], total) for k in LABELS}, "distribution": dist,
            "trend": {"labels": days, **{k: [by_day[k][x] for x in days] for k in LABELS}},
            "recent": recent,
            "todo": {"high": high[:30], "high_open": sum(1 for x in high if not x["resolved"]),
                     "suggestions": sugg[:30], "suggestions_open": sum(1 for x in sugg if not x["resolved"])},
            "model": "TF-IDF + Logistic Regression" if get_model() else "keyword fallback"}


# ----------------------------------------------------------------------- bot conversation
PROMPT = ("💬 *Feedback*\n\nTell me what you think about CareerMate - what you liked, what went wrong, or any idea/suggestion. "
          "Just type it in one message.\n\nSend *exit* to cancel.")
THANKS = {
    "positive": "😊 Thank you so much for the kind words! It really motivates the team. Share the bot with your friends!",
    "negative": "🙏 Sorry about that, and thank you for telling us. Your feedback has been sent to the team and we'll work on fixing it.",
    "suggestion": "💡 Great suggestion, thank you! We've passed it to the team to consider for a future update.",
}


def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


def _set_state(phone: str, state: dict):
    import database as db
    db.save_state(phone, state)


async def start(phone: str, arg: str, say) -> bool:
    """/feedback [text]. True = finished (state cleared)."""
    if arg and arg.strip():
        return await handle(phone, arg, say)
    await asyncio.to_thread(_set_state, phone, {"mode": MODE})
    await say(PROMPT)
    return False


async def handle(phone: str, text: str, say) -> bool:
    text = _tidy(text)
    if text.lower() in EXIT_WORDS:
        await asyncio.to_thread(_set_state, phone, {})
        await say("👋 Cancelled. Send */feedback* any time.")
        return True
    if len(text) < 3:
        await say("🤔 Please type your feedback in a few words (or *exit*).")
        return False
    try:
        res = await asyncio.to_thread(save, phone, text)
    except Exception:
        log.exception("could not save feedback")
        await asyncio.to_thread(_set_state, phone, {})
        await say("⚠️ I couldn't save your feedback just now. Please try again in a few minutes.")
        return True
    await asyncio.to_thread(_set_state, phone, {})
    try:
        import analytics
        analytics.log_event(phone, "feedback", command="/feedback", meta={"label": res["label"], "category": res["category"]})
    except Exception:
        pass
    await say(THANKS[res["label"]])
    return True


# ----------------------------------------------------------------------- CLI
if __name__ == "__main__":  # pragma: no cover
    import sys
    for t in sys.argv[1:] or ["Please add dark mode", "bot is very slow", "great bot thanks"]:
        print(classify(t), subcategory(classify(t)[0], t), "|", t)
