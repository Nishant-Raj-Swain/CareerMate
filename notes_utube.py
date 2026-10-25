# notes_utube.py
# ---------------------------------------------------------------------------
# /utube_notes - YouTube link in, study notes out (WhatsApp summary + a PDF).
#
#   user:  /utube_notes
#   bot:   "Send me the YouTube link"
#   user:  https://youtu.be/xxxxxxxxxxx
#   bot:   progress messages, then thumbnail + summary / takeaways / outline + a PDF with the full notes
#
# Pipeline
#   1. video id from any YouTube URL form (watch, youtu.be, shorts, live, embed, mobile)
#   2. transcript with youtube-transcript-api (manual captions preferred, then auto captions;
#      English first, otherwise whatever language the video has)
#   3. not English -> translated to English chunk by chunk (timestamps kept)
#   4. notes: the transcript is cut into ~9k-character windows; each window becomes 1-4 titled sections of
#      point-wise notes (+ key terms); a final pass writes the summary, takeaways and revision questions
#   5. PDF (reportlab): thumbnail, summary, takeaways, sections with clickable timestamps, glossary, Q&A
#   6. fallback when YouTube blocks the server / has no captions: Gemini reads the video from its URL
#
# Uses chat_pdf.llm() for the text models (NVIDIA Kimi -> OpenRouter -> Gemini, same keys as /chat_pdf).
#
# Needs:   pip install youtube-transcript-api reportlab requests
# Env (all optional):
#   NOTES_PROXY_URL            e.g. http://user:pass@host:port  - YouTube often blocks cloud-server IPs
#                              (Render, AWS, ...). A proxy fixes it; the Gemini fallback also covers most cases.
#   WEBSHARE_PROXY_USER / WEBSHARE_PROXY_PASS    use Webshare residential proxies instead
#   NOTES_MAX_MINUTES (240)    longest video accepted        NOTES_MAX_PER_DAY (6)   videos per user per day
#   NOTES_GEMINI_VIDEO (1)     0 disables the Gemini-reads-the-video fallback
#   WHATSAPP_GRAPH_BASE        default https://graph.facebook.com/v21.0   (WHATSAPP_ACCESS_TOKEN and
#                              WHATSAPP_PHONE_NUMBER_ID are the same variables main.py already uses)
#
# main.py hooks:
#   await notes_utube.start(phone, arg, say=lambda m: wa.text(phone, m))        # on /utube_notes [link]
#   done = await notes_utube.handle(phone, text, say=...)                        # while state mode == MODE
#   (handle() returns True when the job is over and the state was cleared -> show the menu)
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import io
import math
import threading
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests

import chat_pdf
import database as db

log = logging.getLogger("notes_utube")

MODE = "notes_utube"
WINDOW_CHARS = 12000                      # transcript characters per notes window (bigger = fewer AI calls)
MAX_MINUTES = int(float(os.getenv("NOTES_MAX_MINUTES", "240")))
MAX_PER_DAY = int(os.getenv("NOTES_MAX_PER_DAY", "6"))
WORKERS = max(1, int(os.getenv("NOTES_WORKERS", "3")))     # parallel AI calls for long videos
WA_LIMIT = 3800
EXIT_WORDS = {"exit", "cancel", "stop", "quit", "/exit", "/cancel", "back"}
CACHE_SECONDS = 24 * 3600

PROMPT = ("🎬 *YouTube Notes*\n\nSend me the YouTube link and I'll turn the video into point-wise study notes "
          "(summary, key points with timestamps, key terms and revision questions) as a PDF.\n\n"
          "_Works with any language - I translate to English. Send *exit* to cancel._")


class NotesError(Exception):
    """Raised with a message that is safe to show to the user."""


# ------------------------------------------------------------------ helpers
_ID = r"([A-Za-z0-9_-]{11})"
_URL_PATTERNS = [
    re.compile(r"youtu\.be/" + _ID),
    re.compile(r"youtube(?:-nocookie)?\.com/(?:watch\?(?:[^\s#]*&)?v=|shorts/|live/|embed/|v/)" + _ID),
]


def extract_video_id(text: str) -> str | None:
    for pat in _URL_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(1)
    return None


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_ts(s) -> int:
    """'1:02:03' / '12:34' / '754' -> seconds (0 if unreadable)."""
    try:
        parts = [int(p) for p in re.findall(r"\d+", str(s))][:3]
    except ValueError:
        return 0
    if not parts:
        return 0
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def link_at(video_id: str, sec: int = 0) -> str:
    return f"https://youtu.be/{video_id}" + (f"?t={int(sec)}" if sec else "")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


# ----------------------------------------------------------------- metadata
def fetch_meta(video_id: str) -> dict:
    try:
        r = requests.get("https://www.youtube.com/oembed",
                         params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"}, timeout=10)
        if r.ok:
            j = r.json()
            return {"title": j.get("title") or "YouTube video", "channel": j.get("author_name") or ""}
    except Exception as e:
        log.info("oembed failed: %s", e)
    return {"title": "YouTube video", "channel": ""}


def fetch_thumbnail(video_id: str) -> bytes | None:
    for name in ("maxresdefault", "hqdefault"):
        try:
            r = requests.get(f"https://i.ytimg.com/vi/{video_id}/{name}.jpg", timeout=10)
            if r.ok and len(r.content) > 5000:
                return r.content
        except Exception:
            pass
    return None


# --------------------------------------------------------------- transcript
def _proxy_config():
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig
    except Exception:
        return None
    if _env("NOTES_PROXY_URL"):
        return GenericProxyConfig(http_url=_env("NOTES_PROXY_URL"), https_url=_env("NOTES_PROXY_URL"))
    if _env("WEBSHARE_PROXY_USER") and _env("WEBSHARE_PROXY_PASS"):
        return WebshareProxyConfig(proxy_username=_env("WEBSHARE_PROXY_USER"), proxy_password=_env("WEBSHARE_PROXY_PASS"))
    return None


class TranscriptUnavailable(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind          # "none" | "blocked" | "unavailable" | "error"


def translate_mode() -> str:
    """youtube (default): YouTube's free caption translation, no AI call | none: the notes model translates while
    writing | llm: a separate AI translation pass per chunk (best wording, most AI calls)."""
    m = _env("NOTES_TRANSLATE", "youtube").lower()
    return m if m in ("youtube", "none", "llm") else "youtube"


def fetch_transcript(video_id: str, translate_en: bool = False) -> dict:
    """{'lang','lang_name','generated','translated','snippets':[(start_sec, text)], 'duration'}
    translate_en=True asks YouTube to translate non-English captions to English (free, no AI)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import youtube_transcript_api._errors as E
    except ImportError:
        raise TranscriptUnavailable("error", "youtube-transcript-api is not installed")
    api = YouTubeTranscriptApi(proxy_config=_proxy_config())
    last = None
    for attempt in range(2):
        try:
            tl = api.list(video_id)
            tr = _pick_transcript(tl)
            translated = False
            if translate_en and not _is_english(tr.language_code) and getattr(tr, "is_translatable", False):
                try:
                    fetched = tr.translate("en").fetch()
                    translated = True
                except Exception as e:                      # not translatable after all: use the original captions
                    log.info("YouTube translation unavailable (%s); using original captions", e)
                    fetched = tr.fetch()
            else:
                fetched = tr.fetch()
            snippets = [(float(s.start), re.sub(r"\s+", " ", s.text.replace("\n", " ")).strip()) for s in fetched]
            snippets = [(a, t) for a, t in snippets if t and not re.fullmatch(r"\[[^\]]*\]", t)]   # drop [Music] etc.
            if not snippets:
                raise TranscriptUnavailable("none", "empty transcript")
            last_s = fetched[-1]
            return {"lang": tr.language_code, "lang_name": tr.language, "generated": bool(tr.is_generated),
                    "translated": translated, "snippets": snippets, "duration": float(last_s.start) + float(getattr(last_s, "duration", 0) or 0)}
        except TranscriptUnavailable:
            raise
        except (E.TranscriptsDisabled, E.NoTranscriptFound) as e:
            raise TranscriptUnavailable("none", str(e)[:120])
        except (E.IpBlocked, E.RequestBlocked) as e:
            raise TranscriptUnavailable("blocked", str(e)[:120])
        except (E.VideoUnavailable, E.VideoUnplayable, E.InvalidVideoId, E.AgeRestricted) as e:
            raise TranscriptUnavailable("unavailable", str(e)[:120])
        except Exception as e:                      # network hiccup etc.
            last = e
            time.sleep(1.5)
    raise TranscriptUnavailable("error", str(last)[:160])


def _pick_transcript(tl):
    """English manual > English auto > any manual > any auto."""
    items = list(tl)
    def score(t):
        en = t.language_code.lower().startswith("en")
        return (0 if en else 2) + (0 if not t.is_generated else 1)
    items.sort(key=score)
    if not items:
        raise TranscriptUnavailable("none", "no transcripts listed")
    return items[0]


def _is_english(lang: str) -> bool:
    return (lang or "").lower().startswith("en")


# ----------------------------------------------------------- LLM utilities
def _llm(messages: list[dict], max_tokens: int = 3000) -> str:
    """Single seam for the text models (tests replace it)."""
    return chat_pdf.llm(messages, max_tokens=max_tokens)


def _parse_json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except ValueError:
            pass
    return None


def _json_call(system: str, user: str, max_tokens: int):
    for attempt in range(2):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if attempt:
            msgs.append({"role": "user", "content": "Your last reply was not valid JSON. Reply again with ONLY the JSON object."})
        out = _llm(msgs, max_tokens)
        data = _parse_json(out)
        if isinstance(data, dict):
            return data
    return None


# --------------------------------------------------------------- chunking
def build_windows(snippets: list[tuple[float, str]], limit: int = WINDOW_CHARS) -> list[dict]:
    """Transcript text with a [m:ss] marker about every 25 s, cut into windows of ~`limit` characters."""
    windows, buf, size, w_start, last_tag = [], [], 0, None, -999.0
    for start, text in snippets:
        if w_start is None:
            w_start = start
        tag = ""
        if start - last_tag >= 25 or not buf:
            tag = f"\n[{fmt_ts(start)}] "
            last_tag = start
        piece = tag + text + " "
        if size + len(piece) > limit and buf:
            windows.append({"start": w_start, "text": "".join(buf).strip()})
            buf, size, w_start, last_tag = [], 0, start, -999.0
            piece = f"\n[{fmt_ts(start)}] " + text + " "
            last_tag = start
        buf.append(piece); size += len(piece)
    if buf:
        windows.append({"start": w_start or 0, "text": "".join(buf).strip()})
    return windows


# ------------------------------------------------------------- translation
def translate_windows(windows: list[dict], lang_name: str) -> tuple[list[dict], int]:
    """Translates each window to English (keeps the [m:ss] markers). Returns (windows, failed_count)."""
    system = ("You are a professional translator. Translate the transcript below from %s into clear, natural English. "
              "Keep every [m:ss] timestamp marker exactly where it is. Fix obvious speech-recognition errors from context, "
              "keep technical terms, names, numbers and code unchanged, and do not summarise or add anything. "
              "Output only the translation." % lang_name)

    def one(w):
        try:
            out = _llm([{"role": "system", "content": system}, {"role": "user", "content": w["text"]}],
                       max_tokens=min(8192, max(2000, len(w["text"]) // 2)))
            return {**w, "text": out.strip()}, 0
        except Exception as e:
            log.warning("translation failed for a window: %s", e)
            return w, 1
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(one, windows))
    return [r[0] for r in res], sum(r[1] for r in res)


# ------------------------------------------------------------------- notes
SECTION_SYSTEM = """You turn video transcripts into excellent study notes for a student who has NOT watched the video.
Return ONLY one JSON object:
{"sections":[{"title":"...","start":"m:ss","points":["..." or {"p":"...","sub":["...","..."]}],"terms":[{"term":"...","meaning":"..."}]}]}
Rules:
- Make 2-5 sections for the text given (one per distinct topic, roughly one per 3-5 minutes of speech), in the order spoken. "start" must be copied from the nearest [m:ss] marker where that topic begins.
- title: 3-8 words, specific (never "Introduction part 2").
- points: 4-10 per section. Each point is a complete, self-contained statement (not a fragment): the idea, the reason or mechanism behind it, and any number, formula, step, example, name or code the speaker gave. Use "sub" bullets for steps, lists or examples that belong to a point.
- Keep the speaker's facts exactly; never invent facts or add outside knowledge. Skip greetings, sponsor reads, "like and subscribe" and filler.
- terms: only important technical terms or definitions introduced in this text (0-5), meaning in one plain sentence.
- Write in clear English even if the transcript is in another language. Plain text only: no markdown, no emojis."""

OVERVIEW_SYSTEM = """You write the front page of a study-notes document from the section notes of one video.
Return ONLY one JSON object:
{"subject":"2-5 word topic label","tldr":"a 4-6 sentence summary of the whole video: what it teaches, how it is organised, what the learner can do afterwards","takeaways":["..."],"insights":[{"t":"short title","d":"2-3 sentences: the idea, why it matters, an example from the video"}],"questions":[{"q":"...","a":"..."}]}
Rules: takeaways = 6-8 of the most important, concrete points a learner must remember (full sentences). insights = 5-7 deeper ideas, trade-offs or common mistakes the speaker explained. questions = 5 revision questions that test understanding of the notes (not trivia), with short correct answers taken from the notes. Plain text only. Use only the provided notes."""


COMBINED_SYSTEM = SECTION_SYSTEM.replace(
    '{"sections":', '{"subject":"2-5 word topic","tldr":"4-6 sentences","takeaways":["6-8 key points"],'
    '"insights":[{"t":"short title","d":"2-3 sentences"}],"questions":[{"q":"...","a":"..."}],"sections":') + \
    "\n- Also give 5 revision questions with short answers, 5-7 insights and 6-8 takeaways."


def _clean_points(raw) -> list:
    out = []
    for p in raw or []:
        if isinstance(p, str) and p.strip():
            out.append({"p": p.strip(), "sub": []})
        elif isinstance(p, dict) and str(p.get("p", "")).strip():
            out.append({"p": str(p["p"]).strip(), "sub": [str(s).strip() for s in (p.get("sub") or []) if str(s).strip()]})
    return out


def _clean_sections(data) -> list[dict]:
    secs = []
    for s in (data or {}).get("sections") or []:
        pts = _clean_points(s.get("points"))
        if not pts:
            continue
        terms = [{"term": str(t.get("term", "")).strip(), "meaning": str(t.get("meaning", "")).strip()}
                 for t in (s.get("terms") or []) if isinstance(t, dict) and t.get("term") and t.get("meaning")]
        secs.append({"title": str(s.get("title") or "Notes").strip()[:90], "start": parse_ts(s.get("start")),
                     "points": pts, "terms": terms})
    return secs


def notes_for_window(w: dict, meta: dict) -> list[dict]:
    user = (f"Video title: {meta.get('title', '')}\nChannel: {meta.get('channel', '')}\n"
            f"{meta.get('lang_note', '')}\nTRANSCRIPT:\n{w['text']}")
    data = _json_call(SECTION_SYSTEM, user, max_tokens=4000)
    secs = _clean_sections(data)
    if secs:
        for s in secs:
            if not s["start"]:
                s["start"] = int(w["start"])
        return secs
    # model would not give JSON: keep a readable plain-text fallback instead of losing the window
    try:
        txt = _llm([{"role": "system", "content": "Write point-wise study notes (one '-' bullet per point, plain text) for this transcript. English."},
                    {"role": "user", "content": w["text"]}], 2500)
        pts = [{"p": re.sub(r"^[-•*\s]+", "", ln).strip(), "sub": []} for ln in txt.splitlines() if ln.strip()]
        return [{"title": "Notes from %s" % fmt_ts(w["start"]), "start": int(w["start"]), "points": pts, "terms": []}] if pts else []
    except Exception:
        return []


def _window_notes_safe(w: dict, meta: dict) -> tuple[list[dict], bool]:
    """Notes for one window; a failure is retried once and never takes the whole video down."""
    for attempt in range(2):
        try:
            secs = notes_for_window(w, meta)
            if secs:
                return secs, True
        except Exception as e:
            log.warning("notes window at %s failed (attempt %d): %s", fmt_ts(w["start"]), attempt + 1, e)
        time.sleep(2.0)
    return [], False


def build_notes(windows: list[dict], meta: dict, progress=None) -> dict:
    """progress(done, total) is called from worker threads as parts finish."""
    if len(windows) == 1:                       # short video: notes + summary + questions in ONE call
        user = (f"Video title: {meta.get('title', '')}\nChannel: {meta.get('channel', '')}\n"
                f"{meta.get('lang_note', '')}\nTRANSCRIPT:\n{windows[0]['text']}")
        try:
            one = _notes_from_json(_json_call(COMBINED_SYSTEM, user, max_tokens=6000) or {})
        except Exception as e:
            log.warning("single-call notes failed, using the normal path: %s", e)
            one = None
        if one:
            for s in one["sections"]:
                if not s["start"]:
                    s["start"] = int(windows[0]["start"])
            return one
    lock, done = threading.Lock(), [0]

    def work(w):
        secs, ok = _window_notes_safe(w, meta)
        with lock:
            done[0] += 1
            n = done[0]
        if progress:
            try:
                progress(n, len(windows))
            except Exception:
                pass
        return w, secs, ok
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(work, windows))
    gaps = [int(w["start"]) for w, secs, ok in results if not ok]
    sections = [s for _, secs, _ in results for s in secs]
    return _assemble(sections, gaps, meta)


def _insights(raw) -> list[dict]:
    return [{"t": str(i.get("t", "")).strip(), "d": str(i.get("d", "")).strip()}
            for i in (raw or []) if isinstance(i, dict) and i.get("t") and i.get("d")][:8]


def _assemble(sections: list[dict], gaps: list[int], meta: dict) -> dict:
    """Sections from any source -> final notes (one overview call: summary, takeaways, insights, questions)."""
    if not sections:
        raise NotesError("I couldn't generate notes for this video right now (the AI service is busy or unavailable). Please try again in a few minutes.")
    sections.sort(key=lambda s: s["start"])
    # the overview sees EVERY section, however long the video: lines shrink to fit a fixed budget
    budget = 14000
    per = max(0, min(400, budget // max(1, len(sections)) - 30))
    lines = []
    for s in sections:
        head = f"- [{fmt_ts(s['start'])}] {s['title']}"
        lines.append(head + (": " + " | ".join(p["p"] for p in s["points"][:3])[:per] if per >= 80 else ""))
    try:
        ov = _json_call(OVERVIEW_SYSTEM, f"Video: {meta.get('title', '')}\n\nSECTION NOTES:\n" + "\n".join(lines), max_tokens=3500) or {}
    except Exception as e:                       # the notes are the product; a missing front page is not fatal
        log.warning("overview call failed: %s", e)
        ov = {}
    glossary, seen = [], set()
    for s in sections:
        for t in s["terms"]:
            k = t["term"].lower()
            if k not in seen:
                seen.add(k); glossary.append(t)
    return {
        "subject": str(ov.get("subject") or "").strip(),
        "tldr": str(ov.get("tldr") or "").strip(),
        "takeaways": [str(x).strip() for x in (ov.get("takeaways") or []) if str(x).strip()][:8],
        "insights": _insights(ov.get("insights")),
        "questions": [{"q": str(q.get("q", "")).strip(), "a": str(q.get("a", "")).strip()}
                      for q in (ov.get("questions") or []) if isinstance(q, dict) and q.get("q") and q.get("a")][:6],
        "sections": sections,
        "glossary": glossary[:30],
        "gaps": gaps,
    }


# --------------------------------------------- fallback: Gemini watches the video
VIDEO_CHUNK_SECONDS = int(os.getenv("NOTES_VIDEO_CHUNK_SECONDS", "1200"))     # one Gemini call per 20 minutes of video
VIDEO_SYSTEM = """You turn a video clip into detailed study notes for a student who has NOT watched it.
Return ONLY one JSON object:
{"sections":[{"title":"...","start":"m:ss","points":["..." or {"p":"...","sub":["...","..."]}],"terms":[{"term":"...","meaning":"..."}]}]}
Rules:
- The clip is a slice of a longer video. "start" is the time where the topic begins, counted from the START OF THIS CLIP as m:ss (the clip's first second is 0:00).
- Cover the clip completely, from its first to its last minute: make one section per 3-5 minutes of the clip (a 20-minute clip needs 4-7 sections), in the order spoken. Never compress a long clip into a few sections.
- title: 3-8 words, specific. points: 5-9 per section, each a complete self-contained statement: the idea, the reason behind it, and any number, formula, step, example, name or code shown or said. Use "sub" bullets for steps and examples. Include what is written on slides or code on screen when it matters.
- Never invent facts. Skip greetings, sponsor reads, "like and subscribe" and filler. terms: 0-5 important technical terms with a one-sentence meaning.
- Write in clear English even if the speech is in another language. Plain text only: no markdown, no emojis.
- If the clip contains no speech or content, return {"sections":[]}."""


def _video_chunk(video_id: str, key: str, model: str, i: int) -> list[dict]:
    a, b = i * VIDEO_CHUNK_SECONDS, (i + 1) * VIDEO_CHUNK_SECONDS
    body = {"contents": [{"role": "user", "parts": [
        {"file_data": {"file_uri": f"https://www.youtube.com/watch?v={video_id}"},
         "video_metadata": {"start_offset": f"{a}s", "end_offset": f"{b}s"}},
        {"text": "Write the study notes for this clip as instructed."}]}],
        "systemInstruction": {"parts": [{"text": VIDEO_SYSTEM}]},
        "generationConfig": {"maxOutputTokens": 12000, "responseMimeType": "application/json",
                             "mediaResolution": "MEDIA_RESOLUTION_LOW"}}
    url = f"{chat_pdf._gemini_base()}/v1beta/models/{model}:generateContent"
    data = chat_pdf._post(url, {"x-goog-api-key": key, "Content-Type": "application/json"}, body, timeout=600, retries=1)
    try:
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"] if not p.get("thought"))
    except (KeyError, IndexError, TypeError):
        return []
    secs = _clean_sections(_parse_json(text) or {})
    for s in secs:
        # timestamps are relative to the clip; if the model answered with absolute times already, keep them
        s["start"] = s["start"] + a if (s["start"] < VIDEO_CHUNK_SECONDS or i == 0) else s["start"]
    return secs


def notes_from_video_url(video_id: str, meta: dict | None = None, progress=None) -> dict:
    """Gemini ingests the public YouTube URL itself (audio + visuals), so no transcript / IP block is involved.
    The video is read in 20-minute slices (so a 2-hour video gets as much detail as a short one); the end of the
    video is found when a slice comes back empty or past the end."""
    key = chat_pdf._gemini_key()
    if not key or _env("NOTES_GEMINI_VIDEO", "1") == "0":
        raise NotesError("no-video-fallback")
    model = chat_pdf._gemini_model()
    max_chunks = max(1, math.ceil(MAX_MINUTES * 60 / VIDEO_CHUNK_SECONDS))
    sections, gaps, i, ended = [], [], 0, False
    while i < max_chunks and not ended:
        batch = list(range(i, min(max_chunks, i + WORKERS)))

        def one(idx):
            for attempt in range(2):
                try:
                    return idx, _video_chunk(video_id, key, model, idx), None
                except Exception as e:
                    err = e
                    time.sleep(2.0)
            return idx, [], err
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(one, batch))
        for idx, secs, err in sorted(results, key=lambda r: r[0]):
            if ended:
                break
            if err is not None:
                if idx > 0 and "HTTP 400" in str(err):       # asked for minutes past the end of the video
                    ended = True
                else:
                    log.warning("video slice %d failed: %s", idx, err)
                    gaps.append(idx * VIDEO_CHUNK_SECONDS)
            elif not secs and idx > 0:
                ended = True
            else:
                sections += secs
        if progress:
            try:
                progress(len(sections))
            except Exception:
                pass
        i += len(batch)
    if not sections:
        raise NotesError("no-video-fallback")
    return _assemble(sections, gaps, meta or {"title": ""})


def _notes_from_json(j: dict) -> dict | None:
    """Sections + overview from ONE model reply (video fallback, and short transcripts)."""
    sections = _clean_sections(j)
    if not sections:
        return None
    gloss, seen = [], set()
    for s in sections:
        for t in s["terms"]:
            if t["term"].lower() not in seen:
                seen.add(t["term"].lower()); gloss.append(t)
    return {"subject": str(j.get("subject") or "").strip(), "tldr": str(j.get("tldr") or "").strip(),
            "takeaways": [str(x).strip() for x in (j.get("takeaways") or []) if str(x).strip()][:8],
            "insights": _insights(j.get("insights")),
            "questions": [{"q": str(q.get("q", "")).strip(), "a": str(q.get("a", "")).strip()}
                          for q in (j.get("questions") or []) if isinstance(q, dict) and q.get("q") and q.get("a")][:6],
            "sections": sorted(sections, key=lambda s: s["start"]), "glossary": gloss[:30]}


# ----------------------------------------------------------- WhatsApp text
def _wa(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)


def render_whatsapp(video_id: str, meta: dict, notes: dict, info: dict) -> list[str]:
    head = [f"🎬 *{meta['title']}*"]
    sub = []
    if meta.get("channel"):
        sub.append(f"👤 {meta['channel']}")
    if info.get("duration"):
        sub.append(f"⏱ {fmt_ts(info['duration'])}")
    if info.get("lang_name") and not _is_english(info.get("lang", "en")):
        sub.append(f"🌐 {info['lang_name']} → English")
    if sub:
        head.append(" • ".join(sub))
    blocks = ["\n".join(head)]
    if notes.get("tldr"):
        blocks.append("📝 *Summary*\n" + notes["tldr"])
    if notes.get("takeaways"):
        blocks.append("⭐ *Key takeaways*\n" + "\n".join(f"• {t}" for t in notes["takeaways"]))
    if notes.get("gaps"):
        blocks.append("⚠️ I couldn't process some parts of the video (around " + ", ".join(fmt_ts(g) for g in notes["gaps"][:8])
                      + "). Those minutes are missing from the notes - send the link again later to retry.")
    outline = "\n".join(f"{i}. [{fmt_ts(s['start'])}] {s['title']}" for i, s in enumerate(notes["sections"], 1))
    blocks.append("🗂 *Outline*\n" + outline)
    return _pack(blocks)


def render_full_text(notes: dict) -> list[str]:
    """Used only if the PDF cannot be produced: the complete notes as chat messages."""
    blocks = []
    for i, s in enumerate(notes["sections"], 1):
        lines = [f"*{i}. {s['title']}* ({fmt_ts(s['start'])})"]
        for p in s["points"]:
            lines.append(f"• {p['p']}")
            lines += [f"   – {x}" for x in p["sub"]]
        blocks.append("\n".join(lines))
    return _pack(blocks)


def _pack(blocks: list[str], limit: int = WA_LIMIT) -> list[str]:
    msgs, cur = [], ""
    for b in blocks:
        b = _wa(b)
        while len(b) > limit:                                    # an oversized block: cut at a line break
            cut = b.rfind("\n", 0, limit) or limit
            msgs_part, b = b[:cut], b[cut:].lstrip("\n")
            if cur:
                msgs.append(cur.strip()); cur = ""
            msgs.append(msgs_part)
        if len(cur) + len(b) + 2 > limit and cur:
            msgs.append(cur.strip()); cur = ""
        cur += b + "\n\n"
    if cur.strip():
        msgs.append(cur.strip())
    return msgs


# --------------------------------------------------------------------- PDF
_FONT = {"name": None, "bold": None}


def _fonts():
    """DejaVu Sans if the system has it (covers arrows, quotes, maths); otherwise Helvetica + ASCII clean-up."""
    if _FONT["name"]:
        return _FONT
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        for reg, bold in (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                          ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
                          ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")):
            if os.path.exists(reg) and os.path.exists(bold):
                pdfmetrics.registerFont(TTFont("NotesSans", reg))
                pdfmetrics.registerFont(TTFont("NotesSans-Bold", bold))
                pdfmetrics.registerFontFamily("NotesSans", normal="NotesSans", bold="NotesSans-Bold",
                                              italic="NotesSans", boldItalic="NotesSans-Bold")
                _FONT.update(name="NotesSans", bold="NotesSans-Bold", unicode=True)
                return _FONT
    except Exception as e:
        log.info("DejaVu font not usable: %s", e)
    _FONT.update(name="Helvetica", bold="Helvetica-Bold", unicode=False)
    return _FONT


_ASCII = {"•": "-", "–": "-", "—": "-", "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "×": "x", "÷": "/",
          "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "✓": "v", " ": " "}


def _pdf_text(t: str) -> str:
    from xml.sax.saxutils import escape
    t = str(t or "")
    if _fonts().get("unicode"):                       # drop glyphs the font lacks (e.g. Hindi in a title) instead of printing boxes
        try:
            from reportlab.pdfbase import pdfmetrics
            cmap = pdfmetrics.getFont("NotesSans").face.charToGlyph
            t = "".join(ch for ch in t if ord(ch) in cmap or ch in "\n\t")
            t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
            t = re.sub(r"\s{2,}", " ", t).strip()
        except Exception:
            pass
    else:
        for k, v in _ASCII.items():
            t = t.replace(k, v)
        t = t.encode("latin-1", "replace").decode("latin-1")
    return escape(t)


def build_pdf(video_id: str, meta: dict, notes: dict, info: dict, thumb: bytes | None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    F = _fonts()
    ink, muted, accent, soft = colors.HexColor("#1f2937"), colors.HexColor("#6b7280"), colors.HexColor("#2f4fd8"), colors.HexColor("#f1f5f9")
    st = lambda name, **kw: ParagraphStyle(name, fontName=kw.pop("font", F["name"]), textColor=kw.pop("color", ink), alignment=TA_LEFT, **kw)
    S = {
        "title": st("title", font=F["bold"], fontSize=19, leading=24, spaceAfter=4),
        "meta": st("meta", fontSize=9, leading=13, color=muted),
        "h1": st("h1", font=F["bold"], fontSize=12.5, leading=16, color=accent, spaceBefore=14, spaceAfter=6),
        "h2": st("h2", font=F["bold"], fontSize=12, leading=16, spaceBefore=12, spaceAfter=4),
        "body": st("body", fontSize=10, leading=14.5, spaceAfter=3),
        "bullet": st("bullet", fontSize=10, leading=14.5, leftIndent=14, bulletIndent=3, spaceAfter=3),
        "sub": st("sub", fontSize=9.5, leading=13.5, leftIndent=30, bulletIndent=19, spaceAfter=2, color=colors.HexColor("#374151")),
        "term": st("term", fontSize=9.5, leading=13),
        "q": st("q", font=F["bold"], fontSize=10, leading=14, spaceBefore=6),
        "a": st("a", fontSize=9.5, leading=13.5, leftIndent=12, color=colors.HexColor("#374151")),
    }
    P = lambda text, style: Paragraph(_pdf_text(text), S[style])
    bullet = lambda text, style="bullet": Paragraph(_pdf_text(text), S[style], bulletText="•" if F.get("unicode") else "-")
    ts_link = lambda sec: (f'<a href="{link_at(video_id, sec)}" color="#2f4fd8">[{fmt_ts(sec)}]</a>')

    story = []
    title_cell = [P(meta["title"], "title")]
    bits = [b for b in (meta.get("channel"), fmt_ts(info["duration"]) if info.get("duration") else "",
                        f"{info['lang_name']} -> English" if info.get("lang_name") and not _is_english(info.get("lang", "en")) else "",
                        notes.get("subject")) if b]
    if bits:
        title_cell.append(P("  |  ".join(bits), "meta"))
    title_cell.append(Paragraph(f'<a href="{link_at(video_id)}" color="#2f4fd8">{link_at(video_id)}</a>', S["meta"]))
    if thumb:
        try:
            from PIL import Image as PILImage
            w, h = PILImage.open(io.BytesIO(thumb)).size
            iw = 62 * mm
            img = Image(io.BytesIO(thumb), width=iw, height=iw * h / w)
            t = Table([[img, title_cell]], colWidths=[66 * mm, None])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(t)
        except Exception as e:
            log.info("thumbnail skipped: %s", e)
            story += title_cell
    else:
        story += title_cell
    story.append(Spacer(1, 6))
    if notes.get("gaps"):
        story.append(P("Note: some parts of the video could not be processed (around " +
                       ", ".join(fmt_ts(g) for g in notes["gaps"][:8]) + ") and are missing from these notes.", "meta"))

    def box(flow_items):
        t = Table([[flow_items]], colWidths=[None])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), soft), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return t
    if notes.get("tldr"):
        story += [P("Summary", "h1"), box([P(notes["tldr"], "body")])]
    if notes.get("takeaways"):
        story += [P("Key takeaways", "h1"), box([bullet(t) for t in notes["takeaways"]])]
    if notes.get("insights"):
        story.append(P("Key insights", "h1"))
        for ins in notes["insights"]:
            story.append(KeepTogether([P(ins["t"], "q"), P(ins["d"], "a")]))
    story.append(P("Detailed notes", "h1"))
    for i, s in enumerate(notes["sections"], 1):
        heading = _pdf_text("%d. %s" % (i, s["title"]))
        head = Paragraph(ts_link(s["start"]) + " " + heading, S["h2"])
        items = [head]
        for p in s["points"]:
            items.append(bullet(p["p"]))
            items += [bullet(x, "sub") for x in p["sub"]]
        if s["terms"]:
            rows = [[P(t["term"], "term"), P(t["meaning"], "term")] for t in s["terms"]]
            tb = Table(rows, colWidths=[42 * mm, None])
            tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), soft), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
                                    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            items += [Spacer(1, 3), tb]
        story.append(KeepTogether(items[:4]))                       # keep heading with its first points
        story += items[4:]
    if notes.get("glossary"):
        story.append(P("Glossary", "h1"))
        rows = [[P(t["term"], "term"), P(t["meaning"], "term")] for t in notes["glossary"]]
        tb = Table(rows, colWidths=[46 * mm, None], repeatRows=0)
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), soft), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(tb)
    if notes.get("questions"):
        story.append(P("Test yourself", "h1"))
        for n, q in enumerate(notes["questions"], 1):
            story.append(KeepTogether([P(f"Q{n}. {q['q']}", "q"), P(f"Answer: {q['a']}", "a")]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(F["name"], 8)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 10 * mm, "Notes generated by CareerMate from the video transcript. Check important details against the video.")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title=meta["title"][:100], author="CareerMate")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _safe_filename(title: str) -> str:
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", title).strip().replace(" ", "_")[:50] or "youtube"
    return f"Notes_{base}.pdf"


# ---------------------------------------------------- WhatsApp media sending
def _graph() -> tuple[str, dict, str]:
    base = _env("WHATSAPP_GRAPH_BASE", "https://graph.facebook.com/v21.0").rstrip("/")
    token = _env("WHATSAPP_ACCESS_TOKEN") or _env("WHATSAPP_TOKEN")
    pid = _env("WHATSAPP_PHONE_NUMBER_ID") or _env("PHONE_NUMBER_ID")
    if not token or not pid:
        raise NotesError("WhatsApp credentials are missing on the server.")
    return base, {"Authorization": f"Bearer {token}"}, pid


def _send_media_sync(phone: str, kind: str, data: bytes, mime: str, filename: str, caption: str = "") -> None:
    base, headers, pid = _graph()
    up = requests.post(f"{base}/{pid}/media", headers=headers, data={"messaging_product": "whatsapp", "type": mime},
                       files={"file": (filename, data, mime)}, timeout=90)
    if up.status_code >= 400:
        raise NotesError(f"media upload failed: HTTP {up.status_code} {up.text[:150]}")
    media_id = up.json().get("id")
    payload = {"messaging_product": "whatsapp", "to": phone, "type": kind,
               kind: {"id": media_id, **({"filename": filename} if kind == "document" else {}),
                      **({"caption": caption[:1000]} if caption else {})}}
    r = requests.post(f"{base}/{pid}/messages", headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=30)
    if r.status_code >= 400:
        raise NotesError(f"media send failed: HTTP {r.status_code} {r.text[:150]}")


async def send_pdf(phone: str, pdf: bytes, filename: str, caption: str = "") -> None:
    await asyncio.to_thread(_send_media_sync, phone, "document", pdf, "application/pdf", filename, caption)


async def send_image(phone: str, jpg: bytes, caption: str = "") -> None:
    await asyncio.to_thread(_send_media_sync, phone, "image", jpg, "image/jpeg", "thumbnail.jpg", caption)


# ------------------------------------------------------------ limits / cache
_USAGE: dict[str, tuple[str, int]] = {}
_CACHE: dict[str, tuple[float, dict]] = {}


def _quota_ok(phone: str) -> bool:
    today = date.today().isoformat()
    d, n = _USAGE.get(phone, (today, 0))
    if d != today:
        n = 0
    if n >= MAX_PER_DAY:
        return False
    _USAGE[phone] = (today, n + 1)
    return True


def _refund(phone: str) -> None:
    d, n = _USAGE.get(phone, (date.today().isoformat(), 0))
    _USAGE[phone] = (d, max(0, n - 1))


def _log(phone: str, ok: bool = True, **meta) -> None:
    try:
        import analytics
        analytics.log_event(phone, "ai_assistance", meta={"feature": "notes_utube", **meta}, success=ok)
    except Exception:
        pass


# --------------------------------------------------------------- public API
def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


async def start(phone: str, arg: str, say) -> bool:
    """/utube_notes [link]. Returns True if a link was given and the job finished (state cleared)."""
    await asyncio.to_thread(db.save_state, phone, {"mode": MODE})
    if arg and extract_video_id(arg):
        return await handle(phone, arg, say)
    await say(PROMPT)
    return False


async def handle(phone: str, text: str, say) -> bool:
    """A message while in notes mode. Returns True when the job is over (success or failure) and the state is cleared."""
    text = (text or "").strip()
    if text.lower() in EXIT_WORDS:
        await asyncio.to_thread(db.save_state, phone, {})
        await say("👋 Cancelled.")
        return True
    vid = extract_video_id(text)
    if not vid:
        await say("🔗 That doesn't look like a YouTube link. Send a link like https://youtu.be/xxxxxxxxxxx (or *exit* to cancel).")
        return False
    if not _quota_ok(phone):
        await asyncio.to_thread(db.save_state, phone, {})
        await say(f"⏳ You've reached today's limit of {MAX_PER_DAY} videos. Please try again tomorrow.")
        return True
    t0 = time.perf_counter()
    ctx: dict = {}                                            # filled by _run (title, duration) for the admin analytics page
    ms = lambda: int((time.perf_counter() - t0) * 1000)
    try:
        await _run(phone, vid, say, ctx)
        _log(phone, video=vid, ms=ms(), **ctx)
    except NotesError as e:
        _refund(phone)
        _log(phone, ok=False, video=vid, ms=ms(), message=str(e)[:200], **ctx)
        await say("❌ " + str(e))
    except Exception as e:
        _refund(phone)
        log.exception("notes_utube failed for %s", vid)
        _log(phone, ok=False, video=vid, ms=ms(), message=str(e)[:200], **ctx)
        await say("⚠️ Something went wrong while making the notes. Please try again in a few minutes.")
    await asyncio.to_thread(db.save_state, phone, {})
    return True


async def _run(phone: str, vid: str, say, ctx: dict | None = None) -> None:
    ctx = ctx if ctx is not None else {}
    cached = _CACHE.get(vid)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        meta, notes, info = cached[1]["meta"], cached[1]["notes"], cached[1]["info"]
        ctx.update(title=meta.get("title", "")[:150], channel=meta.get("channel", "")[:80], duration=int(info.get("duration") or 0), cached=True)
        await say("⚡ I already made notes for this video - sending them now.")
    else:
        await say("⏳ Got it! Fetching the video transcript...")
        meta = await asyncio.to_thread(fetch_meta, vid)
        ctx.update(title=meta.get("title", "")[:150], channel=meta.get("channel", "")[:80])
        info, notes = {}, None
        try:
            mode = translate_mode()
            tr = await asyncio.to_thread(fetch_transcript, vid, mode == "youtube")
            info = {"lang": tr["lang"], "lang_name": tr["lang_name"], "duration": tr["duration"], "generated": tr["generated"]}
            ctx["duration"] = int(tr["duration"] or 0)
            if tr["duration"] > MAX_MINUTES * 60:
                raise NotesError(f"That video is {fmt_ts(tr['duration'])} long. I can handle videos up to {MAX_MINUTES // 60}h{MAX_MINUTES % 60:02d}m.")
            windows = build_windows(tr["snippets"])
            if not _is_english(tr["lang"]):
                if tr.get("translated"):            # YouTube already gave English captions: no AI translation needed
                    meta = {**meta, "lang_note": f"(The transcript was machine-translated from {tr['lang_name']} by YouTube; "
                                                 "fix obvious translation errors from context.)\n"}
                    await say(f"🌐 The video is in *{tr['lang_name']}*. Using YouTube's English translation and writing the notes (this can take a minute or two)...")
                elif mode == "llm":
                    await say(f"🌐 The video is in *{tr['lang_name']}*. Translating to English, then writing the notes (this can take a minute or two)...")
                    windows, failed = await asyncio.to_thread(translate_windows, windows, tr["lang_name"])
                    if failed:
                        log.warning("%d/%d windows left untranslated", failed, len(windows))
                else:                               # the notes model translates while it writes (no extra calls)
                    meta = {**meta, "lang_note": f"(The transcript is in {tr['lang_name']}: write the notes in English.)\n"}
                    await say(f"🌐 The video is in *{tr['lang_name']}*. Writing the notes in English (this can take a minute or two)...")
            else:
                await say(f"✍️ Writing point-wise notes for *{meta['title'][:80]}* ({fmt_ts(tr['duration'])}). This can take a minute or two...")
            if len(windows) >= 5:
                est = max(2, round(len(windows) / WORKERS * 0.8))
                await say(f"⏱ This is a long video ({fmt_ts(tr['duration'])}), so I'm processing it in {len(windows)} parts - roughly {est}-{est * 2} minutes. I'll keep you posted.")
            loop = asyncio.get_running_loop()
            marks = {math.ceil(len(windows) * f) for f in (0.25, 0.5, 0.75)} if len(windows) >= 4 else set()

            def progress(n, total):
                if n in marks:
                    asyncio.run_coroutine_threadsafe(say(f"📚 Notes progress: {n}/{total} parts done..."), loop)
            notes = await asyncio.to_thread(build_notes, windows, meta, progress)
        except TranscriptUnavailable as e:
            log.warning("transcript unavailable for %s: %s", vid, e)
            reason = {"blocked": "YouTube blocked the transcript request from my server",
                      "none": "this video has no captions", "unavailable": "this video is private, age-restricted or unavailable"}.get(e.kind, "I couldn't read its transcript")
            if e.kind == "unavailable":
                raise NotesError("I can't access that video - it may be private, age-restricted or removed.")
            await say(f"ℹ️ {reason[0].upper() + reason[1:]}, so I'll try listening to the video itself. This takes a bit longer...")
            try:
                notes = await asyncio.to_thread(notes_from_video_url, vid, meta)
                info = {}
            except NotesError:
                raise NotesError(f"I couldn't get the transcript ({reason}). Try a video that has captions, or try again later.")
            except Exception as e2:
                log.warning("video fallback failed: %s", e2)
                raise NotesError(f"I couldn't get the transcript ({reason}) and reading the video directly failed. Please try another video.")
        _CACHE[vid] = (time.time(), {"meta": meta, "notes": notes, "info": info})
        if len(_CACHE) > 60:
            _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]), None)

    thumb = await asyncio.to_thread(fetch_thumbnail, vid)
    if thumb:
        try:
            await send_image(phone, thumb, f"🎬 {meta['title']}"[:900])
        except Exception as e:
            log.info("thumbnail not sent: %s", e)
    for m in render_whatsapp(vid, meta, notes, info):
        await say(m)
    pdf, why = None, ""
    try:
        pdf = await asyncio.to_thread(build_pdf, vid, meta, notes, info, thumb)
    except Exception as e:
        log.exception("PDF BUILD failed (is reportlab/pillow installed?): %s", e)
        why = "PDF could not be built"
    if pdf:
        try:
            await send_pdf(phone, pdf, _safe_filename(meta["title"]), "📄 Full notes (tap the [timestamps] to jump to that part of the video)")
        except Exception as e:
            log.exception("PDF SEND failed (WhatsApp media upload): %s", e)
            pdf, why = None, "PDF could not be uploaded to WhatsApp"
    if not pdf:
        await say(f"📄 ({why}) - here are the full notes instead:")
        for m in render_full_text(notes):
            await say(m)
    await say("✅ Done! Send */utube_notes* for another video, or */menu* for the menu.")


def selftest() -> None:  # pragma: no cover
    """python notes_utube.py selftest  -> checks everything the PDF + WhatsApp upload need, on THIS server."""
    import importlib
    ok = True
    for m in ("youtube_transcript_api", "reportlab", "PIL", "requests"):
        try:
            importlib.import_module(m); print("OK      module", m)
        except Exception as e:
            ok = False; print("MISSING module", m, "->", e, " (pip install youtube-transcript-api reportlab pillow requests)")
    print("font    ", _fonts()["name"], "(DejaVu = full unicode, Helvetica = ASCII only)")
    try:
        sec = [{"title": "Test", "start": 0, "points": [{"p": "A point", "sub": ["a sub point"]}], "terms": []}]
        pdf = build_pdf("dQw4w9WgXcQ", {"title": "Self test", "channel": "me"}, {"sections": sec, "tldr": "x", "takeaways": ["y"],
                        "insights": [{"t": "t", "d": "d"}], "questions": [], "glossary": []}, {}, None)
        print("OK      PDF built,", len(pdf), "bytes")
    except Exception as e:
        ok = False; print("FAILED  PDF build:", repr(e))
    try:
        base, headers, pid = _graph()
        r = requests.get(f"{base}/{pid}", headers=headers, params={"fields": "display_phone_number,verified_name"}, timeout=20)
        print("OK     " if r.status_code == 200 else "FAILED ", "WhatsApp credentials: HTTP", r.status_code, r.text[:160])
        ok = ok and r.status_code == 200
    except Exception as e:
        ok = False; print("FAILED  WhatsApp credentials:", repr(e))
    print("Gemini key:", "set" if chat_pdf._gemini_key() else "NOT SET (the 'watch the video' fallback needs it)")
    print("ALL GOOD" if ok else "Fix the lines marked MISSING/FAILED above.")


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest(); raise SystemExit(0)
    if len(sys.argv) < 2:
        print("usage: python notes_utube.py <youtube link>   (writes notes.pdf; needs network + AI keys)")
        raise SystemExit(1)
    v = extract_video_id(sys.argv[1]) or sys.argv[1]
    m = fetch_meta(v); tr = fetch_transcript(v, translate_mode() == "youtube")
    w = build_windows(tr["snippets"])
    if not _is_english(tr["lang"]) and not tr.get("translated") and translate_mode() == "llm":
        w, _ = translate_windows(w, tr["lang_name"])
    n = build_notes(w, m)
    open("notes.pdf", "wb").write(build_pdf(v, m, n, {"lang": tr["lang"], "lang_name": tr["lang_name"], "duration": tr["duration"]}, fetch_thumbnail(v)))
    print("wrote notes.pdf with", len(n["sections"]), "sections")
