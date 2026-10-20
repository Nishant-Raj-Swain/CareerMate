
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
import zipfile
from datetime import date
from urllib.parse import quote, quote_plus, urljoin, urlparse

import httpx
import requests

log = logging.getLogger("pyq")

# =====================================================================
#  CONFIG
# =====================================================================
MODE = "pyq"
UA = os.getenv("PYQ_UA") or "CareerMateBot/1.0 (educational WhatsApp bot; previous year question papers)"
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
MAX_MB = float(os.getenv("PYQ_MAX_MB", "45"))                    # larger files -> send link instead
MAX_PER_DAY = int(os.getenv("PYQ_MAX_PER_DAY", "10"))
TIMEOUT = float(os.getenv("PYQ_TIMEOUT", "20"))
EXIT_WORDS = {"exit", "cancel", "stop", "quit", "/exit", "/cancel"}
# NOTE: "back" is NOT in EXIT_WORDS — it's handled per-step for navigation.

HUB_URL = "https://www.pyqonline.com/"
PAGE_SIZE = 30                                                    # 30 items per WhatsApp message (WhatsApp-safe)


class PyqError(Exception):
    pass


# =====================================================================
#  STATE PERSISTENCE  (same shape as books.py)
# =====================================================================
def _save(phone: str, state: dict) -> None:
    """Persist conversation state. Falls back to in-memory dict if the
    `database` module isn't available (e.g. when running standalone in Colab)."""
    try:
        import database as db
        db.save_state(phone, state)
    except Exception:
        _FALLBACK_STATE[phone] = state


def _load(phone: str) -> dict:
    try:
        import database as db
        return (db.get_user(phone) or {}).get("state") or {}
    except Exception:
        return _FALLBACK_STATE.get(phone, {})


_FALLBACK_STATE: dict[str, dict] = {}


# =====================================================================
#  QUOTA / LOGGING  (same shape as books.py)
# =====================================================================
_USAGE: dict[str, tuple[str, int]] = {}


def _quota_ok(phone: str) -> bool:
    today = date.today().isoformat()
    d, n = _USAGE.get(phone, (today, 0))
    n = 0 if d != today else n
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
        analytics.log_event(phone, "pyq", meta=meta, success=ok)
    except Exception:
        pass


# =====================================================================
#  WHATSAPP CLOUD API — same shape as books.py
# =====================================================================
def _graph():
    base = (os.getenv("WHATSAPP_GRAPH_BASE") or "https://graph.facebook.com/v21.0").rstrip("/")
    token = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
    pid = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID")
    if not token or not pid:
        raise PyqError("WhatsApp credentials are missing on the server.")
    return base, {"Authorization": f"Bearer {token}"}, pid


def _send_document_sync(phone: str, data: bytes, mime: str, filename: str, caption: str) -> None:
    base, headers, pid = _graph()
    up = requests.post(f"{base}/{pid}/media", headers=headers,
                       data={"messaging_product": "whatsapp", "type": mime},
                       files={"file": (filename, data, mime)}, timeout=120)
    if up.status_code >= 400:
        raise PyqError(f"media upload failed: HTTP {up.status_code} {up.text[:150]}")
    r = requests.post(f"{base}/{pid}/messages",
                      headers={**headers, "Content-Type": "application/json"}, timeout=30,
                      json={"messaging_product": "whatsapp", "to": phone, "type": "document",
                            "document": {"id": up.json().get("id"),
                                         "filename": filename, "caption": caption[:1000]}})
    if r.status_code >= 400:
        raise PyqError(f"media send failed: HTTP {r.status_code} {r.text[:150]}")


async def _send_file(phone: str, name: str, data: bytes, link: str, say,
                     caption: str = "") -> bool:
    """Try to send PDF bytes via WhatsApp; on failure, fall back to the link."""
    try:
        await asyncio.to_thread(_send_document_sync, phone, data,
                                "application/pdf",
                                _safe_filename(name) + ".pdf",
                                caption)
        return True
    except Exception as e:
        log.exception("document send failed: %s", e)
        await say(f"📎 *{name}*\n\nI couldn't attach the file, but you can download it here:\n{link}")
        return True


# =====================================================================
#  HELPERS
# =====================================================================
def _safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().replace(" ", "_")[:80] or "paper"
    return base


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# =====================================================================
#  STEP 1 — University list from the source site's sidebar
# =====================================================================
_UNI_CACHE: dict = {"at": 0.0, "items": []}
_UNI_TTL = 86400  # 1 day


async def _fetch_university_list(client: httpx.AsyncClient) -> list[dict]:
    """The source homepage embeds a JS array of {url, description} pairs.
    That array IS the left sidebar. Parse it once a day and cache."""
    if _UNI_CACHE["items"] and time.time() - _UNI_CACHE["at"] < _UNI_TTL:
        return _UNI_CACHE["items"]
    r = await client.get(HUB_URL, timeout=TIMEOUT)
    r.raise_for_status()
    pattern = re.compile(
        r'\{\s*url:\s*"([^"]+)"\s*,\s*description:\s*"([^"]+)"\s*\}',
        re.IGNORECASE,
    )
    items = [{"url": u, "description": d} for u, d in pattern.findall(r.text)]
    items = [it for it in items
             if any(k in it["description"].upper()
                    for k in ("UNIVERSITY", "INSTITUTE", "COLLEGE", "VIDYAPEETH"))]
    _UNI_CACHE["at"] = time.time()
    _UNI_CACHE["items"] = items
    log.info("pyq: cached %d universities", len(items))
    return items


def _search_universities(items: list[dict], query: str, limit: int = 30) -> list[dict]:
    """Fuzzy rank by token overlap. Strict (AND) hits bubble to the top."""
    q = _norm(query)
    if not q:
        return items[:limit]
    qtokens = q.split()

    def score(it: dict) -> int:
        d = _norm(it["description"])
        if q in d:
            return 100 + sum(d.count(t) for t in qtokens)
        return sum(1 for t in qtokens if t in d)

    ranked = sorted(items, key=score, reverse=True)
    strict = [it for it in ranked if all(t in _norm(it["description"]) for t in qtokens)]
    return (strict or ranked)[:limit]


# =====================================================================
#  STEP 2 — Course list on a university's sister-site
# =====================================================================
async def _list_courses(client: httpx.AsyncClient, university_url: str) -> list[tuple[str, str]]:
    """Detect the course nav-bar (B.Tech / BCA / MCA / MBA ...).
    Branch-level links (ending in -question-papers.html) are filtered OUT here."""
    r = await client.get(university_url, timeout=TIMEOUT)
    r.raise_for_status()
    # Lazy import: keeps the module importable even if bs4 is missing at module load
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")

    courses, seen = [], set()
    kw = re.compile(
        r'\b(B\.?Tech|M\.?Tech|B\.?Arch|B\.?Plan|B\.?Pharm|M\.?Pharm|'
        r'B\.?A|M\.?A|B\.?Sc|M\.?Sc|B\.?Com|M\.?Com|BCA|MCA|MBA|BBA|'
        r'BHMCT|MAM|Ph\.?D|Diploma)\b', re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if not text or len(text) > 25:
            continue
        if not kw.search(text):
            continue
        href = a["href"].split("#")[0]
        if "question-papers.html" in href:
            continue
        full = urljoin(university_url, href)
        if full == university_url or full in seen:
            continue
        seen.add(full)
        courses.append((text, full))

    if not courses:                                  # fallback: <select> switcher
        for sel in soup.find_all("select"):
            for opt in sel.find_all("option"):
                v = opt.get("value", "").strip()
                if v and v != "#" and kw.search(opt.text):
                    courses.append((opt.text.strip(), urljoin(university_url, v)))

    # de-dup preserving order
    uniq, seen2 = [], set()
    for name, url in courses:
        key = name.lower()
        if key in seen2:
            continue
        seen2.add(key)
        uniq.append((name, url))
    return uniq


# =====================================================================
#  STEP 2b — Branch list (CSE / IT / EEE / Mechanical ...)
# =====================================================================
async def _list_branches(client: httpx.AsyncClient, course_url: str) -> list[tuple[str, str]]:
    """Look for branch-specific sub-pages. Returns [] if course has no branches."""
    try:
        r = await client.get(course_url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception:
        return []
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")

    parsed = urlparse(course_url)
    path = parsed.path.lower()
    m = re.search(r'/([a-z][a-z0-9\-]+?)(?:-question-papers\.html|\.html)?/?$', path)
    course_slug = m.group(1) if m else ""
    course_slug = re.sub(r'-question-papers$', '', course_slug)
    course_prefix = course_slug.split("-")[0] if course_slug else ""

    branches, seen_urls, first_year_added = [], set(), False
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        if not text or len(text) > 35:
            continue

        icon = a.find("i")
        icon_classes = icon.get("class", []) if icon else []
        is_first_year = ("fa-star" in icon_classes
                         and "first-year" in href.lower())
        if is_first_year and not first_year_added:
            branches.insert(0, (text, course_url))
            first_year_added = True
            continue

        if "-question-papers.html" in href:
            full_url = urljoin(course_url, href.split("#")[0])
            if full_url in seen_urls or full_url == course_url:
                continue
            if course_prefix:
                bp = urlparse(full_url).path.lower()
                if not bp.startswith(f"/{course_prefix}-"):
                    continue
            seen_urls.add(full_url)
            branches.append((text, full_url))

    # If URL had no prefix to infer, derive it from the first branch link
    if not course_prefix and branches:
        first_path = urlparse(branches[0][1]).path.lower()
        m2 = re.match(r'/([a-z]+)-', first_path)
        if m2:
            cp = m2.group(1)
            branches = [(n, u) for n, u in branches
                        if urlparse(u).path.lower().startswith(f"/{cp}-")]
    return branches


# =====================================================================
#  STEP 3 — Paper list (filtered by semester + year)
# =====================================================================
async def _fetch_papers(client: httpx.AsyncClient, branch_url: str,
                        semester: int | None = None, year: int | None = None) -> list[dict]:
    """Scrape every paper link from the branch page; filter by sem/year if given."""
    r = await client.get(branch_url, timeout=TIMEOUT)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")

    papers = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/papers/" not in href or not href.endswith(".html"):
            continue
        page_url = urljoin(branch_url, href)
        name = os.path.basename(href)[:-5]            # strip ".html"

        m = re.search(r'(\d+)-sem.*?-(\d{4})\.html$', href, re.IGNORECASE)
        sem = int(m.group(1)) if m else None
        yr = int(m.group(2)) if m else None

        if semester is not None and sem != semester:
            continue
        if year is not None and yr != year:
            continue

        papers.append({
            "name": name, "sem": sem, "year": yr,
            "page_url": page_url, "pdf_url": page_url[:-5] + ".pdf",
        })

    seen, uniq = set(), []
    for p in papers:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        uniq.append(p)
    return uniq


# =====================================================================
#  STEP 4 — Download a paper's PDF bytes
# =====================================================================
async def _download_paper_bytes(client: httpx.AsyncClient, paper: dict) -> tuple[bytes | None, str]:
    """Returns (pdf_bytes or None, reason). Reason is 'ok' / 'too big' / 'not pdf' / error msg."""
    cap = int(MAX_MB * 1024 * 1024)
    url = paper["pdf_url"]
    try:
        r = await client.get(url, timeout=TIMEOUT, follow_redirects=True)
        if r.status_code != 200:
            # Fallback: scrape the .html page for the real download CTA
            try:
                r2 = await client.get(paper["page_url"], timeout=TIMEOUT, follow_redirects=True)
                from bs4 import BeautifulSoup
                sp = BeautifulSoup(r2.text, "lxml")
                cta = sp.select_one("a.download-cta, a[href$='.pdf']")
                if cta and cta.get("href"):
                    real_url = urljoin(paper["page_url"], cta["href"])
                    r = await client.get(real_url, timeout=TIMEOUT, follow_redirects=True)
                    url = real_url
            except Exception:
                pass

        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        data = r.content
        if len(data) > cap:
            return None, "too big"
        if data[:4] != b"%PDF":
            return None, "not a PDF"
        return data, "ok"
    except Exception as e:
        return None, str(e)[:80]


# =====================================================================
#  CONVERSATION PROMPTS  (mirror books.py style)
# =====================================================================
PROMPT = ("📝 *Previous Year Questions*\n\nChoose a section:\n"
          "*1.* University PYQ  (University → Course → Branch → Sem/Year)\n"
          "*2.* Company-wise PYQ  (TCS, Infosys, Wipro, Amazon ... placement papers)\n"
          "*3.* Common Interview Questions  (DSA, DBMS, OS, Networks, HR ...)\n\n"
          "Reply with *1*, *2*, or *3* (or send *exit* to cancel).")

_UNI_SEARCH_PROMPT = ("📝 *Universities*\n\n"
                     "Type your *university name* (e.g. *BPUT*, *AKTU*, *BHU*, *Gauhati*) "
                     "and I'll find the closest matches.\n\n"
                     "Or send *all* to browse the full list, *back* for menu, *exit* to cancel.")
_UNI_RESULTS_PROMPT = ("📝 *Universities*\n\n"
                       "Did you mean:\n\n")
_UNI_PROMPT = _UNI_SEARCH_PROMPT  # alias for backward compat
_COURSE_PROMPT = ("📝 *{uni}*\n\n"
                  "Send the *number* of a course (or *next* / *prev* / *back* / *exit*):\n\n")
_BRANCH_PROMPT = ("📝 *{uni} - {course}*\n\n"
                  "Send the *number* of a branch (or *next* / *prev* / *back* / *exit*):\n\n")
_FILTER_PROMPT = ("📝 *{uni} - {course}{branch}*\n\n"
                  "Reply with:\n"
                  "  • Semester number (e.g. *1*, *3*, *6*) — or *all*\n"
                  "  • Then year (e.g. *2025*, *2026*) — or *all*\n\n"
                  "Or *back* / *exit*.\n"
                  "_Send them in one message separated by space, e.g. `6 2025`_")
_PAPERS_PROMPT = ("📝 Found *{n}* papers. Reply with:\n"
                  "  • A *number* to download that paper\n"
                  "  • *all* to download every matched paper\n"
                  "  • *next* / *prev* for more pages\n"
                  "  • *back* to change semester/year, *exit* to cancel\n\n")

_COMPANY_PROMPT = ("🏢 *Company-wise PYQ*\n\n"
                   "Type a *company name* (e.g. TCS, Infosys, Wipro) — or *back* / *exit*:\n\n")
_COMPANY_PAPERS_PROMPT = ("🏢 *{company}* — Found *{n}* paper(s)\n\n"
                          "Reply with:\n"
                          "  • A *number* to download that paper as PDF\n"
                          "  • *all* to download every paper\n"
                          "  • *next* / *prev* for more pages\n"
                          "  • *back* to type another company, *exit*\n\n")
_INTERVIEW_PROMPT = ("💬 *Common Interview Questions*\n\n"
                     "Pick a section:\n"
                     "*1.* HR Interview\n"
                     "*2.* Group Discussion (GD)\n"
                     "*3.* Technical Interview\n\n"
                     "Reply with *1*, *2*, or *3* (or *back* / *exit*).\n")
_HR_PROMPT = ("👤 *HR Interview*\n\n"
              "Send the *number* of a question (or *next* / *prev* / *back* / *exit*):\n\n")
_GD_PROMPT = ("👥 *Group Discussion Topics*\n\n"
              "Send the *number* of a topic (or *next* / *prev* / *back* / *exit*):\n\n")
_TECH_PROMPT = ("💻 *Technical Interview*\n\n"
                "Send the *number* of a topic (or *next* / *prev* / *back* / *exit*):\n\n")


# =====================================================================
#  SECTION 2 — COMPANY-WISE PYQ  (scraped from the source site)
#
#  Flow:
#    1. User types a company name (e.g. "TCS", "Infosys", "Wipro")
#    2. We fetch https://www.indiabix.com/placement-papers/<slug>/ which
#       returns a paginated list of interview-experience papers.
#    3. Each paper has a title, posted-by, and a link to a detail page with
#       the full interview experience text.
#    4. User picks a paper (or 'all') -> we fetch each paper's content,
#       bundle them into a single PDF via reportlab, send via WhatsApp.
# =====================================================================

INDIABIX_BASE = "https://www.indiabix.com"


async def _indiabix_search_companies(client: httpx.AsyncClient,
                                      query: str) -> list[tuple[str, str]]:
    """Fetch the placement-papers index and return companies that
    fuzzy-match the user's query. Returns [(company_name, slug), ...]."""
    r = await client.get(f"{INDIABIX_BASE}/placement-papers/", timeout=TIMEOUT)
    r.raise_for_status()
    # Each company link looks like:
    #   <a href="https://www.indiabix.com/placement-papers/<slug>/"><slug-or-name><span...>...</span></a>
    # The visible text may contain inner <span> tags (e.g. for full company name).
    # We capture the slug (from URL) + all text content (stripping tags later).
    pairs = re.findall(
        r'href="https://www\.indiabix\.com/placement-papers/([a-z0-9-]+)/"\s*>([^<]+(?:<span[^>]*>[^<]*</span>[^<]*)*)</a>',
        r.text, re.IGNORECASE
    )
    # Deduplicate by slug
    seen = set()
    out = []
    for slug, raw_name in pairs:
        if slug in seen:
            continue
        seen.add(slug)
        # Strip inner tags
        name = re.sub(r'<[^>]+>', '', raw_name).strip()
        if not name:
            name = slug.replace("-", " ").title()
        out.append((name, slug))
    # Fuzzy filter by query
    q = _norm(query)
    if not q:
        return out
    qtokens = q.split()
    def matches(name, slug):
        combined = _norm(name) + " " + _norm(slug)
        return all(t in combined for t in qtokens)
    strict = [(n, s) for n, s in out if matches(n, s)]
    if strict:
        return strict
    # Loose match: any token appears
    loose = [(n, s) for n, s in out
             if any(t in _norm(n) or t in _norm(s) for t in qtokens)]
    return loose[:10]


async def _indiabix_list_company_papers(client: httpx.AsyncClient,
                                         slug: str) -> list[dict]:
    """Fetch the first page of papers for a company. Returns:
    [{title, posted_by, url, hearts}, ...]
    The source site paginates with /page-N; we only fetch page 1 to keep the bot
    responsive (typically 10 papers per page)."""
    r = await client.get(f"{INDIABIX_BASE}/placement-papers/{slug}/",
                         timeout=TIMEOUT)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    papers = []
    # Pattern: <div class="paper-list row"> <div class="col-12 paper-head">
    #   <a href="https://www.indiabix.com/placement-papers/<slug>/<id>">Title</a>
    #   </div> <div class="col-12 paper-details"> ...
    #   <div class="postedby-name">Name</div> ... <div class="postedby-name">(N)</div>
    # </div>
    blocks = re.findall(
        r'<div class="paper-list row">\s*'
        r'<div class="col-12 paper-head"><a href="([^"]+)">([^<]+)</a></div>\s*'
        r'<div class="col-12 paper-details">.*?'
        r'<div class="postedby-name">([^<]+)</div>.*?</div>\s*</div>',
        r.text, re.DOTALL
    )
    for url, title, posted_by in blocks:
        title = title.strip()
        posted_by = posted_by.strip()
        papers.append({
            "title": title,
            "posted_by": posted_by,
            "url": url,
        })
    return papers


async def _indiabix_fetch_paper_content(client: httpx.AsyncClient,
                                         paper_url: str) -> str:
    """Fetch an individual paper page and extract the main content text
    (the interview experience / question write-up)."""
    r = await client.get(paper_url, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text
    # The paper content lives in <div class="paper-data">...</div>
    # The content is plain text with <br /> line breaks and &nbsp; entities.
    # NOTE: We use [\s\S]*? instead of .*? with DOTALL because of a Python
    # regex quirk where (.*?) + \s* + literal "=..." fails to match.
    # We also use ["']  instead of a bare "  for the same reason.
    m = re.search(r'<div class=["\']paper-data["\']>([\s\S]*?)</div>\s*<div class=["\']edit-btn-box',
                  html)
    if not m:
        return ""
    content_html = m.group(1)
    # Convert <br> to newlines
    content = re.sub(r'<br\s*/?>', '\n', content_html, flags=re.IGNORECASE)
    # Strip remaining tags
    content = re.sub(r'<[^>]+>', ' ', content)
    # Decode HTML entities
    content = re.sub(r'&nbsp;', ' ', content)
    content = re.sub(r'&amp;', '&', content)
    content = re.sub(r'&lt;', '<', content)
    content = re.sub(r'&gt;', '>', content)
    content = re.sub(r'&#\d+;', '', content)
    content = re.sub(r'&#x[0-9a-fA-F]+;', '', content)
    # Clean whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)
    return content.strip()


# =====================================================================
#  SECTION 3 — COMMON INTERVIEW QUESTIONS  (scraped from the source site)
#
#  Three sub-sections:
#    1. HR Interview       -> https://www.indiabix.com/hr-interview/<topic>/
#    2. Group Discussion   -> https://www.indiabix.com/group-discussion/<topic>/
#    3. Technical Interview-> https://www.indiabix.com/technical/<topic>/<subtopic>/
#
#  For HR and GD: each topic page has the question/topic as h1 + a list of
#  user-submitted answer discussions (div.bix-sun-discussion).
#  For Technical: the topic page (e.g. /technical/python/) lists sub-topics;
#  each sub-topic page (e.g. /technical/python/variables/) has Q&A blocks
#  (div.bix-tech-qa with bix-td-qno / bix-td-qtxt / bix-tbl-answer).
# =====================================================================

async def _indiabix_list_hr_questions(client: httpx.AsyncClient) -> list[tuple[str, str]]:
    """Returns [(question_text, url), ...] for all HR interview questions."""
    r = await client.get(f"{INDIABIX_BASE}/hr-interview/questions-and-answers/",
                         timeout=TIMEOUT)
    r.raise_for_status()
    pairs = re.findall(
        r'<a[^>]+href="(https://www\.indiabix\.com/hr-interview/[a-z0-9-]+/)"[^>]*>([^<]+)</a>',
        r.text, re.IGNORECASE
    )
    seen = set()
    out = []
    for url, text in pairs:
        text = text.strip()
        # Skip the index link
        if "questions-and-answers" in url or not text:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((text, url))
    return out


async def _indiabix_list_gd_topics(client: httpx.AsyncClient) -> list[tuple[str, str]]:
    """Returns [(topic_text, url), ...] for all GD topics."""
    r = await client.get(f"{INDIABIX_BASE}/group-discussion/topics-with-answers/",
                         timeout=TIMEOUT)
    r.raise_for_status()
    pairs = re.findall(
        r'<a[^>]+href="(https://www\.indiabix\.com/group-discussion/[a-z0-9-]+/)"[^>]*>([^<]+)</a>',
        r.text, re.IGNORECASE
    )
    seen = set()
    out = []
    for url, text in pairs:
        text = text.strip()
        if "topics-with-answers" in url or not text:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((text, url))
    return out


async def _indiabix_list_tech_topics(client: httpx.AsyncClient) -> list[tuple[str, str]]:
    """Returns [(topic_name, url), ...] for technical interview topics
    (Python, C, Java, Data Structures, ...)."""
    r = await client.get(f"{INDIABIX_BASE}/technical/interview-questions-and-answers/",
                         timeout=TIMEOUT)
    r.raise_for_status()
    pairs = re.findall(
        r'<a[^>]+href="(https://www\.indiabix\.com/technical/[a-z0-9-]+/)"[^>]*>([^<]+)</a>',
        r.text, re.IGNORECASE
    )
    seen = set()
    out = []
    for url, text in pairs:
        text = text.strip()
        if "interview-questions-and-answers" in url or not text:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((text, url))
    return out


async def _indiabix_list_tech_subtopics(client: httpx.AsyncClient,
                                          topic_url: str) -> list[tuple[str, str]]:
    """For a tech topic page (e.g. /technical/python/), return its sub-topics
    (e.g. Variables, Data Types, ...).

    If the topic page has NO sub-topics (i.e., the Q&As are listed directly
    on the topic page itself, like /technical/dbms-basics/), we return a
    single-entry list pointing back to the topic URL — so the user can still
    pick it and we'll scrape the topic page directly for Q&As."""
    r = await client.get(topic_url, timeout=TIMEOUT)
    r.raise_for_status()
    # Sub-topics live inside <ul class="need-ul-filter"> as:
    #   <li><a href="https://www.indiabix.com/technical/python/variables/">Variables<span...></span></a></li>
    # The visible text may contain inner <span> tags — strip them later.
    prefix = topic_url.replace(INDIABIX_BASE, "").rstrip("/")
    # Match either absolute URLs (https://www.indiabix.com/technical/python/variables/)
    # or relative ones (/technical/python/variables/).
    pattern = re.compile(
        rf'<a[^>]+href="(?:https://www\.indiabix\.com)?({re.escape(prefix)}/[a-z0-9-]+/)"[^>]*>([^<]+(?:<span[^>]*>[^<]*</span>[^<]*)*)</a>',
        re.IGNORECASE
    )
    seen = set()
    out = []
    for sub_path, raw_text in pattern.findall(r.text):
        # Strip inner tags
        text = re.sub(r'<[^>]+>', '', raw_text).strip()
        if not text or len(text) > 80:
            continue
        full_url = INDIABIX_BASE + sub_path
        if full_url == topic_url or full_url in seen:
            continue
        seen.add(full_url)
        out.append((text, full_url))

    # Fallback: if no sub-topics found, treat the topic page itself as a Q&A page.
    # This happens for topics like DBMS Basics, Networking Basics, etc. where
    # the Q&As are listed directly on the topic page (with pagination).
    if not out:
        # Use the topic name (last path segment) as the label
        topic_name = prefix.rsplit("/", 1)[-1].replace("-", " ").title()
        out.append((topic_name, topic_url))
        log.info("pyq: topic %s has no sub-topics — treating topic page "
                 "as Q&A source", topic_url)

    return out


async def _indiabix_fetch_discussion_answers(client: httpx.AsyncClient,
                                                page_url: str) -> list[str]:
    """For an HR question or GD topic page, extract all user-submitted
    discussion answers. Returns a list of answer strings.

    The source site paginates discussions (e.g. 64 pages for popular GD topics).
    We fetch page 1 (which has ~10 answers) and return them. The PDF generator
    will then pick the top 5."""
    r = await client.get(page_url, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text
    answers = []
    # Each answer is in <div class="bix-sun-discussion d-flex flex-column px-2">
    # Inside it: <div class="user-content">TEXT</div>
    # We split by the opening div tag (similar to tech Q&A extraction).
    blocks = re.split(r'<div class="bix-sun-discussion d-flex flex-column px-2"', html)
    for block in blocks[1:]:  # skip the part before the first block
        # Take up to the next bix-sun-discussion or end of relevant section
        # (the block ends naturally before the next discussion or a <hr>)
        end = block.find('<hr>')
        if end > 0:
            block = block[:end]
        m = re.search(r'<div class="user-content"[^>]*>(.*?)</div>', block, re.DOTALL)
        if not m:
            continue
        text = m.group(1)
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'</p>', '\n\n', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        if text and len(text) > 20:
            answers.append(text)
    return answers


async def _indiabix_fetch_tech_qas(client: httpx.AsyncClient,
                                     subtopic_url: str) -> list[dict]:
    """For a technical sub-topic page (e.g. /technical/python/variables/),
    extract Q&A pairs from ALL pages.

    The source site paginates Q&A pages: /technical/python/variables/ (page 1),
    /technical/python/variables/2 (page 2), ..., /technical/python/variables/N.
    The first page contains a "Page 1 of N" marker that tells us the total.
    We iterate through all N pages and combine all Q&As.

    Returns [{qno, question, answer, answer_html}, ...]."""
    all_qas: list[dict] = []
    # Start with the base URL (page 1)
    current_url = subtopic_url.rstrip("/")
    page_num = 1
    total_pages = 1  # default; updated after first page is fetched

    while page_num <= total_pages:
        try:
            r = await client.get(current_url, timeout=TIMEOUT)
            r.raise_for_status()
        except Exception as e:
            log.warning("pyq fetch page %d failed: %s", page_num, e)
            break

        html = r.text

        # Detect total pages from "Page X of Y" marker (only on first page)
        if page_num == 1:
            pag_m = re.search(r'Page\s+\d+\s+of\s+(\d+)', html, re.IGNORECASE)
            if pag_m:
                total_pages = int(pag_m.group(1))
                log.info("pyq: %s has %d page(s) of Q&A",
                         subtopic_url, total_pages)

        # Extract Q&A blocks from this page
        page_qas = _extract_qas_from_html(html)
        if not page_qas:
            break  # no more Q&As
        all_qas.extend(page_qas)

        # Next page
        page_num += 1
        if page_num > total_pages:
            break
        # Build URL: /technical/python/variables/2  (no trailing slash on base)
        current_url = f"{subtopic_url.rstrip('/')}/{page_num}"
        await asyncio.sleep(0.3)  # be polite

    return all_qas


def _extract_qas_from_html(html: str) -> list[dict]:
    """Extract Q&A blocks from a single technical page.
    Each Q&A is in <div class="bix-tech-qa"> ... </div>."""
    qas = []
    # Use a non-greedy split
    blocks = re.split(r'<div class="bix-tech-qa">', html)
    for block in blocks[1:]:  # skip the part before the first block
        # Each block ends at the next <hr> or next bix-tech-qa (already split)
        # Take up to the next <hr>
        end = block.find('<hr>')
        if end > 0:
            block = block[:end]
        # Question number
        qno_m = re.search(r'<div class="bix-td-qno"[^>]*>(.*?)</div>', block, re.DOTALL)
        qno = re.sub(r'<[^>]+>', '', qno_m.group(1)).strip() if qno_m else ""
        # Question text
        qtxt_m = re.search(r'<div class="bix-td-qtxt[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
        qtxt = re.sub(r'<[^>]+>', ' ', qtxt_m.group(1)).strip() if qtxt_m else ""
        qtxt = re.sub(r'\s+', ' ', qtxt)
        # Answer (HTML preserved for code-block extraction later)
        ans_m = re.search(r'<div class="bix-tbl-answer"[^>]*>(.*?)</div>\s*<div', block, re.DOTALL)
        ans_html = ans_m.group(1).strip() if ans_m else ""
        # Plain-text version
        ans_text = re.sub(r'<br\s*/?>', '\n', ans_html)
        ans_text = re.sub(r'</p>', '\n\n', ans_text)
        ans_text = re.sub(r'<[^>]+>', ' ', ans_text)
        ans_text = re.sub(r'&nbsp;', ' ', ans_text)
        ans_text = re.sub(r'&amp;', '&', ans_text)
        ans_text = re.sub(r'&lt;', '<', ans_text)
        ans_text = re.sub(r'&gt;', '>', ans_text)
        ans_text = re.sub(r'&#\d+;', '', ans_text)
        ans_text = re.sub(r'\n\s*\n', '\n\n', ans_text)
        ans_text = re.sub(r'[ \t]+', ' ', ans_text).strip()
        if qtxt:
            qas.append({
                "qno": qno,
                "question": qtxt,
                "answer": ans_text,
                "answer_html": ans_html,
            })
    return qas


# =====================================================================
#  PDF GENERATION  (reportlab)  — shared by all 3 interview sections
# =====================================================================
def _pdf_setup_fonts():
    """Pick fonts. Returns (body_font, bold_font, mono_font).
    Uses DejaVu if available (handles unicode + em-dashes + code symbols)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font, bold, mono = "Times-Roman", "Times-Bold", "Courier"
    for reg_path, bold_path in (
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSerif.ttf",
         "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ):
        if os.path.exists(reg_path) and os.path.exists(bold_path):
            try:
                pdfmetrics.registerFont(TTFont("PPBody", reg_path))
                pdfmetrics.registerFont(TTFont("PPBody-Bold", bold_path))
                font, bold = "PPBody", "PPBody-Bold"
                break
            except Exception:
                pass
    for mono_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ):
        if os.path.exists(mono_path):
            try:
                pdfmetrics.registerFont(TTFont("PPMono", mono_path))
                mono = "PPMono"
                break
            except Exception:
                pass
    return font, bold, mono


def _md_to_xml(text: str, mono_font: str) -> str:
    """Convert minimal markdown to reportlab tags, and XML-escape everything else."""
    import re as _re
    if not text:
        return ""
    # Convert markdown bold/italic/code
    text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = _re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = _re.sub(r"`([^`]+)`", r'<font face="%s">\1</font>' % mono_font, text)
    # Strip markdown headings
    text = _re.sub(r"^#+\s*", "", text, flags=_re.MULTILINE)
    # Bullets
    text = _re.sub(r"^\s*[-*]\s+", "• ", text, flags=_re.MULTILINE)
    # XML-escape everything that isn't our intentional tags
    out, i = [], 0
    tag_re = _re.compile(r"<(?:/?(?:b|i|font[^>]*)|br/?)>")
    while i < len(text):
        if text[i] == "<":
            m = tag_re.match(text, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        if text[i] == "<":
            out.append("&lt;"); i += 1; continue
        if text[i] == ">":
            out.append("&gt;"); i += 1; continue
        if text[i] == "&":
            out.append("&amp;"); i += 1; continue
        out.append(text[i]); i += 1
    return "".join(out)


def _indiabix_company_papers_to_pdf(company_name: str,
                                      papers: list[dict]) -> bytes | None:
    """Build a PDF bundling multiple company placement papers.
    Each paper = {title, posted_by, url, content}."""
    import io
    try:
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        PageBreak, HRFlowable, Preformatted)
    except ImportError:
        log.warning("reportlab not installed: cannot build company PDF")
        return None
    font, bold, mono = _pdf_setup_fonts()
    st = {
        "title":     ParagraphStyle("t", fontName=bold, fontSize=22, leading=28,
                                    alignment=TA_CENTER, spaceAfter=10),
        "subtitle":  ParagraphStyle("st", fontName=font, fontSize=13, leading=18,
                                    alignment=TA_CENTER, spaceAfter=6),
        "meta":      ParagraphStyle("m", fontName=font, fontSize=10, leading=14,
                                    alignment=TA_CENTER, textColor=colors.grey),
        "paper_h":   ParagraphStyle("ph", fontName=bold, fontSize=14, leading=20,
                                    spaceBefore=18, spaceAfter=4,
                                    textColor=colors.HexColor("#0b5394")),
        "paper_sub": ParagraphStyle("ps", fontName=font, fontSize=9, leading=12,
                                    spaceAfter=8, textColor=colors.grey),
        "body":      ParagraphStyle("b", fontName=font, fontSize=10, leading=14,
                                    alignment=TA_JUSTIFY, spaceAfter=6),
    }
    story = [Spacer(1, 50 * mm)]
    story.append(Paragraph(_md_to_xml(company_name, mono), st["title"]))
    story.append(Paragraph(f"Placement Papers — {len(papers)} paper(s)", st["subtitle"]))
    story.append(Spacer(1, 6 * mm))
    story.append(PageBreak())
    for i, p in enumerate(papers, 1):
        story.append(Paragraph(f"{i}. {_md_to_xml(p.get('title', ''), mono)}",
                                st["paper_h"]))
        sub = f"Posted by: {p.get('posted_by', '—')}"
        story.append(Paragraph(_md_to_xml(sub, mono), st["paper_sub"]))
        story.append(HRFlowable(width="100%", thickness=0.3,
                                color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 3 * mm))
        content = p.get("content", "")
        # Split content into paragraphs
        for para in content.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            # Long paragraphs may need to be split further
            story.append(Paragraph(_md_to_xml(para, mono), st["body"]))
        if i < len(papers):
            story.append(PageBreak())
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.drawString(15 * mm, 10 * mm, f"{company_name} - Placement Papers")
        canvas.restoreState()
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4,
                      leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=15 * mm, bottomMargin=18 * mm,
                      title=f"{company_name} Placement Papers",
                      author="").build(story,
                                                onFirstPage=_footer,
                                                onLaterPages=_footer)
    return buf.getvalue()


def _indiabix_discussion_to_pdf(section_name: str, topic: str,
                                  answers: list[str]) -> bytes | None:
    """Build a PDF for an HR question or GD topic, bundling all user-submitted
    discussion answers. Used by Section 3 (HR + GD)."""
    import io
    try:
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        PageBreak, HRFlowable)
    except ImportError:
        return None
    font, bold, mono = _pdf_setup_fonts()
    st = {
        "title":     ParagraphStyle("t", fontName=bold, fontSize=20, leading=26,
                                    alignment=TA_CENTER, spaceAfter=8),
        "subtitle":  ParagraphStyle("st", fontName=font, fontSize=12, leading=16,
                                    alignment=TA_CENTER, textColor=colors.grey,
                                    spaceAfter=6),
        "ans_h":     ParagraphStyle("ah", fontName=bold, fontSize=11, leading=15,
                                    spaceBefore=10, spaceAfter=4,
                                    textColor=colors.HexColor("#0b5394")),
        "body":      ParagraphStyle("b", fontName=font, fontSize=10, leading=14,
                                    alignment=TA_JUSTIFY, spaceAfter=6),
    }
    story = [Spacer(1, 50 * mm)]
    story.append(Paragraph(_md_to_xml(topic, mono), st["title"]))
    # Limit to top 5 answers (most recent / most upvoted on the source site)
    top_answers = answers[:5]
    story.append(Paragraph(f"{section_name} — Top {len(top_answers)} Answer(s)", st["subtitle"]))
    story.append(PageBreak())
    for i, ans in enumerate(top_answers, 1):
        story.append(Paragraph(f"Answer {i}", st["ans_h"]))
        story.append(HRFlowable(width="100%", thickness=0.3,
                                color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 2 * mm))
        for para in ans.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(_md_to_xml(para, mono), st["body"]))
        if i < len(top_answers):
            story.append(Spacer(1, 6 * mm))
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.drawString(15 * mm, 10 * mm, f"{section_name} - {topic[:40]}")
        canvas.restoreState()
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4,
                      leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=15 * mm, bottomMargin=18 * mm,
                      title=f"{section_name} - {topic}",
                      author="").build(story,
                                                onFirstPage=_footer,
                                                onLaterPages=_footer)
    return buf.getvalue()


def _indiabix_tech_qas_to_pdf(topic_name: str, subtopic_name: str,
                                 qas: list[dict]) -> bytes | None:
    """Build a PDF for technical Q&A. Each Q has a number, question, and an
    answer that may contain code blocks (rendered in monospace)."""
    import io
    import re as _re
    try:
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        PageBreak, HRFlowable, Preformatted)
    except ImportError:
        return None
    font, bold, mono = _pdf_setup_fonts()
    st = {
        "title":   ParagraphStyle("t", fontName=bold, fontSize=20, leading=26,
                                  alignment=TA_CENTER, spaceAfter=8),
        "subtitle":ParagraphStyle("st", fontName=font, fontSize=12, leading=16,
                                  alignment=TA_CENTER, textColor=colors.grey,
                                  spaceAfter=6),
        "q_title": ParagraphStyle("qt", fontName=bold, fontSize=12, leading=16,
                                  spaceBefore=12, spaceAfter=4,
                                  textColor=colors.HexColor("#0b5394")),
        "body":    ParagraphStyle("b", fontName=font, fontSize=10, leading=14,
                                  alignment=TA_JUSTIFY, spaceAfter=6),
        "code":    ParagraphStyle("c", fontName=mono, fontSize=8.5, leading=11,
                                  leftIndent=8, rightIndent=8,
                                  backColor=colors.HexColor("#f5f5f5"),
                                  borderColor=colors.HexColor("#e0e0e0"),
                                  borderWidth=0.5, borderPadding=4,
                                  spaceBefore=4, spaceAfter=6),
    }
    story = [Spacer(1, 50 * mm)]
    story.append(Paragraph(_md_to_xml(f"{topic_name} - {subtopic_name}", mono),
                            st["title"]))
    story.append(Paragraph(f"Technical Interview — {len(qas)} Q&A", st["subtitle"]))
    story.append(PageBreak())
    for i, qa in enumerate(qas, 1):
        qno = qa.get("qno", str(i)).strip()
        q = qa.get("question", "").strip()
        ans_html = qa.get("answer_html", "")
        story.append(Paragraph(f"Q{qno}. {_md_to_xml(q, mono)}", st["q_title"]))
        story.append(HRFlowable(width="100%", thickness=0.3,
                                color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 2 * mm))
        # Parse answer HTML: extract <p>...</p> and <pre><code>...</code></pre>
        # Simplify: split by <pre> to separate code blocks from prose
        parts = _re.split(r'<pre><code[^>]*>(.*?)</code></pre>', ans_html, flags=_re.DOTALL)
        # parts = [prose, code, prose, code, ...]
        for j, part in enumerate(parts):
            if j % 2 == 0:
                # Prose — may have <p> tags
                prose = part
                # Convert <p>...</p> to paragraphs
                paras = _re.findall(r'<p[^>]*>(.*?)</p>', prose, _re.DOTALL)
                if not paras:
                    # No <p> tags — treat the whole thing as one paragraph
                    cleaned = _re.sub(r'<[^>]+>', ' ', prose)
                    cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
                    if cleaned:
                        story.append(Paragraph(_md_to_xml(cleaned, mono), st["body"]))
                else:
                    for p in paras:
                        cleaned = _re.sub(r'<[^>]+>', ' ', p)
                        cleaned = _re.sub(r'&nbsp;', ' ', cleaned)
                        cleaned = _re.sub(r'&lt;', '<', cleaned)
                        cleaned = _re.sub(r'&gt;', '>', cleaned)
                        cleaned = _re.sub(r'&amp;', '&', cleaned)
                        cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
                        if cleaned:
                            story.append(Paragraph(_md_to_xml(cleaned, mono), st["body"]))
            else:
                # Code block
                code = _re.sub(r'<[^>]+>', '', part)
                code = _re.sub(r'&nbsp;', ' ', code)
                code = _re.sub(r'&lt;', '<', code)
                code = _re.sub(r'&gt;', '>', code)
                code = _re.sub(r'&amp;', '&', code)
                code = code.strip()
                if code:
                    story.append(Preformatted(_md_to_xml(code, mono), st["code"]))
        if i < len(qas):
            story.append(Spacer(1, 4 * mm))
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.drawString(15 * mm, 10 * mm, f"{topic_name} - {subtopic_name}"[:50])
        canvas.restoreState()
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4,
                      leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=15 * mm, bottomMargin=18 * mm,
                      title=f"{topic_name} - {subtopic_name}",
                      author="").build(story,
                                                onFirstPage=_footer,
                                                onLaterPages=_footer)
    return buf.getvalue()


# =====================================================================
#  PUBLIC INTERFACE — required by the bot
# =====================================================================
def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


async def start(phone: str, arg: str, say) -> bool:
    """/pyq [arg]. True = done (state cleared), False = waiting for more input.

    If `arg` is provided, treat it as a company name and route directly
    to the company-paper search flow. Otherwise show the 3-section menu."""
    if arg and arg.strip():
        # Treat any argument as a company name search
        await asyncio.to_thread(_save, phone,
            {"mode": MODE, "step": "company_search"})
        if not _quota_ok(phone):
            await asyncio.to_thread(_save, phone, {})
            await say(f"⏳ Daily limit reached ({MAX_PER_DAY}). Try tomorrow.")
            return True
        await _company_fetch_and_show(phone, say, arg.strip())
        return False
    # No arg -> show menu
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
    await say(PROMPT)
    return False


# =====================================================================
#  PAGINATED LIST RENDERERS
# =====================================================================
async def _show_universities(phone: str, say, items: list[dict], page: int = 0) -> None:
    if not items:
        await say("⚠️ No universities found. Try *back* for the menu or *exit*.")
        return
    total_pages = (len(items) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = items[start_i:start_i + PAGE_SIZE]
    lines = [_UNI_PROMPT]
    for i, it in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {it['description'][:90]}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* for menu", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _show_courses(phone: str, say, uni_name: str, uni_url: str,
                        courses: list[tuple[str, str]], page: int = 0) -> None:
    if not courses:
        await say(f"⚠️ No courses detected on *{uni_name}*. Try another university, or *back*.")
        return
    total_pages = (len(courses) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = courses[start_i:start_i + PAGE_SIZE]
    lines = [_COURSE_PROMPT.format(uni=uni_name[:60])]
    for i, (name, _) in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {name}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* for universities", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _show_branches(phone: str, say, uni_name: str, course_name: str,
                         course_url: str, branches: list[tuple[str, str]],
                         page: int = 0) -> None:
    if not branches:
        await say(f"📝 No branch subdivision for *{course_name}*. "
                  f"Reply with semester + year (e.g. `1 2026`), or *back* / *exit*.")
        return
    total_pages = (len(branches) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = branches[start_i:start_i + PAGE_SIZE]
    lines = [_BRANCH_PROMPT.format(uni=uni_name[:40], course=course_name)]
    for i, (name, _) in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {name}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* for courses", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _show_papers(phone: str, say, papers: list[dict], page: int = 0) -> None:
    if not papers:
        await say("😕 No papers matched. Send *back* to change the filter, or *exit*.")
        return
    total_pages = (len(papers) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = papers[start_i:start_i + PAGE_SIZE]
    lines = [_PAPERS_PROMPT.format(n=len(papers))]
    for i, p in enumerate(chunk, 1):
        sem_str = f"sem {p['sem']}" if p["sem"] else "sem ?"
        yr_str = str(p["year"]) if p["year"] else "????"
        # Shorten the paper name for WhatsApp readability
        short = p["name"][:80]
        lines.append(f"*{start_i + i}.* {sem_str}, {yr_str}  |  {short}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* to filter", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


# =====================================================================
#  RENDERERS FOR SECTION 2 (Company-wise PYQ) and 3 (Interview Qs)
# =====================================================================
async def _show_company_papers_list(phone: str, say, company_name: str,
                                       papers: list[dict], page: int = 0) -> None:
    """Paginated list of placement papers for a company."""
    if not papers:
        await say(f"⚠️ No papers found for *{company_name}*. "
                  f"Try another company, or *back* / *exit*.")
        return
    total_pages = (len(papers) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = papers[start_i:start_i + PAGE_SIZE]
    lines = [_COMPANY_PAPERS_PROMPT.format(company=company_name, n=len(papers))]
    for i, p in enumerate(chunk, 1):
        title = p.get("title", "")[:70]
        posted = p.get("posted_by", "")[:30]
        lines.append(f"*{start_i + i}.* {title}")
        lines.append(f"    _by {posted}_")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* to type another company", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _show_interview_menu(phone: str, say) -> None:
    """Show the 3-subsection menu for Common Interview Questions."""
    await say(_INTERVIEW_PROMPT)


async def _show_paginated_simple(say, prompt: str, items: list[tuple[str, str]],
                                    page: int = 0,
                                    back_label: str = "back for menu") -> None:
    """Generic paginated list renderer for HR/GD/Technical topic lists."""
    if not items:
        await say("⚠️ No items found. Send *back* or *exit*.")
        return
    total_pages = (len(items) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = items[start_i:start_i + PAGE_SIZE]
    lines = [prompt]
    for i, (name, _) in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {name[:80]}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += [f"*{back_label}*", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _show_hr_questions(phone: str, say, items: list[tuple[str, str]],
                                page: int = 0) -> None:
    await _show_paginated_simple(say, _HR_PROMPT, items, page,
                                  back_label="back for interview menu")


async def _show_gd_topics(phone: str, say, items: list[tuple[str, str]],
                            page: int = 0) -> None:
    await _show_paginated_simple(say, _GD_PROMPT, items, page,
                                  back_label="back for interview menu")


async def _show_tech_topics(phone: str, say, items: list[tuple[str, str]],
                              page: int = 0) -> None:
    await _show_paginated_simple(say, _TECH_PROMPT, items, page,
                                  back_label="back for interview menu")


async def _show_tech_subtopics(phone: str, say, topic_name: str,
                                 subtopics: list[tuple[str, str]],
                                 page: int = 0) -> None:
    """Paginated list of sub-topics for a chosen tech topic."""
    if not subtopics:
        await say(f"⚠️ No sub-topics found for *{topic_name}*. Send *back* or *exit*.")
        return
    total_pages = (len(subtopics) + PAGE_SIZE - 1) // PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * PAGE_SIZE
    chunk = subtopics[start_i:start_i + PAGE_SIZE]
    lines = [f"💻 *{topic_name}*\n\n"
             f"Pick a sub-topic (or *next* / *prev* / *back* / *exit*):\n"]
    for i, (name, _) in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {name[:80]}")
    nav = []
    if page > 0:
        nav.append("*prev*")
    if page < total_pages - 1:
        nav.append("*next*")
    nav += ["*back* for tech topics", "*exit*"]
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


# =====================================================================
#  DOWNLOAD + DELIVER  (selected papers or all)
# =====================================================================
async def _deliver_papers(phone: str, say, papers: list[dict]) -> int:
    """Download each paper and send via WhatsApp. Returns count delivered."""
    delivered = 0
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        for i, p in enumerate(papers, 1):
            await say(f"⏳ [{i}/{len(papers)}] Downloading *{p['name'][:60]}* ...")
            data, why = await _download_paper_bytes(client, p)
            if not data:
                note = "too large for WhatsApp" if why == "too big" else f"couldn't download ({why})"
                await say(f"📎 *{p['name']}*\n\nHere's the direct link ({note}):\n{p['pdf_url']}")
                continue
            size_kb = len(data) / 1024
            caption = (f"📝 {p['name']}\n"
                       f"📊 Size: {size_kb:.0f} KB")
            await _send_file(phone, p["name"], data, p["pdf_url"], say, caption)
            delivered += 1
            await asyncio.sleep(0.5)                # be polite to WhatsApp rate-limits
    return delivered


# =====================================================================
#  PARSE USER SELECTION (e.g. "1,3,5" or "1-4" or "all")
# =====================================================================
def _parse_paper_selection(text: str, total: int) -> list[int] | None:
    """Returns 0-indexed positions, or None if 'all', or [] if invalid."""
    low = text.strip().lower()
    if low == "all":
        return list(range(total))
    picked = set()
    for token in low.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]), int(parts[1])
                for k in range(min(a, b), max(a, b) + 1):
                    if 1 <= k <= total:
                        picked.add(k - 1)
        elif token.isdigit():
            k = int(token)
            if 1 <= k <= total:
                picked.add(k - 1)
    return sorted(picked)


# =====================================================================
#  Helpers for Section 2 (company search) and 3 (interview fetch + PDF)
# =====================================================================
async def _company_fetch_and_show(phone: str, say, company_query: str) -> bool:
    """Search for the company, fetch the paper list, save state,
    and show the paginated list. Returns True on success."""
    await say(f"🔎 Searching for *{company_query}* ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        matches = await _indiabix_search_companies(client, company_query)
    if not matches:
        await say(f"😕 No company named *{company_query}* found. "
                  f"Try another name (e.g. 'TCS', 'Infosys', 'Wipro'), or *back* / *exit*.")
        return False
    if len(matches) == 1:
        company_name, slug = matches[0]
    else:
        # Multiple matches — pick the best one (shortest name = most specific)
        # or ask user to disambiguate
        if len(matches) <= 5:
            # Save matches and ask user
            await asyncio.to_thread(_save, phone, {
                "mode": MODE, "step": "company_pick",
                "matches": matches,
            })
            lines = [f"🔍 Multiple companies match *{company_query}*. Pick one:\n"]
            for i, (name, _) in enumerate(matches, 1):
                lines.append(f"*{i}.* {name}")
            lines.append("\nSend *back* to type another name, or *exit*.")
            await say("\n".join(lines))
            return False
        # More than 5 — pick the first
        company_name, slug = matches[0]

    await say(f"🔎 Fetching papers for *{company_name}* ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        papers = await _indiabix_list_company_papers(client, slug)
    if not papers:
        await say(f"😕 No papers found for *{company_name}*. "
                  f"Try another company, or *back* / *exit*.")
        return False
    await asyncio.to_thread(_save, phone, {
        "mode": MODE, "step": "company_papers",
        "company_name": company_name,
        "company_slug": slug,
        "papers": papers,
        "page": 0,
    })
    await _show_company_papers_list(phone, say, company_name, papers, page=0)
    return True


async def _company_deliver_papers(phone: str, say, company_name: str,
                                     papers: list[dict], picked_indices: list[int]) -> None:
    """Fetch content for selected papers, build a single PDF, send via WhatsApp."""
    to_fetch = [papers[i] for i in picked_indices]
    await say(f"📥 Fetching {len(to_fetch)} paper(s) ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        for i, p in enumerate(to_fetch, 1):
            await say(f"⏳ [{i}/{len(to_fetch)}] {p['title'][:60]} ...")
            try:
                content = await _indiabix_fetch_paper_content(client, p["url"])
                p["content"] = content or "(No content available.)"
            except Exception as e:
                log.warning("paper fetch failed: %s", e)
                p["content"] = f"(Failed to fetch content: {e})"
            await asyncio.sleep(0.3)  # be polite
    await say(f"📄 Generating PDF ...")
    pdf_bytes = await asyncio.to_thread(
        _indiabix_company_papers_to_pdf, company_name, to_fetch)
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        await say("⚠️ PDF generation failed. Please try another paper, or *back* / *exit*.")
        return
    if len(pdf_bytes) > MAX_MB * 1024 * 1024:
        await say(f"📎 The PDF is too large for WhatsApp ({len(pdf_bytes)//1024} KB). "
                  f"Try picking fewer papers, or *back*.")
        return
    size_kb = len(pdf_bytes) / 1024
    caption = (f"📝 {company_name} - Placement Papers\n"
               f"📊 {len(to_fetch)} paper(s)\n"
               f"📦 Size: {size_kb:.0f} KB")
    await _send_file(phone, f"{company_name}_papers", pdf_bytes,
                      "https://www.indiabix.com/placement-papers/", say, caption)
    await say(f"✅ Sent *{company_name}* ({len(to_fetch)} papers). "
              f"Send *back* for more, or */pyq* for the menu.")


async def _hr_fetch_and_deliver(phone: str, say, question_text: str,
                                  page_url: str) -> None:
    """Fetch discussion answers for an HR question, build PDF, send."""
    await say(f"⏳ Fetching answers for *{question_text[:60]}* ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        answers = await _indiabix_fetch_discussion_answers(client, page_url)
    if not answers:
        await say("⚠️ No answers found for this question. Try another, or *back*.")
        return
    await say(f"📄 Generating PDF ({len(answers)} answers) ...")
    pdf_bytes = await asyncio.to_thread(
        _indiabix_discussion_to_pdf, "HR Interview", question_text, answers)
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        await say("⚠️ PDF generation failed. Try another question, or *back* / *exit*.")
        return
    if len(pdf_bytes) > MAX_MB * 1024 * 1024:
        await say(f"📎 PDF too large ({len(pdf_bytes)//1024} KB). Try another question, or *back*.")
        return
    size_kb = len(pdf_bytes) / 1024
    caption = (f"👤 HR Interview - {question_text[:50]}\n"
               f"📊 Top {min(len(answers), 5)} answer(s)\n"
               f"📦 Size: {size_kb:.0f} KB")
    await _send_file(phone, f"hr_{question_text[:30]}", pdf_bytes, page_url, say, caption)
    await say(f"✅ Sent HR question (top {min(len(answers), 5)} answers). "
              f"Send *back* for more, or */pyq* for the menu.")


async def _gd_fetch_and_deliver(phone: str, say, topic_text: str,
                                  page_url: str) -> None:
    """Fetch discussion answers for a GD topic, build PDF, send."""
    await say(f"⏳ Fetching discussion for *{topic_text[:60]}* ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        answers = await _indiabix_fetch_discussion_answers(client, page_url)
    if not answers:
        await say("⚠️ No discussion found for this topic. Try another, or *back*.")
        return
    await say(f"📄 Generating PDF ({len(answers)} contributions) ...")
    pdf_bytes = await asyncio.to_thread(
        _indiabix_discussion_to_pdf, "Group Discussion", topic_text, answers)
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        await say("⚠️ PDF generation failed. Try another topic, or *back* / *exit*.")
        return
    if len(pdf_bytes) > MAX_MB * 1024 * 1024:
        await say(f"📎 PDF too large ({len(pdf_bytes)//1024} KB). Try another topic, or *back*.")
        return
    size_kb = len(pdf_bytes) / 1024
    caption = (f"👥 GD Topic - {topic_text[:50]}\n"
               f"📊 Top {min(len(answers), 5)} contribution(s)\n"
               f"📦 Size: {size_kb:.0f} KB")
    await _send_file(phone, f"gd_{topic_text[:30]}", pdf_bytes, page_url, say, caption)
    await say(f"✅ Sent GD topic (top {min(len(answers), 5)} contributions). "
              f"Send *back* for more, or */pyq* for the menu.")


async def _tech_fetch_and_deliver(phone: str, say, topic_name: str,
                                     subtopic_name: str, subtopic_url: str) -> None:
    """Fetch Q&As for a technical sub-topic, build PDF, send."""
    await say(f"⏳ Fetching Q&A for *{topic_name} - {subtopic_name}* ...")
    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        qas = await _indiabix_fetch_tech_qas(client, subtopic_url)
    if not qas:
        await say("⚠️ No Q&A found for this sub-topic. Try another, or *back*.")
        return
    await say(f"📄 Generating PDF ({len(qas)} Q&A) ...")
    pdf_bytes = await asyncio.to_thread(
        _indiabix_tech_qas_to_pdf, topic_name, subtopic_name, qas)
    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        await say("⚠️ PDF generation failed. Try another sub-topic, or *back* / *exit*.")
        return
    if len(pdf_bytes) > MAX_MB * 1024 * 1024:
        await say(f"📎 PDF too large ({len(pdf_bytes)//1024} KB). Try another, or *back*.")
        return
    size_kb = len(pdf_bytes) / 1024
    caption = (f"💻 {topic_name} - {subtopic_name}\n"
               f"📊 {len(qas)} Q&A\n"
               f"📦 Size: {size_kb:.0f} KB")
    await _send_file(phone, f"{topic_name}_{subtopic_name}", pdf_bytes,
                      subtopic_url, say, caption)
    await say(f"✅ Sent *{topic_name} - {subtopic_name}* ({len(qas)} Q&A). "
              f"Send *back* for more, or */pyq* for the menu.")


# =====================================================================
#  STATE MACHINE — main handler
# =====================================================================
async def handle(phone: str, text: str, say) -> bool:
    """Continuation handler. True = done (state cleared)."""
    text = (text or "").strip()
    low = text.lower()
    if low in EXIT_WORDS:
        await asyncio.to_thread(_save, phone, {})
        await say("👋 Cancelled.")
        return True

    state = _load(phone)
    step = state.get("step", "menu")

    try:
        # =============================================================
        #  STEP: menu  (1 = University PYQ, 2 = Company-wise, 3 = Interview Qs)
        # =============================================================
        if step == "menu":
            if text == "1":
                # Section 1: University PYQ — ask user to type university name
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "uni", "page": 0})
                await say(_UNI_SEARCH_PROMPT)
                return False
            if text == "2":
                # Section 2: Company-wise PYQ — ask user to type a company name
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "company_search"})
                await say(_COMPANY_PROMPT)
                return False
            if text == "3":
                # Section 3: Common Interview Questions — show HR/GD/Technical sub-menu
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "interview_menu"})
                await _show_interview_menu(phone, say)
                return False
            await say("🤔 Reply with *1* (University PYQ), *2* (Company-wise PYQ), "
                      "*3* (Common Interview Questions), or *exit*.")
            return False

        # =============================================================
        #  STEP: uni  (search-first: user types name → show top matches → pick)
        #  Mirrors the Colab script's main() loop:
        #    q = ask("Enter university name")
        #    matches = search_university(items, q)
        #    show top 8; user picks by number
        # =============================================================
        if step == "uni":
            # Navigation keywords
            if low == "back":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
                await say(PROMPT)
                return False

            # If user previously asked for a search and we cached matches,
            # a digit picks from those matches. Otherwise, free text = new search.
            cached_matches = state.get("uni_matches") or []

            # Handle 'all' → browse full list (paginated)
            if low == "all":
                await say("🔎 Loading all universities ...")
                async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                    items = await _fetch_university_list(client)
                state["items"] = items
                state["page"] = 0
                state.pop("uni_matches", None)
                await asyncio.to_thread(_save, phone, state)
                await _show_universities(phone, say, items, page=0)
                return False

            # Paginate the full list (only after 'all')
            if low in ("next", "prev") and state.get("items"):
                items = state.get("items") or []
                page = state.get("page", 0)
                if low == "next":
                    page += 1
                else:
                    page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_universities(phone, say, items, page=page)
                return False

            # Digit → pick from cached matches (or full list if 'all' was used)
            if text.isdigit():
                n = int(text)
                if cached_matches:
                    if not 1 <= n <= len(cached_matches):
                        await say(f"🤔 Reply 1-{len(cached_matches)}, "
                                  f"or type another university name, or *back* / *exit*.")
                        return False
                    it = cached_matches[n - 1]
                else:
                    items = state.get("items") or []
                    if not items:
                        await say("🤔 Type your university name first, or *all* to browse, "
                                  "or *back* / *exit*.")
                        return False
                    if not 1 <= n <= len(items):
                        await say(f"🤔 Reply 1-{len(items)}, or type a university name, "
                                  f"or *back* / *exit*.")
                        return False
                    it = items[n - 1]

                uni_url = it["url"] if it["url"].startswith("http") else "https://" + it["url"]
                uni_name = it["description"][:60]
                await say(f"🔎 Loading courses for *{uni_name}* ...")
                async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                    courses = await _list_courses(client, uni_url)
                if not courses:
                    await say(f"⚠️ Couldn't detect courses on *{uni_name}*. "
                              f"Try another university, or *back*.")
                    return False
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "course",
                    "uni_name": uni_name, "uni_url": uni_url,
                    "courses": courses, "page": 0,
                })
                await _show_courses(phone, say, uni_name, uni_url, courses, page=0)
                return False

            # Free text → search (mirrors Colab's `search_university(items, q)`)
            if not text or len(text) < 2:
                await say("🤔 Please type your university name (e.g. *BPUT*, *AKTU*, *BHU*), "
                          "or *all* to browse, or *back* / *exit*.")
                return False

            await say(f"🔎 Searching for *{text}* ...")
            async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                items = await _fetch_university_list(client)
            matches = _search_universities(items, text, limit=8)

            if not matches:
                # No match — try to recommend similar (top 5 by loose match)
                loose = []
                qtokens = _norm(text).split()
                for it in items:
                    d = _norm(it["description"])
                    if any(t in d for t in qtokens):
                        loose.append(it)
                if loose:
                    top5 = loose[:5]
                    state["uni_matches"] = top5
                    state.pop("items", None)
                    await asyncio.to_thread(_save, phone, state)
                    lines = [f"😕 No exact match for *{text}*.\n"]
                    lines.append("Did you mean one of these?\n")
                    for i, it in enumerate(top5, 1):
                        lines.append(f"*{i}.* {it['description'][:90]}")
                    lines.append("\nType a *number* to pick, or send another name, "
                                 "or *all* to browse, or *back* / *exit*.")
                    await say("\n".join(lines))
                else:
                    await say(f"😕 No university matches *{text}*. "
                              f"Try another name (e.g. *BPUT*, *AKTU*, *BHU*), "
                              f"or *all* to browse, or *back* / *exit*.")
                return False

            # Single exact match → go straight to courses
            if len(matches) == 1:
                it = matches[0]
                uni_url = it["url"] if it["url"].startswith("http") else "https://" + it["url"]
                uni_name = it["description"][:60]
                await say(f"✅ Found: *{uni_name}*. Loading courses ...")
                async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                    courses = await _list_courses(client, uni_url)
                if not courses:
                    await say(f"⚠️ Couldn't detect courses on *{uni_name}*. "
                              f"Try another university, or *back*.")
                    return False
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "course",
                    "uni_name": uni_name, "uni_url": uni_url,
                    "courses": courses, "page": 0,
                })
                await _show_courses(phone, say, uni_name, uni_url, courses, page=0)
                return False

            # Multiple matches → show top 8 and ask user to pick (mirrors Colab)
            top = matches[:8]
            state["uni_matches"] = top
            state.pop("items", None)
            await asyncio.to_thread(_save, phone, state)
            lines = [f"🔍 Found *{len(matches)}* match(es) for *{text}*.\n"]
            lines.append("Pick one:\n")
            for i, it in enumerate(top, 1):
                lines.append(f"*{i}.* {it['description'][:90]}")
            lines.append("\nType a *number* to pick, or send another name, "
                         "or *all* to browse, or *back* / *exit*.")
            await say("\n".join(lines))
            return False

        # =============================================================
        #  STEP: course
        # =============================================================
        if step == "course":
            courses = state.get("courses") or []
            uni_name = state.get("uni_name", "")
            uni_url = state.get("uni_url", "")
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_courses(phone, say, uni_name, uni_url, courses, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_courses(phone, say, uni_name, uni_url, courses, page=page)
                return False
            if low == "back":
                # Restore full university list for the picker
                async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                    items = await _fetch_university_list(client)
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "uni",
                                                       "items": items, "page": 0})
                await _show_universities(phone, say, items, page=0)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(courses):
                    await say(f"🤔 Reply with a number from 1 to {len(courses)} "
                              f"(or *next* / *back* / *exit*).")
                    return False
                course_name, course_url = courses[n - 1]
                await say(f"🔎 Loading branches for *{course_name}* ...")
                async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                    branches = await _list_branches(client, course_url)
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "branch",
                    "uni_name": uni_name, "uni_url": uni_url,
                    "course_name": course_name, "course_url": course_url,
                    "branches": branches, "page": 0,
                })
                if not branches:
                    # Skip directly to filter step
                    state["step"] = "filter"
                    state["branch_name"] = ""
                    state["branch_url"] = course_url
                    await asyncio.to_thread(_save, phone, state)
                    await say(_FILTER_PROMPT.format(
                        uni=uni_name, course=course_name, branch=""))
                    return False
                await _show_branches(phone, say, uni_name, course_name,
                                     course_url, branches, page=0)
                return False
            await say("🤔 Reply with a course *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # =============================================================
        #  STEP: branch
        # =============================================================
        if step == "branch":
            branches = state.get("branches") or []
            uni_name = state.get("uni_name", "")
            course_name = state.get("course_name", "")
            course_url = state.get("course_url", "")
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_branches(phone, say, uni_name, course_name,
                                     course_url, branches, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_branches(phone, say, uni_name, course_name,
                                     course_url, branches, page=page)
                return False
            if low == "back":
                state["step"] = "course"
                state["page"] = 0
                await asyncio.to_thread(_save, phone, state)
                await _show_courses(phone, say, uni_name, uni_url,
                                    state.get("courses") or [], page=0)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(branches):
                    await say(f"🤔 Reply with a number from 1 to {len(branches)} "
                              f"(or *next* / *back* / *exit*).")
                    return False
                branch_name, branch_url = branches[n - 1]
                state["step"] = "filter"
                state["branch_name"] = branch_name
                state["branch_url"] = branch_url
                state.pop("page", None)
                await asyncio.to_thread(_save, phone, state)
                await say(_FILTER_PROMPT.format(
                    uni=uni_name, course=course_name,
                    branch=f" - {branch_name}"))
                return False
            await say("🤔 Reply with a branch *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # =============================================================
        #  STEP: filter  (semester + year, e.g. "6 2025" or "all")
        # =============================================================
        if step == "filter":
            if low == "back":
                # Go back one step (branch if there were branches, else course)
                if state.get("branches"):
                    state["step"] = "branch"
                    state["page"] = 0
                else:
                    state["step"] = "course"
                    state["page"] = 0
                await asyncio.to_thread(_save, phone, state)
                if state["step"] == "branch":
                    await _show_branches(phone, say, state.get("uni_name", ""),
                                         state.get("course_name", ""),
                                         state.get("course_url", ""),
                                         state.get("branches") or [], page=0)
                else:
                    await _show_courses(phone, say, state.get("uni_name", ""),
                                        state.get("uni_url", ""),
                                        state.get("courses") or [], page=0)
                return False

            # Parse "6 2025", "all", "6 all", "all 2025", "6"
            tokens = text.split()
            semester, year = None, None
            for t in tokens:
                if t.lower() == "all":
                    continue
                if t.isdigit():
                    n = int(t)
                    if 1900 < n < 2100:
                        year = n
                    elif 1 <= n <= 12:
                        semester = n
            branch_url = state.get("branch_url") or state.get("course_url", "")
            branch_name = state.get("branch_name", "")
            uni_name = state.get("uni_name", "")
            course_name = state.get("course_name", "")

            if not _quota_ok(phone):
                await asyncio.to_thread(_save, phone, {})
                await say(f"⏳ You've reached today's limit of {MAX_PER_DAY} paper batches. "
                          f"Please try again tomorrow.")
                return True

            await say(f"🔎 Fetching papers (sem={semester or 'ALL'}, year={year or 'ALL'}) ...")
            async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
                papers = await _fetch_papers(client, branch_url, semester=semester, year=year)

            if not papers:
                _refund(phone)
                await say("😕 No papers matched that filter. Try widening it (e.g. just `6`, "
                          "or just `2025`), or *back* / *exit*.")
                return False

            state["step"] = "papers"
            state["papers"] = papers
            state["page"] = 0
            state["filter"] = {"sem": semester, "year": year}
            await asyncio.to_thread(_save, phone, state)
            await _show_papers(phone, say, papers, page=0)
            return False

        # =============================================================
        #  STEP: papers  (pick number, 'all', or paginate)
        # =============================================================
        if step == "papers":
            papers = state.get("papers") or []
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_papers(phone, say, papers, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_papers(phone, say, papers, page=page)
                return False
            if low == "back":
                state["step"] = "filter"
                state.pop("papers", None)
                state["page"] = 0
                await asyncio.to_thread(_save, phone, state)
                await say(_FILTER_PROMPT.format(
                    uni=state.get("uni_name", ""),
                    course=state.get("course_name", ""),
                    branch=f" - {state.get('branch_name', '')}" if state.get("branch_name") else ""))
                return False

            picked = _parse_paper_selection(text, len(papers))
            if picked is None:                          # 'all'
                to_send = papers
            elif picked:
                to_send = [papers[i] for i in picked]
            else:
                await say(f"🤔 Reply with a paper *number* (1-{len(papers)}), "
                          f"a range like *1-3*, *all*, or *next* / *back* / *exit*.")
                return False

            await say(f"📥 Sending *{len(to_send)}* paper(s) ...")
            delivered = await _deliver_papers(phone, say, to_send)
            await asyncio.to_thread(_save, phone, {})
            _log(phone, ok=True, count=delivered,
                 uni=state.get("uni_name", ""),
                 course=state.get("course_name", ""),
                 branch=state.get("branch_name", ""),
                 filter=state.get("filter"))
            await say(f"✅ Done! Sent *{delivered}/{len(to_send)}* paper(s). "
                      f"Send */pyq* for another, or */menu* for the menu.")
            return True

        # =============================================================
        #  STEP: company_search  (user types a company name)
        # =============================================================
        if step == "company_search":
            if low == "back":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
                await say(PROMPT)
                return False
            if not text or len(text) < 2:
                await say("🤔 Please type a company name (e.g. *TCS*, *Infosys*, "
                          "*Wipro*), or *back* / *exit*.")
                return False
            if not _quota_ok(phone):
                await asyncio.to_thread(_save, phone, {})
                await say(f"⏳ You've reached today's limit of {MAX_PER_DAY} "
                          f"PYQ batches. Please try again tomorrow.")
                return True
            await _company_fetch_and_show(phone, say, text)
            return False

        # =============================================================
        #  STEP: company_pick  (multiple company matches)
        # =============================================================
        if step == "company_pick":
            matches = state.get("matches") or []
            if low == "back":
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "company_search"})
                await say(_COMPANY_PROMPT)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(matches):
                    await say(f"🤔 Reply 1-{len(matches)}, or *back* / *exit*.")
                    return False
                company_name, slug = matches[n - 1]
                await say(f"🔎 Fetching papers for *{company_name}* ...")
                async with httpx.AsyncClient(follow_redirects=True,
                                              headers=HEADERS) as client:
                    papers = await _indiabix_list_company_papers(client, slug)
                if not papers:
                    await say(f"😕 No papers found for *{company_name}*. "
                              f"Try another, or *back* / *exit*.")
                    return False
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "company_papers",
                    "company_name": company_name,
                    "company_slug": slug,
                    "papers": papers,
                    "page": 0,
                })
                await _show_company_papers_list(phone, say, company_name, papers, page=0)
                return False
            await say(f"🤔 Reply 1-{len(matches)}, or *back* / *exit*.")
            return False

        # =============================================================
        #  STEP: company_papers  (paginated list, pick a paper or 'all')
        # =============================================================
        if step == "company_papers":
            papers = state.get("papers") or []
            company_name = state.get("company_name", "")
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_company_papers_list(phone, say, company_name, papers, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_company_papers_list(phone, say, company_name, papers, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "company_search"})
                await say(_COMPANY_PROMPT)
                return False
            picked = _parse_paper_selection(text, len(papers))
            if picked is None:                  # 'all'
                picked = list(range(len(papers)))
            if not picked:
                await say(f"🤔 Reply with a paper *number* (1-{len(papers)}), "
                          f"a range like *1-3*, *all*, or *next* / *back* / *exit*.")
                return False
            await _company_deliver_papers(phone, say, company_name, papers, picked)
            # Stay in company_papers so user can pick another
            return False

        # =============================================================
        #  STEP: interview_menu  (3 sub-sections: HR / GD / Technical)
        # =============================================================
        if step == "interview_menu":
            if low == "back":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
                await say(PROMPT)
                return False
            if text == "1":
                await say("🔎 Loading HR interview questions ...")
                async with httpx.AsyncClient(follow_redirects=True,
                                              headers=HEADERS) as client:
                    items = await _indiabix_list_hr_questions(client)
                if not items:
                    await say("⚠️ Couldn't load HR questions. Try again, or *back*.")
                    return False
                # Save complete state in one shot (no reload needed)
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "hr_list", "page": 0,
                    "items": items,
                })
                await _show_hr_questions(phone, say, items, page=0)
                return False
            if text == "2":
                await say("🔎 Loading GD topics ...")
                async with httpx.AsyncClient(follow_redirects=True,
                                              headers=HEADERS) as client:
                    items = await _indiabix_list_gd_topics(client)
                if not items:
                    await say("⚠️ Couldn't load GD topics. Try again, or *back*.")
                    return False
                # Save complete state in one shot (no reload needed)
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "gd_list", "page": 0,
                    "items": items,
                })
                await _show_gd_topics(phone, say, items, page=0)
                return False
            if text == "3":
                await say("🔎 Loading technical interview topics ...")
                async with httpx.AsyncClient(follow_redirects=True,
                                              headers=HEADERS) as client:
                    items = await _indiabix_list_tech_topics(client)
                if not items:
                    await say("⚠️ Couldn't load tech topics. Try again, or *back*.")
                    return False
                # Save complete state in one shot (no reload needed)
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "tech_list", "page": 0,
                    "items": items,
                })
                await _show_tech_topics(phone, say, items, page=0)
                return False
            await say("🤔 Reply with *1* (HR), *2* (GD), *3* (Technical), "
                      "or *back* / *exit*.")
            return False

        # =============================================================
        #  STEP: hr_list  (paginated HR questions)
        # =============================================================
        if step == "hr_list":
            items = state.get("items") or []
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_hr_questions(phone, say, items, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_hr_questions(phone, say, items, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "interview_menu"})
                await _show_interview_menu(phone, say)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(items):
                    await say(f"🤔 Reply 1-{len(items)}, or *next* / *back* / *exit*.")
                    return False
                if not _quota_ok(phone):
                    await asyncio.to_thread(_save, phone, {})
                    await say(f"⏳ Daily limit reached ({MAX_PER_DAY}). "
                              f"Try again tomorrow.")
                    return True
                q_text, q_url = items[n - 1]
                await _hr_fetch_and_deliver(phone, say, q_text, q_url)
                # Stay in hr_list for another pick
                return False
            await say(f"🤔 Reply with a *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # =============================================================
        #  STEP: gd_list  (paginated GD topics)
        # =============================================================
        if step == "gd_list":
            items = state.get("items") or []
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_gd_topics(phone, say, items, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_gd_topics(phone, say, items, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "interview_menu"})
                await _show_interview_menu(phone, say)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(items):
                    await say(f"🤔 Reply 1-{len(items)}, or *next* / *back* / *exit*.")
                    return False
                if not _quota_ok(phone):
                    await asyncio.to_thread(_save, phone, {})
                    await say(f"⏳ Daily limit reached ({MAX_PER_DAY}).")
                    return True
                t_text, t_url = items[n - 1]
                await _gd_fetch_and_deliver(phone, say, t_text, t_url)
                return False
            await say(f"🤔 Reply with a *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # =============================================================
        #  STEP: tech_list  (paginated tech topics: Python, C, Java ...)
        # =============================================================
        if step == "tech_list":
            items = state.get("items") or []
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_tech_topics(phone, say, items, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_tech_topics(phone, say, items, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "interview_menu"})
                await _show_interview_menu(phone, say)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(items):
                    await say(f"🤔 Reply 1-{len(items)}, or *next* / *back* / *exit*.")
                    return False
                topic_name, topic_url = items[n - 1]
                await say(f"🔎 Loading sub-topics for *{topic_name}* ...")
                async with httpx.AsyncClient(follow_redirects=True,
                                              headers=HEADERS) as client:
                    subtopics = await _indiabix_list_tech_subtopics(client, topic_url)
                if not subtopics:
                    await say(f"⚠️ No sub-topics for *{topic_name}*. Try another, or *back*.")
                    return False
                # Preserve items list so we can restore it on "back"
                await asyncio.to_thread(_save, phone, {
                    "mode": MODE, "step": "tech_subtopics",
                    "topic_name": topic_name,
                    "topic_url": topic_url,
                    "subtopics": subtopics,
                    "items": items,  # for "back" navigation
                    "page": 0,
                })
                await _show_tech_subtopics(phone, say, topic_name, subtopics, page=0)
                return False
            await say(f"🤔 Reply with a *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # =============================================================
        #  STEP: tech_subtopics  (paginated sub-topics; pick -> PDF)
        # =============================================================
        if step == "tech_subtopics":
            subtopics = state.get("subtopics") or []
            topic_name = state.get("topic_name", "")
            page = state.get("page", 0)
            if low == "next":
                page += 1
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_tech_subtopics(phone, say, topic_name, subtopics, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                state["page"] = page
                await asyncio.to_thread(_save, phone, state)
                await _show_tech_subtopics(phone, say, topic_name, subtopics, page=page)
                return False
            if low == "back":
                # Restore tech_list items
                await asyncio.to_thread(_save, phone,
                    {"mode": MODE, "step": "tech_list", "page": 0,
                     "items": state.get("items", [])})
                await _show_tech_topics(phone, say, state.get("items", []), page=0)
                return False
            if text.isdigit():
                n = int(text)
                if not 1 <= n <= len(subtopics):
                    await say(f"🤔 Reply 1-{len(subtopics)}, or *next* / *back* / *exit*.")
                    return False
                if not _quota_ok(phone):
                    await asyncio.to_thread(_save, phone, {})
                    await say(f"⏳ Daily limit reached ({MAX_PER_DAY}).")
                    return True
                sub_name, sub_url = subtopics[n - 1]
                await _tech_fetch_and_deliver(phone, say, topic_name, sub_name, sub_url)
                return False
            await say(f"🤔 Reply with a *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # Unknown step -> reset
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
        await say(PROMPT)
        return False

    except Exception:
        _refund(phone)
        log.exception("pyq.handle failed")
        await asyncio.to_thread(_save, phone, {})
        await say("⚠️ Something went wrong while fetching papers. Please try again in a few minutes.")
        return True


# =====================================================================
#  CLI / COLAB STANDALONE MODE
#  (when run directly, not imported by the bot)
# =====================================================================
async def _cli(argv) -> None:
    """Interactive CLI / Colab mode (no WhatsApp, no database)."""
    if len(argv) >= 3 and argv[1] == "probe":
        # Debug a university name
        async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
            items = await _fetch_university_list(client)
            matches = _search_universities(items, argv[2], limit=5)
            print(f"Top {len(matches)} matches for '{argv[2]}':")
            for i, it in enumerate(matches, 1):
                print(f"  {i}. {it['description'][:80]}  ->  {it['url']}")
            if matches:
                uni_url = matches[0]["url"]
                if not uni_url.startswith("http"):
                    uni_url = "https://" + uni_url
                courses = await _list_courses(client, uni_url)
                print(f"\nCourses on {uni_url}:")
                for n, u in courses:
                    print(f"  - {n:12}  ->  {u}")
        return

    # Full interactive flow
    print("=" * 60)
    print("   Previous-Year-Question Scraper (CLI)")
    print("=" * 60)

    async with httpx.AsyncClient(follow_redirects=True, headers=HEADERS) as client:
        # --- University ---
        items = await _fetch_university_list(client)
        print(f"\nFound {len(items)} universities.")
        q = input("\nEnter university name (e.g. BPUT / AKTU / BHU): ").strip()
        if not q:
            return
        matches = _search_universities(items, q, limit=8)
        if not matches:
            print("No match.")
            return
        print("\nClosest matches:")
        for i, it in enumerate(matches, 1):
            print(f"  {i}. {it['description'][:90]}  ->  {it['url']}")
        c = input("\nPick number: ").strip()
        if not c.isdigit() or not 1 <= int(c) <= len(matches):
            return
        uni = matches[int(c) - 1]
        uni_url = uni["url"] if uni["url"].startswith("http") else "https://" + uni["url"]
        uni_name = uni["description"][:60]
        print(f"\n>> {uni_name}\n   {uni_url}")

        # --- Course ---
        courses = await _list_courses(client, uni_url)
        if not courses:
            print("No courses detected.")
            return
        print("\nCourses:")
        for i, (n, _) in enumerate(courses, 1):
            print(f"  {i}. {n}")
        c = input("\nPick course number: ").strip()
        if not c.isdigit() or not 1 <= int(c) <= len(courses):
            return
        course_name, course_url = courses[int(c) - 1]
        print(f"\n>> {course_name}\n   {course_url}")

        # --- Branch ---
        branches = await _list_branches(client, course_url)
        if branches:
            print("\nBranches:")
            for i, (n, _) in enumerate(branches, 1):
                print(f"  {i}. {n}")
            c = input("\nPick branch number (or blank for first-year): ").strip()
            if c.isdigit() and 1 <= int(c) <= len(branches):
                branch_name, branch_url = branches[int(c) - 1]
            else:
                branch_name, branch_url = branches[0]
        else:
            print("\n(no branch subdivision — using course page)")
            branch_name, branch_url = "", course_url
        print(f"\n>> Branch: {branch_name or '(none)'}\n   {branch_url}")

        # --- Filter ---
        sem_raw = input("\nSemester number (blank=ALL): ").strip()
        year_raw = input("Year (blank=ALL): ").strip()
        semester = int(sem_raw) if sem_raw.isdigit() else None
        year = int(year_raw) if year_raw.isdigit() else None

        papers = await _fetch_papers(client, branch_url, semester=semester, year=year)
        if not papers:
            print("\nNo papers matched.")
            return
        print(f"\nMatched {len(papers)} paper(s):")
        for i, p in enumerate(papers, 1):
            print(f"  {i:>3}. sem {p['sem']}, {p['year']}  |  {p['name']}")

        # --- Download ---
        mode = input("\nDownload: 1=ALL  2=pick  3=abort [1]: ").strip() or "1"
        if mode == "3":
            return
        if mode == "1":
            to_dl = papers
        else:
            sel = input("Numbers (e.g. 1,3,5 or 1-4): ").strip()
            picked = _parse_paper_selection(sel, len(papers))
            if not picked:
                print("No valid selection.")
                return
            to_dl = [papers[i] for i in picked]

        out_dir = "/content/pyq_papers" if os.path.isdir("/content") else "./pyq_papers"
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for i, p in enumerate(to_dl, 1):
            fname = p["name"] + ".pdf"
            fpath = os.path.join(out_dir, fname)
            print(f"  [{i}/{len(to_dl)}] {fname} ... ", end="", flush=True)
            data, why = await _download_paper_bytes(client, p)
            if data:
                with open(fpath, "wb") as f:
                    f.write(data)
                print(f"OK ({len(data)/1024:.0f} KB)")
                saved.append(fpath)
            else:
                print(f"FAIL ({why})  ->  {p['pdf_url']}")

        if saved:
            zip_path = os.path.join(os.path.dirname(out_dir), "pyq_papers.zip")
            if os.path.isdir("/content"):
                zip_path = "/content/pyq_papers.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for f in saved:
                    z.write(f, os.path.basename(f))
            print(f"\nZipped {len(saved)} file(s) -> {zip_path}")
        print("\nDONE.")


if __name__ == "__main__":                          # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_cli(sys.argv))
