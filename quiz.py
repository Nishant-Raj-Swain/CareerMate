# quiz.py
# ---------------------------------------------------------------------------
# Topic quizzes for the WhatsApp bot (same job as quiz_service.py, same style:
# async, httpx + BeautifulSoup, returns [{"question", "options", "correct"}]).
#
#   /practice            ->  categories (Aptitude, English, Python, ...)
#   user picks one       ->  its sub-topics (Average, Interest, ...)
#   user picks one       ->  20 random questions, played by main.py's normal quiz handler
#
# Drop-in for quiz_service.fetch_indiabix_questions:
#       questions = await quiz.fetch_quiz_questions(topic="average", total_needed=20)
#   topic = a topic name ("average"), a category ("aptitude") or one of quiz_service's keys
#   ("python", "c", "java", "sql", "networking", "reasoning", "logical", ...). Returns [] on failure,
#   exactly like quiz_service, so main.py's existing "couldn't load questions" branch works.
#
# Menu flow used by main.py (state is stored in users.state like every other mode):
#       msgs, questions, topic = await quiz.practice_start(phone, arg)    # /practice [arg]
#       msgs, questions, topic = await quiz.practice_reply(phone, text)   # while state mode == quiz.MODE
#   questions is None until a topic is chosen; then main.py builds the standard quiz_state and calls
#   send_next_question (see _practice_output in main.py).
#
# How questions are collected
#   A topic page has up to 5 sections x 10 pages x 10 questions (?section=N&page=M). Each quiz picks a few
#   random (section, page) pages, pools their questions, samples 20 and shuffles question order and, where
#   safe, option order, so two quizzes on one topic are almost never the same.
#   Questions without a readable answer are skipped (quiz_service defaults them to option A; this does not).
#
# Being a good citizen (please keep these)
#   - robots.txt is fetched and obeyed; one request at a time, QUIZ_DELAY seconds apart; pages cached
#     for QUIZ_CACHE_HOURS. The source site's content is theirs: check their terms before production use.
#
# The parser does not depend on CSS class names (it anchors on the "Discuss" / "View Answer" controls).
# If the site changes, check it with:   python quiz.py probe "<topic page url>"
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import random
import re
import time
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("quiz")

BASE = "https://www.examveda.com"          # the only place the source site is named
HEADERS = {
    "User-Agent": os.getenv("QUIZ_UA") or "CareerMateBot/1.0 (educational WhatsApp bot; respects robots.txt)",
    "Accept": "text/html,application/xhtml+xml",
}
DELAY = float(os.getenv("QUIZ_DELAY", "1.2"))                  # seconds between requests
CACHE_SECONDS = float(os.getenv("QUIZ_CACHE_HOURS", "12")) * 3600
PAGES_PER_QUIZ = 3                                              # random pages pooled per quiz (10 questions each)
MAX_EXTRA_PAGES = 3                                             # more pages if the pool is too small
MAX_OPTIONS = 4                                                 # main.py's quiz handler accepts A-D
MODE = "practice_menu"
WA_LIMIT = 3800

# Menu names from the site's "Practice MCQs" dropdown. Their URLs are discovered from the home page.
CATEGORY_NAMES = [
    "Aptitude", "English", "Verbal Reasoning", "Non Verbal Reasoning", "General Knowledge(GK)",
    "General Science(GS)", "Data Interpretation(DI)", "Banking Awareness", "Current Affairs",
    "Computer Fundamentals", "Networking", "SQL", "Database", "C Programming", "C++ Programming",
    "Java", "Python", "Commerce", "Management", "Law", "Political Science", "Psychology", "Agriculture",
]
CATEGORY_FALLBACK = {"Aptitude": "/mcq-question-on-arithmetic-ability/"}
# the topic keys quiz_service.py already understands -> a category here
ALIASES = {"c": "C Programming", "cpp": "C++ Programming", "sql": "SQL", "reasoning": "Verbal Reasoning",
           "logical": "Verbal Reasoning", "aptitude": "Aptitude", "networking": "Networking", "java": "Java",
           "python": "Python"}

LEAF_RE = re.compile(r"/practice-mcq-question-on-[^/]+/?", re.I)       # a topic that has questions
CATEGORY_RE = re.compile(r"/mcq-question(?:s)?-on-[^/]+/?", re.I)      # a page that lists topics


class ScrapeError(Exception):
    pass


# ------------------------------------------------------------------ HTTP layer
_cache: dict[str, tuple[float, object]] = {}
_robots: dict = {"at": 0.0, "disallow": []}
_lock: asyncio.Lock | None = None
_last_request = 0.0


def _cached(key: str):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    return None


def _store(key: str, value):
    _cache[key] = (time.time(), value)
    if len(_cache) > 400:                                   # crude bound: drop the oldest 100
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
            _cache.pop(k, None)
    return value


def _same_site(netloc: str) -> bool:
    host = urlparse(BASE).netloc
    return netloc == host or netloc.endswith("." + host.replace("www.", ""))


async def _load_robots(client: httpx.AsyncClient) -> list[str]:
    if _robots["disallow"] and time.time() - _robots["at"] < 86400:
        return _robots["disallow"]
    rules, applies = [], False
    try:
        r = await client.get(BASE + "/robots.txt")
        if r.status_code == 200:
            for line in r.text.splitlines():
                line = line.split("#")[0].strip()
                if not line or ":" not in line:
                    continue
                key, val = [x.strip() for x in line.split(":", 1)]
                if key.lower() == "user-agent":
                    applies = val == "*"
                elif applies and key.lower() == "disallow" and val:
                    rules.append(val)                       # conservative: any Disallow match blocks
    except httpx.HTTPError as e:
        log.warning("robots.txt unavailable (%s); using defaults", e)
    _robots.update(at=time.time(), disallow=rules or ["/admin/", "/account/"])
    return _robots["disallow"]


async def _get(url: str) -> str:
    """Polite GET: robots-checked, one at a time, spaced out, cached. Returns HTML."""
    global _lock, _last_request
    url = urldefrag(url)[0]
    hit = _cached("html:" + url)
    if hit is not None:
        return hit
    _lock = _lock or asyncio.Lock()
    last = None
    async with _lock:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers=HEADERS) as client:
            p = urlparse(url)
            if p.netloc and not _same_site(p.netloc):
                raise ScrapeError(f"not the quiz site: {url}")
            if any((p.path or "/").startswith(rule) for rule in await _load_robots(client)):
                raise ScrapeError(f"blocked by robots.txt: {url}")
            for attempt in range(3):
                wait = DELAY + random.uniform(0, 0.4) - (time.time() - _last_request)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    r = await client.get(url)
                except httpx.HTTPError as e:
                    last = str(e); _last_request = time.time(); continue
                _last_request = time.time()
                if r.status_code == 200:
                    return _store("html:" + url, r.text)
                if r.status_code in (403, 429) or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"; await asyncio.sleep(2 * (attempt + 1)); continue
                raise ScrapeError(f"HTTP {r.status_code} for {url}")
    raise ScrapeError(f"could not fetch {url}: {last}")


def allowed(url: str) -> bool:
    """True if robots.txt (as last loaded) does not disallow this URL."""
    p = urlparse(url)
    if p.netloc and not _same_site(p.netloc):
        return False
    return not any((p.path or "/").startswith(r) for r in _robots["disallow"])


# --------------------------------------------------------------- text cleaning
_LATEX = [
    (r"\\eqalign\s*\{", ""), (r"\\left|\\right", ""), (r"\\,|\\;|\\!|\\ ", " "), (r"\\%", "%"),
    (r"\\times", "×"), (r"\\div", "÷"), (r"\\cdot", "·"), (r"\\pm", "±"), (r"\\pi", "π"), (r"\\infty", "∞"),
    (r"\\leq?\b", "≤"), (r"\\geq?\b", "≥"), (r"\\neq?\b", "≠"), (r"\\approx", "≈"), (r"\\therefore", "∴"),
    (r"\\Rightarrow|\\implies", "⇒"), (r"\\rightarrow|\\to\b", "→"), (r"\\degree|\^\{?\\circ\}?", "°"),
    (r"\\text\s*\{([^{}]*)\}", r"\1"), (r"\\mathrm\s*\{([^{}]*)\}", r"\1"),
    (r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)"), (r"\\sqrt\s*(\w)", r"√\1"),
    (r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1/\2)"),
    (r"\^\{(\d+)\}", lambda m: m.group(1).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))),
    (r"\^(\d)", lambda m: m.group(1).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))),
    (r"_\{([^{}]*)\}", r"_\1"),
]


def clean(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\xa0", " ")
    t = re.sub(r"\$\$|\\\(|\\\)|\\\[|\\\]", " ", t)
    for _ in range(2):                                       # twice, for nested fractions etc.
        for pat, rep in _LATEX:
            t = re.sub(pat, rep, t)
    t = t.replace("\\\\", "\n").replace("&", " ").replace("{", "").replace("}", "")
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)                   # leftover \command -> command
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t.strip()


# ---------------------------------------------------------------- HTML parsing
_NAV_HINT = re.compile(r"(nav|menu|footer|header|breadcrumb|sidebar|dropdown|social)", re.I)


def _in_chrome(tag) -> bool:
    """True if the link sits in site chrome (menus, header, footer, breadcrumbs)."""
    for p in tag.parents:
        if getattr(p, "name", None) in ("nav", "header", "footer"):
            return True
        cls = " ".join(p.get("class", [])) if hasattr(p, "get") else ""
        if cls and _NAV_HINT.search(cls) or (hasattr(p, "get") and _NAV_HINT.search(p.get("id") or "")):
            return True
    return False


def _name_from_slug(href: str) -> str:
    slug = urlparse(href).path.strip("/").split("/")[-1]
    slug = re.sub(r"^(practice-)?mcq-questions?-on-", "", slug)
    return slug.replace("-", " ").title()


def _link_name(a, href: str) -> str:
    for s in a.stripped_strings:
        if not re.search(r"practice mcq|questions and answers?|view all|^\d+$", s, re.I) and 1 < len(s) < 80:
            return clean(s)
    return _name_from_slug(href)


def parse_categories(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    wanted = {re.sub(r"\s+", "", n).lower(): n for n in CATEGORY_NAMES}
    found: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:")):
            continue
        key = re.sub(r"\s+", "", a.get_text(" ", strip=True)).lower()
        if key in wanted and wanted[key] not in found:
            found[wanted[key]] = urljoin(BASE, href)
    return [{"name": n, "url": found[n], "kind": "category"} for n in CATEGORY_NAMES if n in found]


def parse_children(html: str, page_url: str) -> dict:
    """Items listed on a category page: leaf topics (have questions) or deeper categories."""
    soup = BeautifulSoup(html, "html.parser")
    leaves, cats, seen = [], [], {page_url.rstrip("/")}
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        pure = urldefrag(href)[0].split("?")[0]
        if urlparse(pure).netloc and not _same_site(urlparse(pure).netloc):
            continue
        if pure.rstrip("/") in seen:
            continue
        if LEAF_RE.search(urlparse(pure).path):
            seen.add(pure.rstrip("/"))
            leaves.append({"name": _link_name(a, pure), "url": pure, "kind": "topic"})
        elif CATEGORY_RE.search(urlparse(pure).path) and not _in_chrome(a):
            seen.add(pure.rstrip("/"))
            cats.append({"name": _link_name(a, pure), "url": pure, "kind": "category"})
    if leaves:
        return {"kind": "topics", "items": leaves}
    return {"kind": "categories", "items": cats}


def list_pagination(html: str, page_url: str) -> list[str]:
    """Other '?page=N' pages of a topic LIST (some category pages are paginated)."""
    soup = BeautifulSoup(html, "html.parser")
    base = urldefrag(page_url)[0].split("?")[0]
    out = {}
    for a in soup.find_all("a", href=True):
        u = urljoin(page_url, a["href"])
        q = parse_qs(urlparse(u).query)
        if u.split("?")[0].rstrip("/") == base.rstrip("/") and "page" in q and "section" not in q:
            out[int(q["page"][0]) if q["page"][0].isdigit() else 0] = u
    return [out[k] for k in sorted(out) if k > 1][:9]


def discover_nav(html: str) -> dict:
    """Which sections / pages a topic page offers (from its ?section= and ?page= links)."""
    sections, pages = {1}, {1}
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        q = parse_qs(urlparse(href.replace("&amp;", "&")).query)
        if "section" in q and q["section"][0].isdigit():
            sections.add(int(q["section"][0]))
        if "page" in q and q["page"][0].isdigit():
            pages.add(int(q["page"][0]))
    for m in re.finditer(r"records of (\d+) records", html):          # "showing 11 - 20 records of 100 records"
        pages.add(max(1, -(-int(m.group(1)) // 10)))
    return {"sections": sorted(sections), "max_page": max(pages)}


_LABEL = re.compile(r"^\(?([A-Ea-e])[\).:]$")
_LABEL_TEXT = re.compile(r"^\(?([A-Ea-e])[\).:]\s+(\S.*)$")
_STOP = {"view answer", "discuss", "save", "report", "share", "answer", "solution", "explanation"}
_ANSWER = re.compile(r"Answer\s*:?\s*(?:Option\s*)?\(?([A-Ea-e])\)?\b", re.I)


def _is_stop(s: str) -> bool:
    low = s.strip().lower().rstrip(":")
    return low in _STOP or low.startswith(("answer:", "answer :", "solution:", "explanation:"))


def _block_containers(soup):
    """One element per question: the smallest ancestor of each 'Discuss' control that also holds options."""
    markers = [a for a in soup.find_all(["a", "button", "span"])
               if a.get_text(strip=True).lower() in ("discuss", "view answer")]
    blocks, seen = [], set()
    for m in markers:
        node = m
        while node is not None and node.parent is not None:
            node = node.parent
            text = node.get_text("\n", strip=True)
            if re.search(r"(?m)^\(?A[\).:]", text) and re.search(r"(?m)^\(?B[\).:]", text):
                # stop at the smallest ancestor that has options; make sure it holds ONE question
                if len([x for x in node.find_all(["a", "button", "span"])
                        if x.get_text(strip=True).lower() == "discuss"]) <= 1:
                    if id(node) not in seen:
                        seen.add(id(node)); blocks.append(node)
                    break
    return blocks


def parse_block(node) -> dict | None:
    toks = [t for t in node.stripped_strings]
    i = 0
    if toks and re.fullmatch(r"\d{1,3}[\).:]?", toks[0]):          # leading question number
        i = 1
    q_parts = []
    while i < len(toks) and not (_LABEL.match(toks[i]) or _LABEL_TEXT.match(toks[i])) and not _is_stop(toks[i]):
        q_parts.append(toks[i]); i += 1
    options: dict[str, list[str]] = {}
    cur = None
    answer = None
    expl: list[str] = []
    in_expl = False
    while i < len(toks):
        t = toks[i]
        m_label, m_text = _LABEL.match(t), _LABEL_TEXT.match(t)
        low = t.lower().rstrip(":")
        if in_expl:
            expl.append(t)
        elif m_label and m_label.group(1).upper() not in options:
            cur = m_label.group(1).upper(); options[cur] = []
        elif m_text and m_text.group(1).upper() not in options and not low.startswith("answer"):
            cur = m_text.group(1).upper(); options[cur] = [m_text.group(2)]
        elif low in ("view answer", "discuss", "save", "report", "share"):
            cur = None
        elif low.startswith(("answer", "solution", "explanation")):
            m = _ANSWER.search(t) or (_ANSWER.search(t + " " + toks[i + 1]) if i + 1 < len(toks) else None)
            if m and answer is None:
                answer = m.group(1).upper()
                if not re.search(r"Answer\s*:?\s*(?:Option\s*)?\(?[A-Ea-e]", t, re.I):
                    i += 1
            if low.startswith(("solution", "explanation")):
                in_expl = True
                rest = t.split(":", 1)[1].strip() if ":" in t else ""
                if rest:
                    expl.append(rest)
            cur = None
        elif cur:
            options[cur].append(t)
        i += 1
    if len(options) < 2 or not q_parts:
        return None
    letters = sorted(options)
    if letters != [chr(65 + n) for n in range(len(letters))]:     # must be A, B, C, ... without gaps
        return None
    return {
        "question": clean(" ".join(q_parts)),
        "options": [clean(" ".join(options[l])) for l in letters],
        "answer_letter": answer,
        "explanation": clean(" ".join(expl))[:700],
    }


def parse_questions(html: str, source: str = "") -> list[dict]:
    """All usable questions on one topic page: question, options (<= MAX_OPTIONS), correct index."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for node in _block_containers(soup):
        q = parse_block(node)
        if not q or not q["answer_letter"]:
            continue                                              # no answer in the page -> cannot grade it
        idx = ord(q["answer_letter"]) - 65
        if idx >= len(q["options"]) or len(q["options"]) > MAX_OPTIONS or not q["question"] or not all(q["options"]):
            continue
        out.append({"question": q["question"], "options": q["options"], "correct": idx,
                    "explanation": q["explanation"], "source": source})
    return out

_FIXED_OPTION = re.compile(r"\b(all|none|both|neither)\b.*\b(above|these|them|of the|and)\b|^\s*(a|b|c|d)\s*(and|&|,)", re.I)


def shuffle_options(q: dict) -> dict:
    """Shuffles option order (and fixes 'correct') unless an option refers to the others."""
    if any(_FIXED_OPTION.search(o) for o in q["options"]):
        return q
    order = list(range(len(q["options"])))
    random.shuffle(order)
    return {**q, "options": [q["options"][i] for i in order], "correct": order.index(q["correct"])}

def _page_url(topic_url: str, section: int, page: int) -> str:
    base = topic_url.split("?")[0]
    return f"{base}?section={section}&page={page}"

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", s.lower()).strip()


def match_item(items: list[dict], query: str) -> dict | None:
    """Number ('3'), exact/partial name, or close spelling."""
    q = _norm(query)
    if not q:
        return None
    if q.isdigit() and 1 <= int(q) <= len(items):
        return items[int(q) - 1]
    names = [_norm(i["name"]) for i in items]
    for n, it in zip(names, items):
        if n == q:
            return it
    sub = [it for n, it in zip(names, items) if q in n or (len(n) > 3 and n in q)]
    if len(sub) == 1:
        return sub[0]
    if sub:
        return min(sub, key=lambda it: len(it["name"]))
    close = difflib.get_close_matches(q, names, n=1, cutoff=0.72)
    return items[names.index(close[0])] if close else None



# ------------------------------------------------------------------ high level
async def get_categories() -> list[dict]:
    hit = _cached("categories")
    if hit is not None:
        return hit
    items = []
    try:
        items = parse_categories(await _get(BASE + "/"))
    except ScrapeError as e:
        log.warning("categories: %s", e)
    have = {i["name"] for i in items}
    for n, path in CATEGORY_FALLBACK.items():
        if n not in have:
            items.append({"name": n, "url": BASE + path, "kind": "category"})
    return _store("categories", items) if items else items


async def get_children(url: str) -> dict:
    hit = _cached("children:" + url)
    if hit is not None:
        return hit
    html = await _get(url)
    res = parse_children(html, url)
    seen = {i["url"] for i in res["items"]}
    for extra in list_pagination(html, url):
        try:
            more = parse_children(await _get(extra), extra)
        except ScrapeError:
            break
        if more["kind"] != res["kind"]:
            break
        for it in more["items"]:
            if it["url"] not in seen:
                seen.add(it["url"]); res["items"].append(it)
    return _store("children:" + url, res) if res["items"] else res


async def get_questions(topic_url: str, total: int = 20, rng: random.Random | None = None) -> list[dict]:
    """`total` random, shuffled questions from one topic. Different every call."""
    rng = rng or random
    first_html = await _get(topic_url.split("?")[0])
    nav = discover_nav(first_html)
    tried, pool, seen_q = set(), [], set()

    async def take(section: int, page: int):
        if (section, page) in tried:
            return
        tried.add((section, page))
        url = _page_url(topic_url, section, page)
        try:
            qs = parse_questions(await _get(url), url)
        except ScrapeError as e:
            log.warning("page failed: %s", e)
            return
        for q in qs:
            key = re.sub(r"\W+", "", q["question"].lower())[:120]
            if key not in seen_q:
                seen_q.add(key); pool.append(q)

    options = [(s, p) for s in nav["sections"] for p in range(1, nav["max_page"] + 1)]
    rng.shuffle(options)
    for s, p in options[:PAGES_PER_QUIZ]:
        await take(s, p)
    extra = iter(options[PAGES_PER_QUIZ:PAGES_PER_QUIZ + MAX_EXTRA_PAGES])
    while len(pool) < total:
        nxt = next(extra, None)
        if nxt is None:
            break
        await take(*nxt)
    if not pool:                                     # last resort: the plain topic page
        await take(1, 1)
    rng.shuffle(pool)
    return [shuffle_options(q) if rng is random else q for q in pool[:total]]




def _public(q: dict) -> dict:
    """The shape quiz_service returns (+ explanation, which main.py ignores)."""
    return {"question": q["question"], "options": q["options"][:MAX_OPTIONS], "correct": q["correct"],
            "explanation": q.get("explanation", "")}


async def fetch_quiz_questions(topic: str = "python", total_needed: int = 20) -> list[dict]:
    """Same contract as quiz_service.fetch_indiabix_questions: list of dicts, [] on any failure."""
    try:
        cats = await get_categories()
        t = (topic or "python").strip().lower()
        t = ALIASES.get(t, t)
        topic_n = _norm(t)
        cat = match_item(cats, t) if cats else None
        # 1) a sub-topic name inside the best-guess category (Aptitude first)
        order = ([cat] if cat else []) + [c for c in cats if c["name"] == "Aptitude" and c is not cat]
        for c in order:
            kids = await get_children(c["url"])
            if kids["kind"] == "topics" and topic_n != _norm(c["name"]):
                hit = match_item(kids["items"], t)
                if hit:
                    return [_public(q) for q in await get_questions(hit["url"], total_needed)]
        # 2) the topic IS a category -> a random sub-topic of it
        if cat:
            kids = await get_children(cat["url"])
            while kids["kind"] == "categories" and kids["items"]:
                kids = await get_children(random.choice(kids["items"])["url"])
            if kids["kind"] == "topics" and kids["items"]:
                return [_public(q) for q in await get_questions(random.choice(kids["items"])["url"], total_needed)]
        return []
    except Exception as e:
        log.exception("quiz fetch failed: %s", e)
        return []


# --------------------------------------------------------- WhatsApp menu flow
def is_practice_state(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


def _menu_messages(title: str, items: list[dict], hint: str) -> list[str]:
    lines = [f"*{i}.* {it['name']}" for i, it in enumerate(items, 1)]
    msgs, cur = [], f"📚 *{title}*\n\n"
    for ln in lines:
        if len(cur) + len(ln) + 1 > WA_LIMIT:
            msgs.append(cur.rstrip()); cur = ""
        cur += ln + "\n"
    msgs.append((cur + f"\n{hint}").strip())
    return msgs


def _save(phone: str, state: dict):
    import database as db
    db.save_state(phone, state)


async def _show(phone: str, title: str, items: list[dict], kind: str, trail: list[str]):
    word = "topic" if kind == "topics" else "category"
    state = {"mode": MODE, "kind": kind, "title": title, "items": items, "trail": trail}
    await asyncio.to_thread(_save, phone, state)
    return _menu_messages(title, items, f"Reply with a *number* or the {word} name. Send *exit* to cancel.")


async def _open(phone: str, item: dict, trail: list[str]):
    """Opens a menu item: shows its children, or loads the quiz if it is a topic."""
    if item["kind"] == "topic":
        questions = [_public(q) for q in await get_questions(item["url"], 20)]
        if not questions:
            return (["❌ I couldn't load questions for that topic right now. Please pick another one or try again shortly."],
                    None, None)
        await asyncio.to_thread(_save, phone, {})
        return ([f"✅ Loaded *{len(questions)} random questions* on *{item['name']}*."], questions, item["name"].lower())
    kids = await get_children(item["url"])
    if not kids["items"]:
        return (["❌ I couldn't find any topics under that category. Please try another."], None, None)
    msgs = await _show(phone, " › ".join(trail + [item["name"]]), kids["items"], kids["kind"], trail + [item["name"]])
    return msgs, None, None


async def practice_start(phone: str, arg: str = ""):
    """/practice [category or topic]. Returns (messages, questions|None, topic|None)."""
    try:
        cats = await get_categories()
        if not cats:
            return (["❌ I couldn't load the practice menu right now. Please try again in a few minutes."], None, None)
        if arg.strip():
            key = ALIASES.get(arg.strip().lower(), arg.strip())
            item = match_item(cats, key)
            if item:
                return await _open(phone, item, [])
            questions = await fetch_quiz_questions(arg.strip(), 20)    # maybe a sub-topic name, e.g. "average"
            if questions:
                await asyncio.to_thread(_save, phone, {})
                return ([f"✅ Loaded *{len(questions)} random questions* on *{arg.strip().title()}*."],
                        questions, arg.strip().lower())
        msgs = await _show(phone, "Choose a category", cats, "categories", [])
        return msgs, None, None
    except Exception:
        log.exception("practice_start failed")
        return (["⚠️ Something went wrong loading the practice menu. Please try again."], None, None)


async def practice_reply(phone: str, text: str):
    """A reply while the user is inside the practice menu."""
    import database as db
    state = (await asyncio.to_thread(db.get_user, phone)).get("state") or {}
    text = (text or "").strip()
    if text.lower() in ("exit", "quit", "cancel", "stop", "/exit", "/cancel"):
        await asyncio.to_thread(_save, phone, {})
        return (["👋 Practice menu closed."], None, None)
    if text.lower() in ("menu", "back", "0"):
        cats = await get_categories()
        return (await _show(phone, "Choose a category", cats, "categories", []), None, None)
    items = state.get("items") or []
    item = match_item(items, text)
    if not item:
        return ([f"🤔 I didn't find that. Reply with a number from 1 to {len(items)}, a name from the list, "
                 "*back* for categories, or *exit*."], None, None)
    try:
        return await _open(phone, item, state.get("trail") or [])
    except Exception:
        log.exception("practice_reply failed")
        return (["⚠️ Something went wrong. Please try again."], None, None)


# --------------------------------------------------------------------- probe
async def _probe(url: str):  # pragma: no cover
    html = await _get(url)
    print("robots allowed:", allowed(url))
    kids = parse_children(html, url)
    print(f"children: {kids['kind']} x{len(kids['items'])}", [i["name"] for i in kids["items"][:8]])
    print("nav:", discover_nav(html))
    qs = parse_questions(html, url)
    print(f"questions parsed on this page: {len(qs)}")
    for q in qs[:3]:
        print(" -", q["question"][:90], "|", q["options"], "| correct:", "ABCDE"[q["correct"]])
    if not qs and LEAF_RE.search(url):
        print("No questions parsed. Save the page (Ctrl+S) and send it to the developer to adjust the parser.")


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "probe":
        asyncio.run(_probe(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "quiz":
        for n, q in enumerate(asyncio.run(get_questions(sys.argv[2])), 1):
            print(f"{n}. {q['question']}\n   {q['options']}  -> {'ABCDE'[q['correct']]}")
    else:
        print('usage: python quiz.py probe "<url>"  |  python quiz.py quiz "<topic url>"')
