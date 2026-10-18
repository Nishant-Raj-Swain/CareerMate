# job_browse.py
# ---------------------------------------------------------------------------
# /job_search  ->  browse fresher jobs in three ways (same menu as the source site's "Jobs" dropdown):
#       1. By Category      (IT / Software, Core Technical, MBA, ...)
#       2. By Qualification (BE / B.Tech, MCA, BCA, ...)
#       3. By City          (Bangalore, Pune, ...)
#   4. (existing) type a role  -> your ai_service.search_jobs
# The user picks a section, then an item (number or name); the bot lists the jobs 10 at a time
# (title, company, location, experience, salary, posted date + the link to apply on the source site).
#
# Public API used by main.py (state lives in users.state like every other mode):
#       msgs, want_role = await job_browse.start(phone, notify)     # /job_search with no argument
#       msgs, want_role = await job_browse.reply(phone, text, notify)   # while state mode == job_browse.MODE
#   want_role is True when the user chose "type a role": main.py then switches to its existing job_search mode.
#
# How it works
#   - The menu links are DISCOVERED from the site's home page at runtime (the site uses numeric ids in its URLs,
#     so nothing is hardcoded) and cached for 12 h.
#   - A listing page is parsed from the visible text/links, not from CSS class names.
#   - Paging uses the site's own ?limit=30&offset=N.
#
# Being a good citizen (please keep these)
#   - robots.txt is fetched and obeyed; one request at a time, JOBS_DELAY seconds apart; listing pages cached
#     JOBS_CACHE_MINUTES. We show only a short summary and always link back to the source job page.
#   - The site's content is theirs: check their terms before production use. Their menu/URLs may change; check with
#         python job_browse.py menu            (what was discovered)
#         python job_browse.py jobs "<listing page url>"   (what was parsed)
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

log = logging.getLogger("job_browse")

BASE = os.getenv("JOBS_BASE", "https://www.freshersworld.com").rstrip("/")
HEADERS = {
    "User-Agent": os.getenv("JOBS_UA") or "CareerMateBot/1.0 (educational WhatsApp bot; respects robots.txt)",
    "Accept": "text/html,application/xhtml+xml",
}
DELAY = float(os.getenv("JOBS_DELAY", "1.2"))
MENU_CACHE = 12 * 3600
PAGE_CACHE = float(os.getenv("JOBS_CACHE_MINUTES", "30")) * 60
PAGE_LIMIT = 30                 # the site's own page size (?limit=30&offset=N)
SHOW = 10                       # jobs per WhatsApp message batch
WA_LIMIT = 3800
MODE = "job_browse"
EXIT_WORDS = {"exit", "quit", "cancel", "stop", "/exit", "/cancel"}

SECTIONS = [
    {"key": "category", "name": "By Category", "emoji": "💼",
     "heading": re.compile(r"jobs?\s*by\s*categor", re.I)},
    {"key": "qualification", "name": "By Qualification", "emoji": "🎓",
     "heading": re.compile(r"jobs?\s*by\s*qualification", re.I)},
    {"key": "city", "name": "By City", "emoji": "🏙️",
     "heading": re.compile(r"jobs?\s*by\s*(city|location)", re.I)},
]

# Labels seen in the site's menu; only used when the headings can't be found in the page (JS-rendered menu).
KNOWN = {
    "category": {"it / software", "core technical", "mba jobs", "internship jobs", "diploma jobs", "research", "bpo jobs",
                 "part time jobs", "health care", "hospitality", "retail", "journalism", "quality control", "sports",
                 "finance", "bank jobs", "walk-ins", "teaching", "pharma jobs", "startup jobs", "tech support"},
    "qualification": {"be / b.tech", "mca", "mba / pgdm", "me / m.tech", "m.sc jobs", "b.com jobs", "bba / bbm jobs",
                      "bca jobs", "bsc jobs", "diploma jobs"},
    "city": {"ahmedabad", "bangalore", "chennai", "delhi / ncr", "hyderabad", "kolkata", "mumbai", "pune", "gurgaon",
             "jaipur", "mangalore", "mysore", "visakhapatnam", "orissa", "bhopal", "mohali", "kanpur", "allahabad",
             "roorkee", "noida", "punjab", "chandigarh", "kochi", "kharagpur", "ranchi", "nasik", "goa"},
}
_SKIP_LABEL = re.compile(r"(login|log in|register|sign ?up|post a (job|free)|employer|premium|sales|contact|about|"
                         r"privacy|terms|download|app store|google play|^other\b|\.\.\.$|…$)", re.I)


class JobsError(Exception):
    pass


# ----------------------------------------------------------------------- polite HTTP
_lock: asyncio.Lock | None = None
_last_request = 0.0
_robots = {"at": 0.0, "disallow": []}
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    return None


def _store(key: str, value):
    if len(_cache) > 300:
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
            _cache.pop(k, None)
    _cache[key] = (time.time(), value)
    return value


def _same_site(netloc: str) -> bool:
    host = urlparse(BASE).netloc
    return netloc == host or netloc.endswith("." + host.replace("www.", ""))


async def _load_robots(client: httpx.AsyncClient) -> list[str]:
    if _robots["at"] and time.time() - _robots["at"] < 86400:
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


def _blocked(path_q: str, rules: list[str]) -> bool:
    return any(rule and path_q.startswith(rule) for rule in rules)


async def _get(url: str, ttl: float) -> str:
    """Polite GET: robots-checked, one at a time, spaced out, cached. Returns HTML."""
    global _lock, _last_request
    url = urldefrag(url)[0]
    hit = _cached("html:" + url, ttl)
    if hit is not None:
        return hit
    _lock = _lock or asyncio.Lock()
    last = None
    async with _lock:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, headers=HEADERS) as client:
            p = urlparse(url)
            if p.netloc and not _same_site(p.netloc):
                raise JobsError(f"not the jobs site: {url}")
            rules = await _load_robots(client)
            if _blocked((p.path or "/") + (("?" + p.query) if p.query else ""), rules):
                raise JobsError(f"blocked by robots.txt: {url}")
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
                    if re.search(r"Just a moment\.\.\.|cf-browser-verification|Attention Required|cf-chl", r.text[:6000]):
                        raise JobsError(f"bot-protection page returned for {url}")
                    return _store("html:" + url, r.text)
                if r.status_code in (403, 429) or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"; await asyncio.sleep(2 * (attempt + 1)); continue
                raise JobsError(f"HTTP {r.status_code} for {url}")
    raise JobsError(f"could not fetch {url}: {last}")


# ----------------------------------------------------------------------- menu discovery
def _label(a: Tag) -> str:
    return re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def discover_menu(html: str, page_url: str = "") -> dict[str, list[dict]]:
    """{section key: [{"name", "url"}]} from the home page's Jobs dropdown.
    Walks the page in document order: links after a 'Job By Category' / 'Jobs by Qualification' / 'Job By City'
    heading belong to that section until the next heading."""
    soup = BeautifulSoup(html, "lxml")
    out: dict[str, list[dict]] = {s["key"]: [] for s in SECTIONS}
    seen: dict[str, set] = {s["key"]: set() for s in SECTIONS}
    current = None
    for node in soup.descendants:
        if isinstance(node, NavigableString):
            t = str(node).strip()
            if 3 <= len(t) <= 40:
                for s in SECTIONS:
                    if s["heading"].search(t):
                        current = s["key"]
                        break
            continue
        if not isinstance(node, Tag) or node.name != "a" or not current:
            continue
        href = (node.get("href") or "").strip()
        name = _label(node)
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")) or not name or len(name) > 40:
            continue
        if _SKIP_LABEL.search(name):
            continue
        url = urldefrag(urljoin(page_url or BASE + "/", href))[0]
        p = urlparse(url)
        if p.netloc and not _same_site(p.netloc):
            continue
        if url.rstrip("/") == BASE or url in seen[current] or len(out[current]) >= 80:
            continue
        seen[current].add(url)
        out[current].append({"name": name, "url": url})
    if not any(out.values()):                       # menu not under those headings: fall back to known labels
        for node in soup.find_all("a", href=True):
            name = _label(node)
            for key, labels in KNOWN.items():
                if _norm(name) in {_norm(x) for x in labels}:
                    url = urldefrag(urljoin(page_url or BASE + "/", node["href"]))[0]
                    if url not in seen[key] and _same_site(urlparse(url).netloc):
                        seen[key].add(url); out[key].append({"name": name, "url": url})
    return out


async def get_menu() -> dict[str, list[dict]]:
    hit = _cached("menu", MENU_CACHE)
    if hit is not None:
        return hit
    html = await _get(BASE + "/", MENU_CACHE)
    menu = discover_menu(html, BASE + "/")
    if not any(menu.values()):
        raise JobsError("menu not found on the home page")
    return _store("menu", menu)


# ----------------------------------------------------------------------- listing parser
_EXP = re.compile(r"^\d+(\.\d+)?(\s*(to|-)\s*\d+(\.\d+)?)?\s*(years?|yrs?|months?)\b", re.I)
_SAL = re.compile(r"(monthly|yearly|per month|per annum|lpa|p\.?a\.?\b|not disclosed|as per)", re.I)
_AGO = re.compile(r"(\bago\b|^today$|^just now$|^yesterday$)", re.I)
_NOISE = re.compile(r"^(apply( now)?|quick apply|view( details| job)?|save|share|walk-?in|new|premium|hot|"
                    r"jobs? opening|read more|more|details|logo)$", re.I)
_JOB_HREF = re.compile(r"/jobs/[^/?#]+-\d{3,}/?$", re.I)


def _title_clean(t: str) -> str:
    t = re.sub(r"\s*Jobs?\s*Opening\b.*$", "", t, flags=re.I).strip(" -|")
    return t or "Job opening"


def _card_for(a: Tag, all_hrefs: set[str]) -> Tag:
    """The smallest ancestor that holds this job's details but no other job's title link."""
    best, node = a, a
    for _ in range(6):
        node = node.parent
        if not isinstance(node, Tag) or node.name in ("body", "html"):
            break
        others = {x.get("href") for x in node.find_all("a", href=True) if _JOB_HREF.search(x.get("href", ""))}
        if len(others - {a.get("href")}) > 0:
            break
        best = node
    return best


def parse_jobs(html: str, page_url: str = "") -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    anchors, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0]
        if _JOB_HREF.search(href) and _label(a):
            full = urljoin(page_url or BASE + "/", href)
            if full not in seen:
                seen.add(full); anchors.append((a, full))
    hrefs = {a.get("href") for a, _ in anchors}
    jobs = []
    for a, full in anchors:
        title_raw = _label(a)
        card = _card_for(a, hrefs)
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in card.get_text("\n").split("\n")]
        lines = [ln for ln in lines if ln and ln != title_raw and not _NOISE.match(ln) and len(ln) < 120]
        exp = next((ln for ln in lines if _EXP.match(ln)), "")
        sal = next((ln for ln in lines if _SAL.search(ln) and re.search(r"\d|disclosed|as per", ln, re.I)), "")
        ago = next((ln for ln in lines if _AGO.search(ln)), "")
        rest = [ln for ln in lines if ln not in (exp, sal, ago) and not re.match(r"^(experience|salary|posted)\b", ln, re.I)]
        jobs.append({"title": _title_clean(title_raw), "company": rest[0] if rest else "",
                     "location": rest[1] if len(rest) > 1 else "", "exp": exp, "salary": sal, "posted": ago, "url": full})
    return jobs


def _page_url(url: str, offset: int) -> str:
    if offset <= 0:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}limit={PAGE_LIMIT}&offset={offset}"


async def get_jobs(url: str, offset: int = 0) -> list[dict]:
    page = _page_url(url, offset)
    hit = _cached("jobs:" + page, PAGE_CACHE)
    if hit is not None:
        return hit
    html = await _get(page, PAGE_CACHE)
    jobs = await asyncio.to_thread(parse_jobs, html, page)
    if not jobs:
        log.warning("no job cards parsed on %s (run: python job_browse.py jobs '%s')", page, page)
    return _store("jobs:" + page, jobs)


# ----------------------------------------------------------------------- rendering
def _job_text(n: int, j: dict) -> str:
    lines = [f"*{n}. {j['title']}*"]
    if j["company"]:
        lines.append(f"🏢 {j['company']}")
    if j["location"]:
        lines.append(f"📍 {j['location']}")
    extra = " • ".join(x for x in (j["exp"] and f"🎯 {j['exp']}", j["salary"] and f"💰 {j['salary']}",
                                   j["posted"] and f"🕒 {j['posted']}") if x)
    if extra:
        lines.append(extra)
    lines.append(f"🔗 {j['url']}")
    return "\n".join(lines)


def _pack(blocks: list[str], head: str = "") -> list[str]:
    msgs, cur = [], head
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > WA_LIMIT:
            msgs.append(cur.strip()); cur = ""
        cur += ("\n\n" if cur and not cur.endswith("\n\n") else "") + b
    if cur.strip():
        msgs.append(cur.strip())
    return msgs


def _sections_text() -> str:
    return ("💼 *Job Search - freshers & entry-level*\n\nHow do you want to browse?\n\n"
            "*1.* 💼 By Category\n*2.* 🎓 By Qualification\n*3.* 🏙️ By City\n*4.* ⌨️ Type a role / skill\n\n"
            "Reply with *1-4* (or *exit*).")


def _list_msgs(section: dict, items: list[dict]) -> list[str]:
    lines = [f"*{i}.* {it['name']}" for i, it in enumerate(items, 1)]
    head = f"{section['emoji']} *Jobs {section['name']}*\n\n"
    msgs, cur = [], head
    for ln in lines:
        if len(cur) + len(ln) + 1 > WA_LIMIT:
            msgs.append(cur.rstrip()); cur = ""
        cur += ln + "\n"
    msgs.append((cur + "\nReply with a *number* or the name. *back* = sections, *exit* = cancel.").strip())
    return msgs


# ----------------------------------------------------------------------- menu flow
def _save(phone: str, state: dict):
    import database as db
    db.save_state(phone, state)


def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


def match_item(items: list[dict], query: str) -> dict | None:
    q = (query or "").strip()
    if not q:
        return None
    if q.isdigit() and 1 <= int(q) <= len(items):
        return items[int(q) - 1]
    nq = _norm(q)
    for it in items:
        if _norm(it["name"]) == nq:
            return it
    hits = [it for it in items if nq and nq in _norm(it["name"])]
    return hits[0] if len(hits) == 1 else (hits[0] if hits and len(nq) >= 3 else None)


def _log(phone: str, **meta):
    try:
        import analytics
        analytics.log_event(phone, "job_search", meta={"source": "browse", **meta})
    except Exception:
        pass


async def start(phone: str, notify=None):
    """/job_search with no argument. Returns (messages, want_role)."""
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "sections"})
    return [_sections_text()], False


async def _open_section(phone: str, idx: int):
    sec = SECTIONS[idx]
    try:
        items = (await get_menu())[sec["key"]]
    except Exception as e:
        log.warning("menu unavailable: %s", e)
        await asyncio.to_thread(_save, phone, {})
        return ["⚠️ I couldn't load the job categories right now. Please try again later, or use */job_search <role>*."]
    if not items:
        return [f"🤔 I couldn't find any items for *{sec['name']}* right now. Try another section."] + [_sections_text()]
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "items", "section": idx, "items": items})
    return _list_msgs(sec, items)


async def _show_jobs(phone: str, state: dict, item: dict):
    """First page of jobs for a chosen category / qualification / city."""
    try:
        jobs = await get_jobs(item["url"], 0)
    except JobsError as e:                           # robots.txt, bot protection, HTTP error: say so, stay in the list
        log.warning("jobs unavailable for %s: %s", item["name"], e)
        jobs = []
    if not jobs:
        return ["😕 I couldn't read any jobs for *%s* right now. Please pick another from the list, *back* or *exit*." % item["name"]]
    st = {"mode": MODE, "step": "jobs", "section": state.get("section"), "items": state.get("items"),
          "item": item, "offset": 0, "shown": 0}
    return await _more(phone, st, first=True)


async def _more(phone: str, st: dict, first: bool = False):
    item, offset, shown = st["item"], int(st.get("offset", 0)), int(st.get("shown", 0))
    jobs = await get_jobs(item["url"], offset)
    if shown >= len(jobs):                           # this page is used up: next page of the site
        try:
            nxt = await get_jobs(item["url"], offset + PAGE_LIMIT)
        except JobsError as e:
            log.warning("next page unavailable for %s: %s", item["name"], e)
            nxt = []
        if not nxt:
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "items", "section": st.get("section"), "items": st.get("items")})
            return ["That's all the jobs I can see for *%s*. Reply with another name/number, *back*, or *exit*." % item["name"]]
        offset, shown, jobs = offset + PAGE_LIMIT, 0, nxt
    batch = jobs[shown:shown + SHOW]
    base_n = offset + shown
    head = f"🎯 *{item['name']}* - jobs {base_n + 1}-{base_n + len(batch)}\n\n" if first or shown == 0 else \
        f"🎯 *{item['name']}* - jobs {base_n + 1}-{base_n + len(batch)}\n\n"
    msgs = _pack([_job_text(base_n + i + 1, j) for i, j in enumerate(batch)], head)
    st.update(offset=offset, shown=shown + len(batch))
    await asyncio.to_thread(_save, phone, st)
    msgs.append("➡️ Reply *more* for the next jobs, a *number/name* to switch (same list), *back* for sections, or *exit*.")
    _log(phone, item=item["name"], offset=base_n)
    return msgs


async def reply(phone: str, text: str, notify=None):
    """A message while the user is inside the job browser. Returns (messages, want_role)."""
    import database as db
    state = (await asyncio.to_thread(db.get_user, phone)).get("state") or {}
    t = (text or "").strip()
    low = t.lower()
    if low in EXIT_WORDS:
        await asyncio.to_thread(_save, phone, {})
        return ["👋 Job search closed."], False
    try:
        if low in ("back", "menu", "0", "sections"):
            return await start(phone)
        step = state.get("step")
        if step == "sections" or not step:
            if low in ("1", "2", "3"):
                return await _open_section(phone, int(low) - 1), False
            if low == "4" or low in ("role", "type", "type a role"):
                await asyncio.to_thread(_save, phone, {})
                return [], True
            for i, s in enumerate(SECTIONS):
                if _norm(s["name"].replace("By ", "")) in _norm(t) and len(_norm(t)) >= 4:
                    return await _open_section(phone, i), False
            return ["🤔 Please reply with *1*, *2*, *3* or *4* (or *exit*).", _sections_text()], False
        if step == "jobs" and low in ("more", "next", "m", "n"):
            return await _more(phone, state), False
        if low in ("more", "next", "m", "n"):
            return ["That's the end of this list. Reply with another number/name, *back* for the sections, or *exit*."], False
        items = state.get("items") or []
        item = match_item(items, t)
        if not item:
            return [f"🤔 I didn't find that. Reply with a number from 1 to {len(items)}, a name from the list, *back* or *exit*."], False
        if notify:
            await notify(f"⏳ Fetching jobs for *{item['name']}*...")
        return await _show_jobs(phone, state, item), False
    except Exception:
        log.exception("job_browse.reply failed")
        return ["⚠️ Something went wrong while loading jobs. Please try again, or use */job_search <role>*."], False


# ----------------------------------------------------------------------- CLI for debugging
async def _cli(argv):  # pragma: no cover
    if len(argv) >= 2 and argv[1] == "menu":
        m = await get_menu()
        for s in SECTIONS:
            print(f"\n== {s['name']} ({len(m[s['key']])})")
            for it in m[s["key"]]:
                print("  ", it["name"], "->", it["url"])
    elif len(argv) >= 3 and argv[1] == "jobs":
        for i, j in enumerate(await get_jobs(argv[2]), 1):
            print(i, j)
    else:
        print('usage: python job_browse.py menu | jobs "<listing url>"')


if __name__ == "__main__":  # pragma: no cover
    import sys
    asyncio.run(_cli(sys.argv))
