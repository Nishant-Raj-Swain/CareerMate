

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import random
import re
import time
from datetime import date
from urllib.parse import unquote, urldefrag, urljoin, urlparse

import httpx
import requests
from bs4 import BeautifulSoup, Tag

log = logging.getLogger("books")

MODE = "books"
UA = os.getenv("BOOKS_UA") or "CareerMateBot/1.0 (educational WhatsApp bot; respects robots.txt)"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5"}
MAX_MB = float(os.getenv("BOOKS_MAX_MB", "45"))                  # bigger PDFs are skipped (WhatsApp / server memory)
MAX_PER_DAY = int(os.getenv("BOOKS_MAX_PER_DAY", "10"))
DELAY = float(os.getenv("FBC_DELAY", "1.2"))
EXIT_WORDS = {"exit", "cancel", "stop", "quit", "/exit", "/cancel"}
STRONG = 0.88                                                    # a match at least this good is sent without asking
TTL_CATEGORIES, TTL_LIST, TTL_BOOK = 12 * 3600, 6 * 3600, 24 * 3600
MAX_HOPS = 5

SITES = {
    "fbc": {"name": "freebookcentre.net", "base": os.getenv("FBC_BASE", "https://www.freebookcentre.net").rstrip("/")},
    "fcb": {"name": "freecomputerbooks.com", "base": os.getenv("FCB_BASE", "https://freecomputerbooks.com").rstrip("/")},
}
# Topic list shown to the user: one entry per subject, with that subject's page on each site that has it.
CATEGORIES = [
    ("Computer Science", {"fbc": "/CompuScience/compscCategory.html", "fcb": "/compscCategory.html"}),
    ("Programming Languages", {"fbc": "/Language/langCategory.html", "fcb": "/langCategory.html"}),
    ("Java", {"fbc": "/JavaTech/javaCategory.html", "fcb": "/javaCategory.html"}),
    ("Unix / Linux", {"fbc": "/UnixCategory/unixCategory.html", "fcb": "/unixCategory.html"}),
    ("Web Technology", {"fbc": "/Web/webCategory.html", "fcb": "/webCategory.html"}),
    ("Databases / Data Science", {"fbc": "/Database/dbCategory.html", "fcb": "/dbCategory.html"}),
    ("Microsoft / .NET", {"fbc": "/MicroSoftTech/microsoftCategory.html", "fcb": "/microsoftCategory.html"}),
    ("Networking", {"fbc": "/Networking/networkCategory.html", "fcb": "/networkCategory.html"}),
    ("Mathematics", {"fbc": "/SpecialCat/Free-Mathematics-Books-Download.html", "fcb": "/mathCategory.html"}),
    ("Mobile Computing", {"fbc": "/mobile-technology/mobile-technology-books.html", "fcb": "/mobileCategory.html"}),
    ("Software Engineering", {"fcb": "/softwareCategory.html"}),
    ("Electronics / Computer Engineering", {"fbc": "/electronics_communication_books/electronics_communication_books.html", "fcb": "/eeCategory.html"}),
    ("Special Topics", {"fbc": "/SpecialCat/specialCategory.html", "fcb": "/specialCategory.html"}),
    ("Physics", {"fbc": "/Physics/Physics-Books-Online.html"}),
    ("Chemistry", {"fbc": "/Chemistry/Chemistry-Books-Online.html"}),
    ("Biology", {"fbc": "/Biology/Biology-Books-Online.html"}),
    ("Medical", {"fbc": "/medical_text_books_journals/medical_text_books_online.html"}),
    ("Earth Sciences", {"fbc": "/EarthSciences/Earth-Sciences-Books.html"}),
    ("Business and Finance", {"fbc": "/Business/Business-and-Finance-Books.html"}),
    ("Law", {"fbc": "/Law/Law-Books.html"}),
    ("Electronics Engineering", {"fbc": "/Electronics/Engineering-Books-Online.html"}),
    ("Hardware Buses", {"fbc": "/HardwareBus/Hardware-Buses-Books.html"}),
    ("Graphics Tools", {"fbc": "/graphics-design-books/web-design-tools.html"}),
    ("File System Specifications", {"fbc": "/file_system_specification/file_system_specifications.html"}),
    ("Fiction", {"fbc": "/Fiction/Literature-and-Fiction.html"}),
    ("Cooking", {"fbc": "/Cooking/Cooking-Food-Drink-Books.html"}),
]


class BooksError(Exception):
    pass


# ----------------------------------------------------------------------- text helpers
def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    return re.sub(r"^(the|a|an) ", "", s.strip()).strip()


_STOP = {"and", "of", "the", "a", "an", "in", "to", "for", "with", "on", "book", "books", "pdf", "free", "by", "download"}


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t not in _STOP]


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def score(query: str, b: dict) -> float:
    """0..1: how well a book answers the query (title first, author as a bonus)."""
    nq, nt = _norm(query), _norm(b["title"])
    if not nq or not nt:
        return 0.0
    qt, tt = {_stem(t) for t in _tokens(query)}, {_stem(t) for t in _tokens(b["title"])}
    s = difflib.SequenceMatcher(None, nq, nt).ratio()
    if nt == nq:
        s = 1.0
    elif qt and qt <= tt:
        s = max(s, 0.95 - min(0.2, 0.04 * len(tt - qt)))
    elif nt.startswith(nq) or nq.startswith(nt):
        s = max(s, 0.85)
    au = {_stem(t) for t in _tokens(b.get("author", ""))}
    if au and au & qt:
        s = min(1.0, s + 0.08)
    return round(s, 3)


def _safe_filename(title: str) -> str:
    return (re.sub(r"[^A-Za-z0-9 _-]+", "", title).strip().replace(" ", "_")[:60] or "book") + ".pdf"


def _label(a: Tag) -> str:
    return re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()


# ----------------------------------------------------------------------- polite HTTP (robots.txt, one at a time, spaced, cached)
_lock: asyncio.Lock | None = None
_last = 0.0
_robots: dict[str, tuple[float, list[str]]] = {}
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float):
    hit = _cache.get(key)
    return hit[1] if hit and time.time() - hit[0] < ttl else None


def _store(key: str, value):
    if len(_cache) > 600:
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:200]:
            _cache.pop(k, None)
    _cache[key] = (time.time(), value)
    return value


def _site_of(url: str) -> str | None:
    host = urlparse(url).netloc
    for k, s in SITES.items():
        if urlparse(s["base"]).netloc == host:
            return k
    return None


def parse_robots(text: str, ua: str) -> list[str]:
    """Disallow rules that apply to us: the '*' group plus any group whose agent name is part of our user-agent."""
    rules, agents, in_rules = [], [], False
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        key, val = [x.strip() for x in line.split(":", 1)]
        key = key.lower()
        if key == "user-agent":
            if in_rules:
                agents, in_rules = [], False
            agents.append(val.lower())
        elif key in ("disallow", "allow"):
            in_rules = True
            if key == "disallow" and val and any(a == "*" or (a and a in ua.lower()) for a in agents):
                rules.append(val)
    return rules


async def _disallowed(client: httpx.AsyncClient, url: str) -> bool:
    p = urlparse(url)
    hit = _robots.get(p.netloc)
    if not hit or time.time() - hit[0] > 86400:
        rules = []
        try:
            r = await client.get(f"{p.scheme}://{p.netloc}/robots.txt")
            if r.status_code == 200:
                rules = parse_robots(r.text, UA)
        except httpx.HTTPError as e:
            log.warning("robots.txt unavailable for %s (%s)", p.netloc, e)
        hit = _robots[p.netloc] = (time.time(), rules)
    target = (p.path or "/") + (("?" + p.query) if p.query else "")
    return any(rule and target.startswith(rule) for rule in hit[1])


def _lk() -> asyncio.Lock:
    global _lock
    _lock = _lock or asyncio.Lock()
    return _lock


async def _space():
    global _last
    wait = DELAY + random.uniform(0, 0.4) - (time.time() - _last)
    if wait > 0:
        await asyncio.sleep(wait)


async def _get(url: str, ttl: float) -> str:
    """Polite GET of a page of one of the two sites. Returns HTML."""
    global _last
    url = urldefrag(url)[0]
    hit = _cached("html:" + url, ttl)
    if hit is not None:
        return hit
    if not _site_of(url):
        raise BooksError(f"not one of the book sites: {url}")
    last = None
    async with _lk():
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, headers=HEADERS) as client:
            if await _disallowed(client, url):
                raise BooksError(f"blocked by robots.txt: {url}")
            for attempt in range(3):
                await _space()
                try:
                    r = await client.get(url)
                except httpx.HTTPError as e:
                    last = str(e); _last = time.time(); continue
                _last = time.time()
                if r.status_code == 200:
                    return _store("html:" + url, r.text)
                if r.status_code in (403, 429) or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"; await asyncio.sleep(2 * (attempt + 1)); continue
                raise BooksError(f"HTTP {r.status_code} for {url}")
    raise BooksError(f"could not fetch {url}: {last}")


# ----------------------------------------------------------------------- reading the pages
_CHROME_TEXT = re.compile(r"^(home|about( us)?|link to us|contact( us)?|privacy policy|terms( & | and )?conditions?|sitemap|search\.*|all|"
                          r"submit|advertise|disclaimer|back to top|next|previous|more\.*|faq|top free books|recent books|"
                          r"other free book sites|search engines|developer tools)$", re.I)
_DETAIL_FBC = re.compile(r"/[^/]*download[^/]*/[^/]+\.html?$", re.I)          # /Java-Books-Download/<Book-Slug>.html
_TOPIC_FILE = re.compile(r"(Books|Category)\.html?$|-[Bb]ooks\.html?$|Books-(Online|Download)\.html?$|-books-online\.html?$")
_CHROME_FILES = {"sitemap.html", "recentbooks.html", "about_books.html", "books_faq.html", "searchengines.html", "webtools.html",
                 "index.html", "privacy.html", "terms.html", "contact.html", "otherbooks.html"}
_COVER = re.compile(r"/covers/([^\"'\s/]+?)(?:_\d+x\d+)?\.(?:jpg|jpeg|png|gif|webp)", re.I)


def _is_topic_url(url: str) -> bool:
    return bool(_TOPIC_FILE.search(urlparse(url).path.rsplit("/", 1)[-1]))


def _is_detail(site: str, url: str, title: str, covers: set[str]) -> bool:
    p = urlparse(url).path
    if site == "fbc":
        return bool(_DETAIL_FBC.search(p))
    base = p.rsplit("/", 1)[-1]
    if p.count("/") != 1 or not base.lower().endswith(".html") or _is_topic_url(url) or base.lower() in _CHROME_FILES or "_" in base:
        return False
    return base[:-5].lower() in covers or len(title.split()) >= 2


def parse_page(html: str, page_url: str, skip_urls: set[str]) -> dict:
    """A category / sub-topic page -> {'subtopics': [{name,url}], 'books': [{title,author,pages,pdf,url,site}]}."""
    site = _site_of(page_url) or "fbc"
    soup = BeautifulSoup(html, "lxml")
    covers = {m.lower() for m in _COVER.findall(html)}
    here = urlparse(page_url).path.rstrip("/")
    subs, seen_subs, groups = [], set(), {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        url = urldefrag(urljoin(page_url, href))[0]
        p = urlparse(url)
        if _site_of(url) != site or not p.path.lower().endswith((".html", ".htm")) or "gotoweb" in p.path:
            continue
        name = _label(a)
        if _is_detail(site, url, name or unquote(p.path.rsplit("/", 1)[-1]).replace("-", " "), covers):
            groups.setdefault(url, []).append(a)
        elif (3 <= len(name) <= 50 and not _CHROME_TEXT.match(name) and url not in skip_urls and p.path.rstrip("/") != here
              and url not in seen_subs and (site == "fbc" or _is_topic_url(url))):
            seen_subs.add(url); subs.append({"name": name, "url": url})
    books = []
    for url, anchors in groups.items():
        title = max((_label(a) for a in anchors), key=len, default="") or unquote(urlparse(url).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0].replace("-", " ")
        card, node = anchors[0], anchors[0]
        for _ in range(6):                                           # the smallest block holding this book but no other book
            node = node.parent
            if not isinstance(node, Tag) or node.name in ("body", "html"):
                break
            others = {urldefrag(urljoin(page_url, x["href"]))[0] for x in node.find_all("a", href=True)}
            if any(o != url and o in groups for o in others):
                break
            card = node
        text = re.sub(r"\s+", " ", card.get_text(" ", strip=True))
        lines = [re.sub(r"\s+", " ", x).strip() for x in card.get_text("\n").split("\n") if x.strip()]
        pages = re.search(r"(\d+)\s*Pages", text, re.I)
        is_pdf = bool(re.search(r"\bPDF\b", text)) or any("pdf" in ((i.get("src") or "") + (i.get("alt") or "")).lower() for i in card.find_all("img"))
        rest = [ln for ln in lines if _norm(ln) != _norm(title) and not re.search(r"\d+\s*Pages", ln, re.I) and not re.fullmatch(r"(pdf|na|\d+)", ln, re.I)
                and len(ln) <= 60 and not re.match(r"post under|download|view", ln, re.I)]
        books.append({"title": title, "author": rest[0] if (rest and site == "fbc") else "", "pages": pages.group(1) if pages else "",
                      "pdf": is_pdf, "url": url, "site": site})
    return {"subtopics": subs[:40], "books": books}


async def _page(url: str, ttl: float = TTL_LIST, skip: set[str] | None = None) -> dict:
    html = await _get(url, ttl)
    return await asyncio.to_thread(parse_page, html, url, skip or set())


def _cat_urls() -> set[str]:
    return {s["base"] + path for _, paths in CATEGORIES for k, path in paths.items() for s in [SITES[k]]}


def _full(cat: dict) -> list[str]:
    return [SITES[k]["base"] + path for k, path in cat["paths"].items()]


async def get_subtopics(cat: dict) -> tuple[list[dict], list[dict]]:
    """Sub-topics of a category from every site that has it, merged by name -> ([{name, urls}], books listed directly on the page)."""
    merged: dict[str, dict] = {}
    direct: list[dict] = []
    skip = _cat_urls()
    for url in _full(cat):
        try:
            pg = await _page(url, TTL_LIST, skip)
        except BooksError as e:
            log.warning("category page failed: %s", e); continue
        direct += pg["books"]
        for s in pg["subtopics"]:
            m = merged.setdefault(_norm(s["name"]), {"name": s["name"], "urls": []})
            if s["url"] not in m["urls"]:
                m["urls"].append(s["url"])
    return list(merged.values()), direct


async def get_books(urls: list[str]) -> list[dict]:
    """Books of one sub-topic (all its pages), the two sites interleaved, duplicates removed, PDF-flagged first."""
    lists = []
    for u in urls:
        try:
            lists.append((await _page(u))["books"])
        except BooksError as e:
            log.warning("topic page failed: %s", e)
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if i < len(lst) and _norm(lst[i]["title"]) not in seen:
                seen.add(_norm(lst[i]["title"])); out.append(lst[i])
    return [b for b in out if b["pdf"]] + [b for b in out if not b["pdf"]]


# ----------------------------------------------------------------------- book page -> real PDF
_PDF_LINK = re.compile(r"\.pdf(\?|#|$)", re.I)
_META_REFRESH = re.compile(r"url\s*=\s*['\"]?([^'\";>]+)", re.I)
_JS_REDIRECT = re.compile(r"(?:window\.|document\.|top\.)?location(?:\.href)?\s*(?:=|\.replace\(|\.assign\()\s*['\"]([^'\"]+)['\"]", re.I)


def _title_words(title: str) -> set[str]:
    return {_stem(t) for t in _tokens(title)}


def pick_pdf_links(soup: BeautifulSoup, page_url: str, title: str) -> list[str]:
    """Direct .pdf links on a page, best first (words of the title in the link / file name win)."""
    tw, out = _title_words(title), []
    for a in soup.find_all("a", href=True):
        url = urljoin(page_url, a["href"].strip())
        if not url.startswith("http") or not _PDF_LINK.search(urlparse(url).path):
            continue
        s = 1 + len(tw & {_stem(t) for t in _tokens(_label(a) + " " + unquote(url))}) + (1 if re.search(r"download|pdf|full", _label(a), re.I) else 0)
        out.append((s, url))
    out.sort(key=lambda x: -x[0])
    seen, res = set(), []
    for _, u in out:
        if u not in seen:
            seen.add(u); res.append(u)
    return res


def next_hop(html: str, page_url: str, title: str) -> str:
    """From an HTML page: a meta / JS redirect, else the best direct-PDF link. '' if none."""
    soup = BeautifulSoup(html, "lxml")
    m = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
    if m and m.get("content"):
        mm = _META_REFRESH.search(m["content"])
        if mm:
            return urljoin(page_url, mm.group(1).strip())
    for sc in soup.find_all("script"):
        mm = _JS_REDIRECT.search(sc.string or sc.get_text() or "")
        if mm:
            return urljoin(page_url, mm.group(1))
    links = pick_pdf_links(soup, page_url, title)
    return links[0] if links else ""


async def _follow(url: str, title: str) -> str:
    """Follow redirects / landing pages (max MAX_HOPS) to a PDF url. '' if none. Never logs in to anything."""
    global _last
    seen = set()
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=20.0, headers=HEADERS) as client:
            for _ in range(MAX_HOPS):
                if url in seen or not url.startswith("http"):
                    return ""
                seen.add(url)
                if _site_of(url):
                    async with _lk():                                # the site's own redirect script: robots + spacing
                        if await _disallowed(client, url):
                            return ""
                        await _space(); _last = time.time()
                if _PDF_LINK.search(urlparse(url).path):
                    return url
                async with client.stream("GET", url) as r:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        url = urljoin(url, r.headers["location"]); continue
                    ctype = r.headers.get("content-type", "").lower()
                    if r.status_code != 200:
                        return ""
                    if "pdf" in ctype:
                        return url
                    if "html" not in ctype and ctype:
                        return ""
                    body = (await r.aread())[:400_000].decode(r.encoding or "utf-8", errors="ignore")
                url = next_hop(body, url, title)
    except httpx.HTTPError as e:
        log.info("follow failed (%s): %s", url, e)
    return ""


async def resolve_pdfs(book: dict) -> list[str]:
    """Up to 3 candidate PDF urls for a book, best first (the first one that really is a PDF gets sent)."""
    hit = _cached("res:" + book["url"], TTL_BOOK)
    if hit is not None:
        return hit
    urls: list[str] = []
    try:
        html = await _get(book["url"], TTL_BOOK)
        soup = BeautifulSoup(html, "lxml")
        hops = []
        for a in soup.find_all("a", href=True):
            href = urljoin(book["url"], a["href"].strip())
            if not href.startswith("http"):
                continue
            if "gotoweb" in href:
                hops.insert(0, href)
            elif not _PDF_LINK.search(urlparse(href).path) and not _site_of(href) and re.search(r"download|read online|view|homepage|mirror|pdf", _label(a), re.I):
                hops.append(href)
        urls = pick_pdf_links(soup, book["url"], book["title"])[:3]
        for h in hops[:3]:
            if urls:
                break
            u = await _follow(h, book["title"])
            if u:
                urls.append(u)
    except BooksError as e:
        log.info("could not read %s: %s", book["url"], e)
        return urls                                                  # not cached: may work next time
    return _store("res:" + book["url"], urls)


# ----------------------------------------------------------------------- download + send (PDF only)
def _download_sync(url: str) -> tuple[bytes | None, str]:
    cap = int(MAX_MB * 1024 * 1024)
    try:
        with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=(10, 60), allow_redirects=True) as r:
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            if int(r.headers.get("Content-Length") or 0) > cap:
                return None, "too big"
            buf = bytearray()
            for chunk in r.iter_content(65536):
                buf += chunk
                if len(buf) > cap:
                    return None, "too big"
    except requests.RequestException as e:
        return None, str(e)[:80]
    return (bytes(buf), "ok") if bytes(buf[:4]) == b"%PDF" else (None, "not a PDF")


def _graph():
    base = (os.getenv("WHATSAPP_GRAPH_BASE") or "https://graph.facebook.com/v21.0").rstrip("/")
    token = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
    pid = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID")
    if not token or not pid:
        raise BooksError("WhatsApp credentials are missing on the server.")
    return base, {"Authorization": f"Bearer {token}"}, pid


def _send_document_sync(phone: str, data: bytes, filename: str, caption: str) -> None:
    base, headers, pid = _graph()
    up = requests.post(f"{base}/{pid}/media", headers=headers, data={"messaging_product": "whatsapp", "type": "application/pdf"},
                       files={"file": (filename, data, "application/pdf")}, timeout=120)
    if up.status_code >= 400:
        raise BooksError(f"media upload failed: HTTP {up.status_code} {up.text[:150]}")
    r = requests.post(f"{base}/{pid}/messages", headers={**headers, "Content-Type": "application/json"}, timeout=30,
                      json={"messaging_product": "whatsapp", "to": phone, "type": "document",
                            "document": {"id": up.json().get("id"), "filename": filename, "caption": caption[:1000]}})
    if r.status_code >= 400:
        raise BooksError(f"media send failed: HTTP {r.status_code} {r.text[:150]}")


async def deliver(phone: str, b: dict) -> bool:
    """Find the book's PDF and send it as a document. True only if a real PDF was sent."""
    for url in await resolve_pdfs(b):
        data, why = await asyncio.to_thread(_download_sync, url)
        if not data:
            log.info("no PDF from %s: %s", url, why)
            continue
        caption = (f"📚 *{b['title']}*" + (f"\n✍️ {b['author']}" if b.get("author") else "") + (f"\n📄 {b['pages']} pages" if b.get("pages") else "") +
                   f"\n🏷 Listed on {SITES[b['site']]['name']} - free for personal, non-commercial use.")
        try:
            await asyncio.to_thread(_send_document_sync, phone, data, _safe_filename(b["title"]), caption)
            return True
        except Exception:
            log.exception("document send failed for %s", b["title"])
            return False
    return False


# ----------------------------------------------------------------------- searching by name (topic index)
async def _topic_index() -> list[dict]:
    """Every topic page of both sites: [{name, url}] (freecomputerbooks sitemap + the category pages of both sites)."""
    hit = _cached("topics", TTL_CATEGORIES)
    if hit is not None:
        return hit
    found: dict[str, dict] = {}
    skip = _cat_urls()

    def add(name, url):
        if name and url not in found and _is_topic_url(url) and url not in skip:
            found[url] = {"name": name, "url": url}
    try:
        html = await _get(SITES["fcb"]["base"] + "/sitemap.html", TTL_CATEGORIES)
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            url = urldefrag(urljoin(SITES["fcb"]["base"] + "/", a["href"]))[0]
            if _site_of(url) == "fcb" and 3 <= len(_label(a)) <= 60:
                add(_label(a), url)
    except BooksError as e:
        log.warning("sitemap unavailable: %s", e)
    for name, paths in CATEGORIES:
        for k, path in paths.items():
            if k == "fcb" and len(found) > 60:
                continue
            try:
                for s in (await _page(SITES[k]["base"] + path, TTL_CATEGORIES, skip))["subtopics"]:
                    add(s["name"], s["url"])
            except BooksError as e:
                log.warning("index page failed: %s", e)
    return _store("topics", list(found.values()))


async def warm():
    try:
        await _topic_index()
    except Exception:
        log.exception("warm-up failed")


def _topic_score(query: str, name: str) -> float:
    qt, nt = {_stem(t) for t in _tokens(query)}, {_stem(t) for t in _tokens(name)}
    if not qt or not nt:
        return 0.0
    hit = len(qt & nt)
    for q in qt - nt:                                               # "algorithm" ~ "algorithms", "cpp" ~ "c++" no; partial words
        if any(len(q) >= 4 and (t.startswith(q) or q.startswith(t)) for t in nt if len(t) >= 4):
            hit += 0.6
    return hit / max(1, min(len(qt), 3)) + 0.1 * (hit / len(nt))


async def search(query: str) -> tuple[list[dict], str]:
    """-> (books ranked by score, name of the best topic). Looks inside the topics that fit the query."""
    index = await _topic_index()
    ranked = sorted(((_topic_score(query, t["name"]), t) for t in index), key=lambda x: -x[0])
    topics = [t for s, t in ranked if s > 0.3][:4]
    if not topics:
        return [], ""
    books, seen = [], set()
    for t in topics:
        try:
            for b in (await _page(t["url"]))["books"]:
                if _norm(b["title"]) not in seen:
                    seen.add(_norm(b["title"])); b["topic"] = t["name"]; books.append(b)
        except BooksError as e:
            log.info("topic page failed: %s", e)
    for b in books:
        b["score"] = score(query, b)
    books.sort(key=lambda b: -b["score"])
    return books, topics[0]["name"]


# ----------------------------------------------------------------------- limits / logging / state
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
        analytics.log_event(phone, "books", meta=meta, success=ok)
    except Exception:
        pass


def _save(phone: str, state: dict):
    import database as db
    db.save_state(phone, state)


def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


# ----------------------------------------------------------------------- conversation
MENU = ("📚 *Free Books* - how do you want to find a book?\n\n"
        "*1.* 📂 Topic wise (Java, Unix/Linux, Networking, Mathematics ...)\n"
        "*2.* 🔎 Book name / author name\n\n"
        "Reply *1* or *2* (or *exit*).")
ASK = ("🔎 Type the *correct name* of the book (add the author if you know it), e.g.\n• Python for Artificial Intelligence\n• Operating Systems\n"
       "I'll look for it and send the PDF. If it isn't available I'll suggest similar books on the same topic.\nSend *exit* to cancel.")
DONE = "✅ Done! Send */books* for more, or */menu* for the menu."


def _numbered(items: list[dict]) -> str:
    return "\n".join(f"*{i}.* {x['name']}" for i, x in enumerate(items, 1))


async def start(phone: str, arg: str, say) -> bool:
    """/books [book name]. True = the job is over (state cleared)."""
    if arg and arg.strip():
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
        return await handle(phone, arg, say)
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
    await say(MENU)
    return False


async def _show_categories(phone: str, say) -> bool:
    cats = [{"name": n, "paths": p} for n, p in CATEGORIES]
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "cat", "cats": cats})
    await say("📂 *Choose a topic* - reply with its number:\n\n" + _numbered(cats) + "\n\n(*0* = back, *exit* = cancel)")
    return False


async def _send_top_books(phone: str, name: str, urls: list[str], say) -> bool:
    """Sends up to 5 books as PDFs. True if at least one PDF went out."""
    books = await get_books(urls)
    if not books:
        await say(f"😕 I couldn't read any books under *{name}* right now. Pick another (*0* = back).")
        return False
    await say(f"⏳ Getting the top books on *{name}*. Each PDF takes a few seconds...")
    sent = 0
    for b in books[:14]:
        if sent >= 5:
            break
        if await deliver(phone, b):
            sent += 1
    if not sent:
        await say("😕 None of those books has a downloadable PDF right now. Please try another sub-topic (*0* = back).")
        return False
    if sent < 5:
        await say(f"ℹ️ Only {sent} book(s) on this topic have a downloadable PDF right now.")
    return True


def _limit_msg() -> str:
    return f"⏳ You've reached today's limit of {MAX_PER_DAY} requests. Please try again tomorrow."


def _alt_text(books: list[dict], topic: str, query: str) -> str:
    lines = [f"*{i}.* {b['title']}" + (f" - {b['author']}" if b.get("author") else "") for i, b in enumerate(books, 1)]
    return (f"😕 I couldn't find *{query[:60]}* exactly. Here are PDF books on *{topic}* that may help - reply with a *number* "
            "(or type another book name, *0* = menu, *exit*):\n\n" + "\n".join(lines))


async def handle(phone: str, text: str, say) -> bool:
    import database as db
    text = (text or "").strip()
    low = text.lower()
    if low in EXIT_WORDS:
        await asyncio.to_thread(_save, phone, {})
        await say("👋 Cancelled.")
        return True
    state = (await asyncio.to_thread(db.get_user, phone)).get("state") or {}
    step = state.get("step")
    try:
        if text == "0" and step in ("cat", "ask", "pick"):
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"}); await say(MENU); return False
        if step == "menu" and text in ("1", "2"):
            if text == "1":
                return await _show_categories(phone, say)
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"}); await say(ASK); return False
        if step in ("menu", "cat", "sub") and text.isdigit() or (step == "sub" and text == "0"):
            if step == "sub" and text == "0":
                return await _show_categories(phone, say)
            if step == "cat":
                cats = state.get("cats") or []
                if not 1 <= int(text) <= len(cats):
                    await say(f"🤔 Please reply with a number from 1 to {len(cats)} (*0* = back)."); return False
                cat = cats[int(text) - 1]
                await say(f"🔎 Loading *{cat['name']}*...")
                subs, direct = await get_subtopics(cat)
                if subs:
                    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "sub", "subs": subs, "cat": cat["name"]})
                    await say(f"📘 *{cat['name']}* - choose a sub-topic:\n\n" + _numbered(subs) + "\n\n(*0* = back, *exit* = cancel)")
                    return False
                if not direct:
                    await say("😕 Nothing readable under that topic right now. Pick another (*0* = back)."); return False
                target = (cat["name"], _full(cat))
            elif step == "sub":
                subs = state.get("subs") or []
                if not 1 <= int(text) <= len(subs):
                    await say(f"🤔 Please reply with a number from 1 to {len(subs)} (*0* = back)."); return False
                target = (subs[int(text) - 1]["name"], subs[int(text) - 1]["urls"])
            else:
                await say("🤔 Please reply *1* or *2* (or *exit*)."); return False
            if not _quota_ok(phone):
                await asyncio.to_thread(_save, phone, {}); await say(_limit_msg()); return True
            if not await _send_top_books(phone, target[0], target[1], say):
                _refund(phone); return False
            await asyncio.to_thread(_save, phone, {})
            _log(phone, topic=state.get("cat", ""), subtopic=target[0])
            await say(DONE)
            return True
        if step == "pick" and text.isdigit():
            cands = state.get("cands") or []
            if not 1 <= int(text) <= len(cands):
                await say(f"🤔 Please reply with a number from 1 to {len(cands)} (*0* = menu)."); return False
            b = cands[int(text) - 1]
            await say(f"⏳ Getting *{b['title']}*...")
            if await deliver(phone, b):
                await asyncio.to_thread(_save, phone, {}); _log(phone, title=b["title"]); await say(DONE); return True
            await say("😕 That book has no downloadable PDF right now. Please pick another number from the list.")
            return False
        if step in ("menu", "cat", "sub") and (text.isdigit() or len(text) < 3):
            await say("🤔 Please reply with one of the numbers shown (or *exit*)."); return False
        # ---- a typed book name
        if len(text) < 3:
            await say("🤔 Please type the book's name (or *exit*)."); return False
        if not _quota_ok(phone):
            await asyncio.to_thread(_save, phone, {}); await say(_limit_msg()); return True
        await say(f"🔎 Looking for *{text[:80]}* (the first search can take a minute)...")
        books, topic = await search(text)
        if books and books[0]["score"] >= STRONG and (len(books) == 1 or books[1]["score"] <= books[0]["score"] - 0.06):
            top = books[0]
            await say(f"✅ Found: *{top['title']}*. Sending the PDF...")
            if await deliver(phone, top):
                await asyncio.to_thread(_save, phone, {}); _log(phone, title=top["title"], query=text[:80]); await say(DONE); return True
            books = books[1:]                                       # found but no downloadable PDF: offer the rest
        alts = [b for b in books if b["pdf"]][:5] or books[:5]
        if alts:
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "pick", "cands": alts})
            await say(_alt_text(alts, topic or "this topic", text)); _log(phone, ok=False, query=text[:80], alternatives=len(alts))
            return False
        _refund(phone)
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
        await say("😕 I couldn't find that book or a similar topic. Try the topic words (e.g. *python*, *networking*, *operating systems*), "
                  "send another name, or send *0* and choose *Topic wise*.")
        _log(phone, ok=False, query=text[:80])
        return False
    except Exception:
        _refund(phone)
        log.exception("books.handle failed")
        await asyncio.to_thread(_save, phone, {})
        await say("⚠️ Something went wrong while looking for the book. Please try again in a few minutes.")
        return True


# ----------------------------------------------------------------------- CLI
async def _cli(argv):  # pragma: no cover
    if len(argv) >= 3 and argv[1] == "topics":
        cat = next((c for c in CATEGORIES if _norm(argv[2]) in _norm(c[0])), None)
        if not cat:
            print("no such category:", [c[0] for c in CATEGORIES]); return
        subs, direct = await get_subtopics({"name": cat[0], "paths": cat[1]})
        print(f"{cat[0]}: {len(subs)} sub-topics, {len(direct)} books listed directly")
        for s in subs:
            print("  -", s["name"], s["urls"])
    elif len(argv) >= 3 and argv[1] == "books":
        for b in (await get_books([argv[2]]))[:5]:
            urls = await resolve_pdfs(b)
            print(b["title"], "|", b["author"], "|", b["pages"], "| PDF:", urls[:2] or "none")
    elif len(argv) >= 3 and argv[1] == "search":
        books, topic = await search(argv[2])
        print("best topic:", topic)
        for b in books[:6]:
            print(b["score"], b["title"], b["url"])
    else:
        print('usage: python books.py topics "<category>" | books "<sub-topic page url>" | search "<book name>"')


if __name__ == "__main__":  # pragma: no cover
    import sys
    asyncio.run(_cli(sys.argv))
