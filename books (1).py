# books.py
# ---------------------------------------------------------------------------
# /books  ->  the bot asks for the book's name, searches LEGAL free sources, and sends the PDF (or EPUB) to the user.
#
#   Sources (all free + legal; each is searched in parallel and the best match wins,
#   so a book missing in one is found in another):
#     1.  OpenStax               free college textbooks (physics, maths, chemistry, economics ...)   - PDF
#     2.  Open Textbook Library  peer-reviewed open textbooks (UMN)                                  - PDF / link
#     3.  Project Gutenberg      public-domain classics (via the Gutendex API)                       - EPUB
#     4.  OAPEN                  open-access academic books                                          - PDF
#     5.  DOAB                    Directory of Open Access Books (peer-reviewed, all OA)            - PDF
#     6.  Google Books          only books flagged "free ebooks" (public domain / publisher-free)   - PDF/EPUB
#     7.  Wikisource            Wikimedia free library (public domain / CC BY-SA)                   - PDF
#     8.  Open Library + Internet Archive  only items flagged public / Creative Commons / public domain - PDF
#
#   Standard Ebooks was considered but is NOT included: their robots.txt explicitly forbids
#   automated downloads (`Disallow: /ebooks/*/downloads/*`). Respect it.
#
#   In-copyright books that are only "borrowable" are never downloaded. Every file is sent with its source + licence.
#   (NCERT school textbooks: the bot sends the official page link - their files are per chapter.)
#
# Public API used by main.py (state lives in users.state like every other mode):
#       done = await books.start(phone, arg, say)      # /books [book name]    (True = finished, state cleared)
#       done = await books.handle(phone, text, say)    # while state mode == books.MODE
#
# Check each source from your server with:   python books.py probe "pride and prejudice"
#                                            python books.py get   "university physics volume 1"   (downloads the file)
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import time
from datetime import date
from urllib.parse import quote, quote_plus

import httpx
import requests

log = logging.getLogger("books")

MODE = "books"
UA = os.getenv("BOOKS_UA") or "CareerMateBot/1.0 (educational WhatsApp bot; free/legal books only)"
HEADERS = {"User-Agent": UA, "Accept": "application/json"}
MAX_MB = float(os.getenv("BOOKS_MAX_MB", "45"))                  # larger files are sent as a link instead
MAX_PER_DAY = int(os.getenv("BOOKS_MAX_PER_DAY", "10"))
TIMEOUT = float(os.getenv("BOOKS_TIMEOUT", "14"))
EXIT_WORDS = {"exit", "cancel", "stop", "quit", "/exit", "/cancel", "back"}

GUTENDEX = os.getenv("GUTENDEX_BASE", "https://gutendex.com").rstrip("/")
OPENLIBRARY = os.getenv("OPENLIBRARY_BASE", "https://openlibrary.org").rstrip("/")
ARCHIVE = os.getenv("ARCHIVE_BASE", "https://archive.org").rstrip("/")
OAPEN = os.getenv("OAPEN_BASE", "https://library.oapen.org").rstrip("/")
OPENSTAX = os.getenv("OPENSTAX_BASE", "https://openstax.org").rstrip("/")
GOOGLE_BOOKS = os.getenv("GOOGLE_BOOKS_BASE", "https://www.googleapis.com").rstrip("/")
NCERT_PAGE = "https://ncert.nic.in/textbook.php"
DOAB = os.getenv("DOAB_BASE", "https://directory.doabooks.org").rstrip("/")
OPEN_TEXTBOOK_LIBRARY = os.getenv("OPEN_TEXTBOOK_LIBRARY_BASE", "https://open.umn.edu").rstrip("/")
WIKISOURCE = os.getenv("WIKISOURCE_BASE", "https://en.wikisource.org").rstrip("/")

# Lower number = higher rank (small bonus for sources that are usually best for the query type)
SOURCE_RANK = {
    "OpenStax": 0, "Open Textbook Library": 1,
    "Project Gutenberg": 2,
    "OAPEN": 3, "DOAB": 4,
    "Google Books": 5, "Wikisource": 6,
    "Open Library": 7, "Internet Archive": 8,
}
STRONG = 0.85                                                    # score above which a match is sent without asking
CUTOFF = 0.40                                                    # below this a candidate is dropped from results


class BooksError(Exception):
    pass


# ----------------------------------------------------------------------- text helpers
def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    s = re.sub(r"^(the|a|an) ", "", s.strip())
    return s.strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t not in ("and", "of", "the", "a", "an", "in", "to")]


def _person(name: str) -> str:
    """'Austen, Jane' -> 'Jane Austen'."""
    if "," in (name or ""):
        last, _, first = name.partition(",")
        return f"{first.strip()} {last.strip()}".strip()
    return (name or "").strip()


def score(query: str, cand: dict) -> float:
    """0..1: how well a candidate answers the query (title first, author as a bonus)."""
    nq, nt = _norm(query), _norm(cand["title"])
    if not nq or not nt:
        return 0.0
    qt, tt = set(_tokens(query)), set(_tokens(cand["title"]))
    s = difflib.SequenceMatcher(None, nq, nt).ratio()
    if nt == nq:
        s = 1.0
    elif qt and qt <= tt:                                      # every word the user typed is in the title
        extra = len(tt - qt)
        s = max(s, 0.95 - min(0.2, 0.04 * extra))
    elif nt.startswith(nq) or nq.startswith(nt):
        s = max(s, 0.85)
    au = _tokens(cand.get("author", ""))
    if au and any(a in qt for a in au):                        # "pride prejudice austen"
        s = min(1.0, s + 0.06)
    return round(s, 3)


def _safe_filename(title: str, ext: str) -> str:
    base = re.sub(r"[^A-Za-z0-9 _-]+", "", title).strip().replace(" ", "_")[:60] or "book"
    return f"{base}.{ext}"


_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def _sig(title: str) -> tuple:
    """Identity of a title: the main title (before a subtitle) + its volume / part numbers. 'Calculus Volume 1' != 'Calculus Volume 2'."""
    base = _norm(re.split(r":| - | \u2013 | \u2014 ", title or "")[0])
    nums = tuple(sorted(re.findall(r"\d+", title or "")))
    roman = tuple(sorted(w for w in _norm(title).split() if w in _ROMAN))
    return base, nums, roman


def _same_book(a: dict, b: dict) -> bool:
    """Two candidates are the same book only if titles agree INCLUDING volume numbers (and the authors' surnames, when both known)."""
    sa, sb = _sig(a["title"]), _sig(b["title"])
    if sa[1:] != sb[1:] and (sa[1] or sb[1] or sa[2] or sb[2]):
        return False
    if not (sa[0] == sb[0] or difflib.SequenceMatcher(None, sa[0], sb[0]).ratio() >= 0.93):
        return False
    aa, ab = _norm(_person(a.get("author", ""))), _norm(_person(b.get("author", "")))
    return not (aa and ab and aa.split(" ")[-1] != ab.split(" ")[-1])


# ----------------------------------------------------------------------- sources
async def _json(client: httpx.AsyncClient, url: str, **params):
    r = await client.get(url, params=params or None)
    r.raise_for_status()
    return r.json()


async def src_gutenberg(client, q: str) -> list[dict]:
    data = await _json(client, f"{GUTENDEX}/books/", search=q)
    out = []
    for b in (data.get("results") or [])[:8]:
        fm = b.get("formats") or {}
        pdf = next((u for k, u in fm.items() if k.startswith("application/pdf")), "")
        epub = next((u for k, u in fm.items() if k.startswith("application/epub+zip")), "")
        url, fmt = (pdf, "pdf") if pdf else (epub, "epub")
        if not url:
            continue
        au = _person(((b.get("authors") or [{}])[0]).get("name", ""))
        out.append({"title": b.get("title", ""), "author": au, "source": "Project Gutenberg", "fmt": fmt, "url": url,
                    "license": "Public domain", "year": ""})
    return out


# --- OpenStax: list endpoint no longer exposes pdf_url; fetch detail pages in parallel --
_openstax: dict = {"at": 0.0, "items": []}
_OPENSTAX_DETAIL_SEMAPHORE = asyncio.Semaphore(8)              # be gentle on openstax.org


async def _openstax_detail(client, page_id: int) -> dict | None:
    async with _OPENSTAX_DETAIL_SEMAPHORE:
        try:
            r = await client.get(f"{OPENSTAX}/apps/cms/api/v2/pages/{page_id}/", timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("openstax detail %s failed: %s", page_id, e)
            return None


async def src_openstax(client, q: str) -> list[dict]:
    if not _openstax["items"] or time.time() - _openstax["at"] > 86400:
        # List endpoint only allows a small set of fields - pdf_url is on the detail page now
        data = await _json(client, f"{OPENSTAX}/apps/cms/api/v2/pages/", type="books.Book", limit=100,
                           fields="title,slug,book_state")
        books = [it for it in (data.get("items") or []) if it.get("book_state") in (None, "live")]
        # Fetch detail pages in parallel to obtain pdf_url for each book
        details = await asyncio.gather(*[_openstax_detail(client, b["id"]) for b in books])
        items = []
        for b, d in zip(books, details):
            if not d:
                continue
            pdf_url = d.get("pdf_url") or d.get("high_resolution_pdf_url") or ""
            hi_url = d.get("high_resolution_pdf_url") or ""
            lo_url = d.get("pdf_url") or ""
            if not (pdf_url or hi_url):
                continue
            au = ""
            authors = d.get("authors") or []
            if authors and isinstance(authors, list) and isinstance(authors[0], dict):
                au = _person(authors[0].get("name", ""))
            items.append({
                "title": d.get("title", b.get("title", "")),
                "author": au or "OpenStax",
                "pdf_url": hi_url or pdf_url,
                "alt_pdf_url": lo_url if hi_url else "",          # fall back to the lower-res version if too big
                "license": f"CC BY (OpenStax, {d.get('license_name', 'free textbook')})",
                "year": str(d.get("publish_date") or "")[:4],
            })
        _openstax.update(at=time.time(), items=items)
        log.info("openstax cache: %d books (with pdf_url)", len(items))
    out = []
    q_tokens = set(_tokens(q))
    for it in _openstax["items"]:
        if not q_tokens:
            continue
        it_tokens = set(_tokens(it["title"])) | set(_tokens(it.get("author", "")))
        if q_tokens & it_tokens:
            out.append({"title": it["title"], "author": it["author"], "source": "OpenStax",
                        "fmt": "pdf", "url": it["pdf_url"],
                        "alt_url": it.get("alt_pdf_url", ""),
                        "license": it["license"], "year": it["year"]})
    return out


# --- Open Textbook Library (UMN): /opentextbooks/textbooks.json -----------------
async def src_open_textbook_library(client, q: str) -> list[dict]:
    """Open Textbook Library (UMN) - peer-reviewed open textbooks.
       The PDF format URL often points to a LibreTexts landing page rather than a direct .pdf file,
       so we set fmt='link' for landing pages (deliver() sends the link) and fmt='pdf' for direct files."""
    data = await _json(client, f"{OPEN_TEXTBOOK_LIBRARY}/opentextbooks/textbooks.json", search=q, limit=10)
    out: list[dict] = []
    for it in (data.get("data") or []):
        if not isinstance(it, dict):
            continue
        # Find a PDF format
        pdf_url, is_direct_pdf = "", False
        for fmt_obj in (it.get("formats") or []):
            if str(fmt_obj.get("type", "")).upper() == "PDF":
                pdf_url = fmt_obj.get("url") or ""
                # If the URL actually ends in .pdf it's a direct download; otherwise it's a landing page
                is_direct_pdf = pdf_url.lower().endswith(".pdf")
                break
        if not pdf_url:
            continue
        # Author extraction (contributors list)
        au = ""
        contributors = it.get("contributors") or []
        if contributors and isinstance(contributors, list):
            first = contributors[0]
            if isinstance(first, dict):
                au = _person(first.get("name", "") or first.get("full_name", ""))
            else:
                au = _person(str(first))
        umn_page = f"{OPEN_TEXTBOOK_LIBRARY}/opentextbooks/textbooks/{it.get('id', '')}"
        out.append({
            "title": it.get("title", ""), "author": au,
            "source": "Open Textbook Library",
            "fmt": "pdf" if is_direct_pdf else "link",
            "url": pdf_url,
            "alt_url": umn_page,
            "license": f"Open textbook (UMN: {it.get('license', 'CC')})",
            "year": str(it.get("copyright_year") or "")[:4],
        })
    return out


async def src_openlibrary(client, q: str) -> list[dict]:
    data = await _json(client, f"{OPENLIBRARY}/search.json", q=q, has_fulltext="true", limit=10,
                       fields="key,title,author_name,ia,ebook_access,first_publish_year")
    out = []
    for d in (data.get("docs") or []):
        if d.get("ebook_access") != "public" or not d.get("ia"):      # borrow-only / print-disabled books are NOT downloadable
            continue
        out.append({"title": d.get("title", ""), "author": (d.get("author_name") or [""])[0], "source": "Open Library",
                    "fmt": "pdf", "url": "", "ia": d["ia"][0], "license": "Public / open access (Internet Archive)",
                    "year": str(d.get("first_publish_year") or "")})
    return out


async def src_archive(client, q: str) -> list[dict]:
    """Internet Archive items whose metadata says public domain / Creative Commons (or items in IA's curated
       public-domain book collections where licenseurl is often missing)."""
    words = " ".join(re.findall(r"[A-Za-z0-9]+", q))
    if not words:
        return []
    query = (f"title:({words}) AND mediatype:texts "
             f"AND (licenseurl:*publicdomain* OR licenseurl:*creativecommons* "
             f"OR (collection:americana AND access-restricted-item:false) "
             f"OR (collection:toronto AND access-restricted-item:false))")
    data = await _json(client, f"{ARCHIVE}/advancedsearch.php", q=query, rows=8, output="json", sort="downloads desc",
                       **{"fl[]": ["identifier", "title", "creator", "licenseurl", "date"]})
    out = []
    for d in ((data.get("response") or {}).get("docs") or []):
        lic = d.get("licenseurl") or ""
        lic = lic[0] if isinstance(lic, list) else lic
        cr = d.get("creator") or ""
        cr = cr[0] if isinstance(cr, list) else cr
        ti = d.get("title") or ""
        ti = ti[0] if isinstance(ti, list) else ti
        out.append({"title": ti, "author": _person(cr), "source": "Open Library", "fmt": "pdf", "url": "", "ia": d.get("identifier"),
                    "license": "Public domain" if "publicdomain" in lic.lower() else "Creative Commons (Internet Archive)",
                    "year": str(d.get("date") or "")[:4]})
    return [o for o in out if o["ia"]]


async def src_oapen(client, q: str) -> list[dict]:
    data = await _json(client, f"{OAPEN}/rest/search", query=f"title:({q})", expand="metadata,bitstreams", limit=8)
    out = []
    for it in (data if isinstance(data, list) else data.get("items", [])):
        bs = [b for b in (it.get("bitstreams") or []) if str(b.get("mimeType", "")).lower() == "application/pdf"
              and str(b.get("bundleName", "ORIGINAL")).upper() == "ORIGINAL"]
        if not bs:
            continue
        b = max(bs, key=lambda x: x.get("sizeBytes") or 0)
        link = b.get("retrieveLink") or ""
        au = ""
        for m in it.get("metadata") or []:
            if m.get("key") == "dc.contributor.author":
                au = _person(m.get("value", "")); break
        out.append({"title": it.get("name", ""), "author": au, "source": "OAPEN", "fmt": "pdf",
                    "url": link if link.startswith("http") else OAPEN + link, "size": b.get("sizeBytes") or 0,
                    "license": "Open access (OAPEN)", "year": ""})
    return out


# --- DOAB: same DSpace REST API shape as OAPEN -------------------------------------
async def src_doab(client, q: str) -> list[dict]:
    """DOAB - Directory of Open Access Books. Peer-reviewed academic books, all open access.
       Uses the same DSpace REST API as OAPEN (they are sister services)."""
    data = await _json(client, f"{DOAB}/rest/search", query=f"title:({q})", expand="metadata,bitstreams", limit=8)
    out = []
    for it in (data if isinstance(data, list) else data.get("items", [])):
        bs = [b for b in (it.get("bitstreams") or [])
              if str(b.get("mimeType", "")).lower() in ("application/pdf", "application/epub+zip")
              and str(b.get("bundleName", "ORIGINAL")).upper() == "ORIGINAL"]
        if not bs:
            continue
        b = max(bs, key=lambda x: x.get("sizeBytes") or 0)
        link = b.get("retrieveLink") or ""
        fmt = "pdf" if "pdf" in str(b.get("mimeType", "")).lower() else "epub"
        au = ""
        for m in it.get("metadata") or []:
            if m.get("key") == "dc.contributor.author":
                au = _person(m.get("value", "")); break
        out.append({"title": it.get("name", ""), "author": au, "source": "DOAB", "fmt": fmt,
                    "url": link if link.startswith("http") else DOAB + link, "size": b.get("sizeBytes") or 0,
                    "license": "Open access (DOAB, peer-reviewed)", "year": ""})
    return out


async def src_google(client, q: str) -> list[dict]:
    params = dict(q=q, filter="free-ebooks", printType="books", maxResults=10)
    key = os.getenv("GOOGLE_BOOKS_API_KEY", "").strip()
    if key:
        params["key"] = key
    data = await _json(client, f"{GOOGLE_BOOKS}/books/v1/volumes", **params)
    out = []
    for it in (data.get("items") or []):
        vi, ai = it.get("volumeInfo") or {}, it.get("accessInfo") or {}
        pdf, epub = ai.get("pdf") or {}, ai.get("epub") or {}
        if pdf.get("isAvailable") and pdf.get("downloadLink"):
            url, fmt = pdf["downloadLink"], "pdf"
        elif epub.get("isAvailable") and epub.get("downloadLink"):
            url, fmt = epub["downloadLink"], "epub"
        else:
            continue
        out.append({"title": vi.get("title", ""), "author": (vi.get("authors") or [""])[0], "source": "Google Books", "fmt": fmt,
                    "url": url, "license": "Public domain" if ai.get("publicDomain") else "Free ebook (Google Books)",
                    "year": str(vi.get("publishedDate", ""))[:4]})
    return out


# --- Wikisource: MediaWiki search API + REST API PDF export ----------------------
async def src_wikisource(client, q: str) -> list[dict]:
    """Wikisource - Wikimedia's free library of source texts. Uses the MediaWiki action API for search
       and the REST API for on-demand PDF export of each book's main page."""
    data = await _json(client, f"{WIKISOURCE}/w/api.php", action="query", list="search",
                       srsearch=q, srnamespace=0, srlimit=10, format="json", srprop="snippet")
    out = []
    for r in (data.get("query", {}).get("search") or []):
        title = r.get("title", "")
        if not title:
            continue
        # REST API: returns a PDF of the page (and any transcluded subpages) on demand.
        # If the page is a single chapter rather than a whole book, the user still gets that chapter's PDF.
        pdf_url = f"{WIKISOURCE}/api/rest_v1/page/pdf/{quote(title, safe='')}"
        out.append({"title": title, "author": "", "source": "Wikisource",
                    "fmt": "pdf", "url": pdf_url,
                    "license": "Public domain / CC BY-SA (Wikimedia)", "year": ""})
    return out


SOURCES = [
    ("OpenStax", src_openstax),
    ("Open Textbook Library", src_open_textbook_library),
    ("Project Gutenberg", src_gutenberg),
    ("OAPEN", src_oapen),
    ("DOAB", src_doab),
    ("Google Books", src_google),
    ("Wikisource", src_wikisource),
    ("Open Library", src_openlibrary),
    ("Internet Archive", src_archive),
]


async def resolve_archive(client, ia_id: str) -> tuple[str, int]:
    """Download URL (+ size) of the best PDF of an Internet Archive item."""
    meta = await _json(client, f"{ARCHIVE}/metadata/{quote(ia_id)}")
    files = [f for f in (meta.get("files") or []) if "pdf" in str(f.get("format", "")).lower() or str(f.get("name", "")).lower().endswith(".pdf")]
    files = [f for f in files if not str(f.get("name", "")).lower().endswith(("_bw.pdf",))] or files
    if not files:
        raise BooksError("no PDF in that archive item")
    order = {"text pdf": 0, "additional text pdf": 1, "image container pdf": 2}
    f = sorted(files, key=lambda x: (order.get(str(x.get("format", "")).lower(), 5), int(x.get("size") or 0)))[0]
    return f"{ARCHIVE}/download/{quote(ia_id)}/{quote(f['name'])}", int(f.get("size") or 0)


# ----------------------------------------------------------------------- search
async def search(query: str) -> tuple[list[dict], dict]:
    """All sources in parallel; returns (ranked candidates, {source: error})."""
    errors: dict[str, str] = {}
    async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, headers=HEADERS) as client:
        async def one(name, fn):
            try:
                return await fn(client, query)
            except Exception as e:                                # one source failing never breaks the search
                errors[name] = str(e)[:120]
                log.warning("books source %s failed: %s", name, e)
                return []
        results = await asyncio.gather(*[one(n, f) for n, f in SOURCES])
    cands = [c for r in results for c in r if c.get("title")]
    for c in cands:
        c["score"] = score(query, c)
        c["rank"] = c["score"] + (0.04 if c["fmt"] == "pdf" else 0) - 0.005 * SOURCE_RANK.get(c["source"], 5)
    cands.sort(key=lambda c: c["rank"], reverse=True)
    groups: list[list[dict]] = []                       # the same book offered by several sources / editions
    for c in cands:
        for g in groups:
            h = g[0]
            if _same_book(c, h):
                g.append(c); break
        else:
            groups.append([c])
    ranked = []
    for g in groups:
        best = g[0]
        best["alts"] = [{k: v for k, v in x.items() if k != "alts"} for x in g[1:6]]   # tried in this order if the best one gives no PDF
        ranked.append(best)
    return [c for c in ranked if c["score"] >= CUTOFF], errors


# ----------------------------------------------------------------------- download + send
def _download_sync(url: str, want: str) -> tuple[bytes | None, str]:
    """Returns (bytes or None, reason). None + 'too big' means: send the link instead."""
    cap = int(MAX_MB * 1024 * 1024)
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
    magic = bytes(buf[:4])
    if want == "pdf" and magic != b"%PDF":
        return None, "not a PDF"
    if want == "epub" and magic[:2] != b"PK":
        return None, "not an EPUB"
    return bytes(buf), "ok"


def _graph():
    base = (os.getenv("WHATSAPP_GRAPH_BASE") or "https://graph.facebook.com/v21.0").rstrip("/")
    token = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
    pid = os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID")
    if not token or not pid:
        raise BooksError("WhatsApp credentials are missing on the server.")
    return base, {"Authorization": f"Bearer {token}"}, pid


def _send_document_sync(phone: str, data: bytes, mime: str, filename: str, caption: str) -> None:
    base, headers, pid = _graph()
    up = requests.post(f"{base}/{pid}/media", headers=headers, data={"messaging_product": "whatsapp", "type": mime},
                       files={"file": (filename, data, mime)}, timeout=120)
    if up.status_code >= 400:
        raise BooksError(f"media upload failed: HTTP {up.status_code} {up.text[:150]}")
    r = requests.post(f"{base}/{pid}/messages", headers={**headers, "Content-Type": "application/json"}, timeout=30,
                      json={"messaging_product": "whatsapp", "to": phone, "type": "document",
                            "document": {"id": up.json().get("id"), "filename": filename, "caption": caption[:1000]}})
    if r.status_code >= 400:
        raise BooksError(f"media send failed: HTTP {r.status_code} {r.text[:150]}")


def _unicode_font():
    """DejaVu Sans when the server has it (accents, quotes, dashes); otherwise Helvetica + latin-1."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for reg, bold in (("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
                      ("/usr/share/fonts/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf"),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("BookFont", reg)); pdfmetrics.registerFont(TTFont("BookFont-Bold", bold))
                return "BookFont", "BookFont-Bold", True
            except Exception:
                pass
    return "Times-Roman", "Times-Bold", False


def epub_to_pdf(data: bytes, title: str, author: str = "") -> bytes | None:
    """Builds a readable PDF from a (public-domain) EPUB: title page, headings, paragraphs, page numbers. None on failure."""
    import io
    import posixpath
    import zipfile
    from urllib.parse import unquote
    from xml.sax.saxutils import escape
    from bs4 import BeautifulSoup
    try:
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.lib.pagesizes import A5
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        log.warning("reportlab is not installed: cannot build a PDF from the EPUB (pip install reportlab)")
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        names = set(z.namelist())
        order: list[str] = []
        try:
            opf_path = BeautifulSoup(z.read("META-INF/container.xml"), "lxml-xml").find("rootfile")["full-path"]
            opf = BeautifulSoup(z.read(opf_path), "lxml-xml")
            base = posixpath.dirname(opf_path)
            manifest = {i.get("id"): i.get("href") for i in opf.find_all("item")}
            for ref in opf.find_all("itemref"):
                href = manifest.get(ref.get("idref"))
                if href:
                    order.append(posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0]))))
        except Exception:
            order = []
        order = [n for n in order if n in names] or sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        blocks, total = [], 0
        for n in order:
            soup = BeautifulSoup(z.read(n), "lxml")
            for t in soup.find_all(["h1", "h2", "h3", "h4", "p", "blockquote", "pre", "li"]):
                if t.find(["p", "h1", "h2", "h3", "h4", "blockquote", "pre", "li"]):
                    continue                               # a wrapper: its children are handled on their own
                txt = " ".join(t.get_text(" ").split())
                if not txt:
                    continue
                total += len(txt)
                if total > 3_000_000:
                    return None                            # absurdly long: let the caller send the EPUB / link instead
                blocks.append((t.name, txt))
        if not blocks:
            return None
        font, bold, uni = _unicode_font()
        fix = (lambda x: x) if uni else (lambda x: x.encode("latin-1", "replace").decode("latin-1"))
        st = {
            "title": ParagraphStyle("t", fontName=bold, fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=10),
            "author": ParagraphStyle("a", fontName=font, fontSize=13, leading=18, alignment=TA_CENTER),
            "note": ParagraphStyle("n", fontName=font, fontSize=8, leading=11, alignment=TA_CENTER),
            "h": ParagraphStyle("h", fontName=bold, fontSize=14, leading=18, spaceBefore=14, spaceAfter=8),
            "p": ParagraphStyle("p", fontName=font, fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5),
            "q": ParagraphStyle("q", fontName=font, fontSize=9, leading=12.5, leftIndent=12, spaceAfter=5),
            "pre": ParagraphStyle("pre", fontName=font, fontSize=8.5, leading=11, leftIndent=8, spaceAfter=3),
        }
        story = [Spacer(1, 50 * mm), Paragraph(escape(fix(title)), st["title"])]
        if author:
            story.append(Paragraph(escape(fix(author)), st["author"]))
        story += [Spacer(1, 12 * mm), Paragraph("Free e-book converted to PDF. Public domain.", st["note"]), PageBreak()]
        for name, txt in blocks:
            kind = "h" if name in ("h1", "h2", "h3", "h4") else ("q" if name in ("blockquote", "li") else ("pre" if name == "pre" else "p"))
            story.append(Paragraph(escape(fix(txt)), st[kind]))

        def footer(canvas, doc):
            canvas.saveState(); canvas.setFont(font, 7.5)
            canvas.drawCentredString(A5[0] / 2, 8 * mm, str(doc.page)); canvas.restoreState()
        buf = io.BytesIO()
        SimpleDocTemplate(buf, pagesize=A5, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                          title=title[:100], author=author[:100]).build(story, onFirstPage=footer, onLaterPages=footer)
        return buf.getvalue()
    except Exception as e:
        log.warning("EPUB -> PDF failed: %s", e)
        return None


async def _try(cand: dict) -> tuple[bytes | None, str, str]:
    """Download one candidate. Returns (bytes or None, url, reason)."""
    url = cand.get("url") or ""
    try:
        if cand.get("ia"):
            async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, headers=HEADERS) as client:
                url, size = await resolve_archive(client, cand["ia"])
            if size and size > MAX_MB * 1024 * 1024:
                return None, url, "too big"
        data, why = await asyncio.to_thread(_download_sync, url, cand["fmt"])
        # OpenStax fallback: high-res PDF too big -> try low-res alt_url
        if not data and cand.get("alt_url") and why == "too big":
            data, why = await asyncio.to_thread(_download_sync, cand["alt_url"], cand["fmt"])
            url = cand["alt_url"] if data else url
        return data, url, why
    except Exception as e:
        return None, url, str(e)[:80]


def _caption(c: dict, extra: str = "") -> str:
    return (f"📚 *{c['title']}*" + (f"\n✍️ {c['author']}" if c.get("author") else "") +
            f"\n🏷 Source: {c['source']} - {c['license']}" + (f"\n{extra}" if extra else ""))


async def _send_file(phone: str, c: dict, data: bytes, ext: str, link: str, say, extra: str = "") -> bool:
    mime = "application/pdf" if ext == "pdf" else "application/epub+zip"
    try:
        await asyncio.to_thread(_send_document_sync, phone, data, mime, _safe_filename(c["title"], ext), _caption(c, extra))
        return True
    except Exception as e:
        log.exception("document send failed: %s", e)
        await say(f"📎 {_caption(c)}\n\nI couldn't attach the file, but you can download it here:\n{link}")
        return True                                              # the user got the link: nothing more to try


async def deliver(phone: str, cand: dict, say) -> None:
    """Gets the book as a real PDF, whatever it takes (all legal):
       1. If cand is a 'link' type (landing page, not a direct PDF), send the link directly.
       2. Try the chosen source's PDF, then the same book's PDF from every other source.
       3. No PDF anywhere -> a PDF built from the public-domain EPUB (Project Gutenberg).
       4. Otherwise the EPUB itself; otherwise a direct link."""
    # If this candidate is explicitly a "link" (e.g., Open Textbook Library landing pages), don't try to download
    if cand.get("fmt") == "link":
        link = cand.get("url") or cand.get("alt_url") or ""
        await say(f"📎 {_caption(cand)}\n\nHere is the link (you may need to click Download PDF on the page):\n{link}")
        return

    group = [cand] + list(cand.get("alts") or [])
    pdfs = [c for c in group if c["fmt"] == "pdf"]
    epubs = [c for c in group if c["fmt"] == "epub"]
    failed: list[tuple[dict, str, str]] = []
    for i, c in enumerate(pdfs):
        data, url, why = await _try(c)
        if data:
            await _send_file(phone, c, data, "pdf", url, say)
            return
        log.info("no PDF from %s (%s): %s", c["source"], why, url)
        failed.append((c, url, why))
    for c in epubs:                                              # make a PDF ourselves from a public-domain EPUB
        if c["source"] == "Project Gutenberg" or str(c.get("license", "")).lower().startswith("public domain"):
            data, url, why = await _try(c)
            if not data:
                failed.append((c, url, why)); continue
            await say("⚙️ No ready-made PDF exists for this book, so I'm making a PDF from the free public-domain edition (about a minute)...")
            pdf = await asyncio.to_thread(epub_to_pdf, data, c["title"], c.get("author", ""))
            if pdf and len(pdf) <= MAX_MB * 1024 * 1024:
                await _send_file(phone, c, pdf, "pdf", url, say, "ℹ️ Converted to PDF from the original EPUB.")
                return
            await _send_file(phone, c, data, "epub", url, say)
            await say("ℹ️ I couldn't turn it into a PDF, so this is the *EPUB* - open it with Google Play Books, Apple Books or any e-reader app.")
            return
    for c in epubs:                                              # not public domain: send the EPUB as it is
        data, url, why = await _try(c)
        if data:
            await _send_file(phone, c, data, "epub", url, say)
            await say("ℹ️ This one only exists as an *EPUB* - open it with Google Play Books, Apple Books or any e-reader app.")
            return
        failed.append((c, url, why))
    c, url, why = failed[0] if failed else (cand, cand.get("url") or "", "unavailable")
    if not url and cand.get("ia"):
        url = f"{ARCHIVE}/details/{quote(cand['ia'])}"
    note = "it's too large for WhatsApp" if why == "too big" else "I couldn't download it myself"
    await say(f"📎 {_caption(cand)}\n\nHere is the direct link ({note}):\n{url}")


# ----------------------------------------------------------------------- limits / logging
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


# ----------------------------------------------------------------------- conversation
PROMPT = ("📚 *Free Books*\n\nSend me the *exact name* of the book (add the author if you know it), e.g.\n"
          "• Pride and Prejudice\n• University Physics Volume 1\n• Calculus Volume 1\n\n"
          "I search OpenStax, Open Textbook Library, Project Gutenberg, OAPEN, DOAB,\n"
          "Google Books, Wikisource, Open Library and the Internet Archive for a *free, legal* copy.\n"
          "Send *exit* to cancel.")
_NCERT = re.compile(r"\b(ncert|cbse|class\s*(\d{1,2})|std\s*\d{1,2})\b", re.I)


def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


async def start(phone: str, arg: str, say) -> bool:
    """/books [book name]. True = the job is over (state cleared)."""
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
    if arg and arg.strip():
        return await handle(phone, arg, say)
    await say(PROMPT)
    return False


def _pick_text(cands: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        lines.append(f"*{i}.* {c['title']}" + (f" - {c['author']}" if c.get("author") else "") +
                     f"\n    {c['fmt'].upper()} • {c['source']}" + (f" • {c['year']}" if c.get("year") else ""))
    return "I found these. Reply with the *number* of the one you want (or type another book name, or *exit*):\n\n" + "\n".join(lines)


async def handle(phone: str, text: str, say) -> bool:
    import database as db
    text = (text or "").strip()
    low = text.lower()
    if low in EXIT_WORDS:
        await asyncio.to_thread(_save, phone, {})
        await say("👋 Cancelled.")
        return True
    state = (await asyncio.to_thread(db.get_user, phone)).get("state") or {}
    cands = state.get("cands") or []
    try:
        if state.get("step") == "pick" and text.isdigit():
            n = int(text)
            if not 1 <= n <= len(cands):
                await say(f"🤔 Please reply with a number from 1 to {len(cands)} (or *exit*).")
                return False
            await say(f"⏳ Getting *{cands[n - 1]['title']}*...")
            await deliver(phone, cands[n - 1], say)
            await asyncio.to_thread(_save, phone, {})
            _log(phone, title=cands[n - 1]["title"], source=cands[n - 1]["source"])
            await say("✅ Done! Send */books* for another book, or */menu* for the menu.")
            return True
        if len(text) < 2:
            await say("🤔 Please send the book's name (or *exit*).")
            return False
        if not _quota_ok(phone):
            await asyncio.to_thread(_save, phone, {})
            await say(f"⏳ You've reached today's limit of {MAX_PER_DAY} books. Please try again tomorrow.")
            return True
        await say(f"🔎 Searching free libraries for *{text[:80]}*...")
        found, errors = await search(text)
        if found and found[0]["score"] >= STRONG and (len(found) == 1 or found[1]["score"] <= found[0]["score"] - 0.08
                                                       or _norm(found[1]["title"]) == _norm(found[0]["title"])):
            top = found[0]
            await say(f"✅ Found: *{top['title']}* ({top['source']}). Sending it...")
            await deliver(phone, top, say)
            await asyncio.to_thread(_save, phone, {})
            _log(phone, title=top["title"], source=top["source"], query=text[:80])
            await say("✅ Done! Send */books* for another book, or */menu* for the menu.")
            return True
        if found:
            shortlist = found[:5]
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "pick", "cands": shortlist})
            await say(_pick_text(shortlist))
            return False
        _refund(phone)
        extra = ""
        if _NCERT.search(text):
            extra = f"\n\n📘 For NCERT / CBSE school textbooks, the official free PDFs are here (pick class and subject):\n{NCERT_PAGE}"
        if len(errors) == len(SOURCES):
            extra += "\n\n⚠️ The book libraries didn't respond just now - please try again in a minute."
        # Always offer a legal "find it" option for books we couldn't locate a free copy of
        worldcat = f"https://search.worldcat.org/search?q={quote_plus(text)}"
        gbooks_search = f"https://www.google.com/search?tbm=bks&q={quote_plus(text)}"
        extra += (f"\n\n📚 *Find it at a library near you:* {worldcat}"
                  f"\n🔍 *Search Google Books:* {gbooks_search}")
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
        await say("😕 I couldn't find a *free, legal* copy of that. It may still be under copyright (then it's only available to buy or borrow).\n"
                  "Try the exact title (and author), or send another book name, or *exit*." + extra)
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
    if len(argv) >= 3 and argv[1] == "probe":
        async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, headers=HEADERS) as client:
            for name, fn in SOURCES:
                try:
                    res = await fn(client, argv[2])
                    print(f"{name:22} OK  {len(res)} result(s)")
                    for c in res[:3]:
                        print("      ", c["title"][:60], "|", c.get("author", "")[:25], "|", c["fmt"], "|", (c.get("url") or ("ia:" + str(c.get("ia"))))[:70])
                except Exception as e:
                    print(f"{name:22} FAILED  {e!r}"[:200])
    elif len(argv) >= 3 and argv[1] == "get":
        found, errors = await search(argv[2])
        for c in found[:5]:
            print(c["score"], c["title"], "|", c["source"], c["fmt"])
        if found:
            url = found[0].get("url")
            if found[0].get("ia"):
                async with httpx.AsyncClient(follow_redirects=True, timeout=TIMEOUT, headers=HEADERS) as client:
                    url, _ = await resolve_archive(client, found[0]["ia"])
            data, why = await asyncio.to_thread(_download_sync, url, found[0]["fmt"])
            if data:
                fn = _safe_filename(found[0]["title"], found[0]["fmt"])
                open(fn, "wb").write(data); print("saved", fn, len(data), "bytes")
            else:
                print("not downloaded:", why, url)
    else:
        print('usage: python books.py probe "<book name>" | get "<book name>"')


if __name__ == "__main__":  # pragma: no cover
    import sys
    asyncio.run(_cli(sys.argv))
