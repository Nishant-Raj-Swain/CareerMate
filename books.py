from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import time
from datetime import date
from urllib.parse import quote, quote_plus, urljoin

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








# ===========================================================================
# DISCOVERY SOURCES (inlined scrapers)
# ===========================================================================
# These two scrapers power the /books 'Browse by Topic' feature and the
# recommendation fallback. They are inlined (not imported) so the bot's books
# feature is self-contained in a single file.
#
# - _FreeBookCentreScraper: scrapes freebookcentre.net (curated index of free
#   programming/technical books). Hides download URLs behind JavaScript, so we
#   use it only for taxonomy + book metadata; PDFs are delivered via the 9 legal
#   sources in SOURCES above.
# - _FreeComputerBooksScraper: scrapes freecomputerbooks.com (similar curated
#   index of free computer/math/tech books). Same approach.

class _FreeBookCentreScraper:
    """Inlined scraper for freebookcentre.net.

    Used as a discovery layer: returns book titles + authors, which books.py then
    searches for in the 9 legal sources to deliver actual PDFs.
    """
    BASE = 'https://www.freebookcentre.net'
    HEADERS = {'User-Agent': 'CareerMateBot/1.0 (educational WhatsApp bot; free/legal books only)', 'Accept': 'text/html'}
    TIMEOUT = 20.0
    _CACHE_TTL = 7 * 86400  # 1 week

    CATEGORIES: list[tuple[str, str]] = [
    ("Computer Science",       "/CompuScience/compscCategory.html"),
    ("Programming Languages",  "/Language/langCategory.html"),
    ("Java / Advanced Java",   "/JavaTech/javaCategory.html"),
    ("Unix / Linux",           "/UnixCategory/unixCategory.html"),
    ("Web Technology",         "/Web/webCategory.html"),
    ("Database Systems",       "/Database/dbCategory.html"),
    ("Medical",                "/medical_text_books_journals/medical_text_books_online.html"),
    ("Microsoft Technology",   "/MicroSoftTech/microsoftCategory.html"),
    ("Special Topics",        "/SpecialCat/specialCategory.html"),
    ("Electronics",            "/electronics_communication_books/electronics_communication_books.html"),
    ("File System Specs",      "/file_system_specification/file_system_specifications.html"),
    ("Graphics Tools",         "/graphics-design-books/web-design-tools.html"),
    ("Mobile Technology",      "/mobile-technology/mobile-technology-books.html"),
    ("Networking",             "/Networking/networkCategory.html"),
    ("Mathematics",            "/SpecialCat/Free-Mathematics-Books-Download.html"),
    ("Physics",                "/Physics/Physics-Books-Online.html"),
    ("Hardware Buses",         "/HardwareBus/Hardware-Buses-Books.html"),
    ("Chemistry",              "/Chemistry/Chemistry-Books-Online.html"),
    ("Biology",                "/Biology/Biology-Books-Online.html"),
    ("Law",                    "/Law/Law-Books.html"),
    ("Electronics Engineering","/Electronics/Engineering-Books-Online.html"),
    ("Business & Finance",     "/Business/Business-and-Finance-Books.html"),
    ("Literature & Fiction",   "/Fiction/Fiction-Books.html"),
    ("Cooking, Food & Drink",  "/Cooking/Cooking-Food-Drink-Books.html"),
    ("Earth Sciences",         "/EarthSciences/Earth-Sciences-Books.html"),
    ("Aerospace Engineering",  "/AeroSpace/AeroSpace-Engineering-Books.html"),
    ("Mechanical Engineering", "/Mechanical/Mechanical-Engineering-Books.html"),
    ("Electrical Engineering", "/Electrical/Electrical-Engineering-Books.html"),
    ("Chemical Engineering",   "/Chemical/Chemical-Engineering-Books.html"),
    ("Civil Engineering",      "/Civil/Civil-Engineering-Books.html"),
    ]

    _subtopic_cache: dict = {}
    _book_cache: dict = {}

    @staticmethod
    def _log(msg, *args):
        log.info("fbc: " + msg, *args)

    async def _fetch(self, client: httpx.AsyncClient, path: str) -> str:
        url = urljoin(self.BASE + "/", path.lstrip("/"))
        r = await client.get(url, timeout=self.TIMEOUT)
        r.raise_for_status()
        return r.text

    def _parse_subtopics(self, html: str, category_path: str) -> list[tuple[str, str]]:
        """Extract subtopic (name, url) pairs from a category page.

        freebookcentre.net category pages list subtopics as <a href="/JavaTech/Free-Advanced-Java-books-download.html">Advanced Java</a>
        We filter to subtopic pages (URLs containing 'Books' or 'books' and ending in .html),
        skip navigation and footer links, and dedupe.
        """
        # Pull all anchor tags with their href and visible text
        pairs = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>\s*([^<]+\w[^<]*)</a>', html)
        seen = set()
        out = []
        for href, text in pairs:
            text = text.strip()
            # Skip empty, navigation, or non-subtopic links
            if not text or len(text) > 60:
                continue
            # Subtopic URLs look like /JavaTech/Free-Advanced-Java-books-download.html
            # i.e. they live in a category subfolder and end in .html, and contain "book"
            href_lower = href.lower()
            if not href.startswith("/"):
                continue
            if not href_lower.endswith(".html"):
                continue
            if "book" not in href_lower:
                continue
            # Skip footer / about / contact / privacy / sitemap
            if any(skip in href_lower for skip in ("about", "contact", "privacy", "terms", "sitemap", "link_us", "aboutus")):
                continue
            # Must be in the same subfolder as the category (e.g. /JavaTech/ for Java category)
            # We compare the first path segment
            cat_seg = category_path.strip("/").split("/")[0] if category_path else ""
            href_seg = href.strip("/").split("/")[0] if href else ""
            if cat_seg and href_seg and cat_seg != href_seg:
                # Allow cross-folder subtopic links (some categories link to /SpecialCat/...)
                # but be conservative: only accept if BOTH segments look like content folders
                if href_seg in ("aboutus", "contactus", "link_us"):
                    continue
            key = (text, href)
            if key in seen:
                continue
            seen.add(key)
            out.append((text, href))
        return out


    def _parse_books(self, html: str, subtopic_url: str) -> list[dict]:
        """Extract book entries from a subtopic page.

        Each book block looks like:
          <div itemtype="http://schema.org/Book" itemscope>
            <a itemprop="image" href="../Java-Books-Download/Java-Notes-for-Professionals.html">...</a>
            <a itemprop="name" href="../Java-Books-Download/Java-Notes-for-Professionals.html">Java Notes for Professionals</a>
            <span itemprop="description">...</span>
            <span itemprop="author">GoalKicker.com</span>
            <span itemprop="numberOfPages">976</span>
          </div>

        We extract: title, author, pages, detail_url.
        """
        # The book detail link is on the itemprop="name" anchor.
        # Pattern: <a [^>]*itemprop="name"[^>]*href="...">Title</a>
        # OR:      <a [^>]*href="..."[^>]*itemprop="name"[^>]*>Title</a>
        title_blocks = re.findall(
            r'<a[^>]*itemprop="name"[^>]*href="([^"]+)"[^>]*>\s*([^<]+\w[^<]*)</a>',
            html, re.IGNORECASE
        )
        if not title_blocks:
            # Fallback: href first, then itemprop
            title_blocks = re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*itemprop="name"[^>]*>\s*([^<]+\w[^<]*)</a>',
                html, re.IGNORECASE
            )

        # Find authors (itemprop="author") — text content of the span/paragraph
        authors = re.findall(r'itemprop="author"[^>]*>([^<]+)', html)
        # Find page counts (itemprop="numberOfPages")
        pages = re.findall(r'itemprop="numberOfPages"[^>]*>([^<]+)', html)

        out = []
        for i, (detail_url, title) in enumerate(title_blocks):
            title = title.strip()
            if not title or len(title) > 200:
                continue
            author = authors[i].strip() if i < len(authors) else ""
            page_count = pages[i].strip() if i < len(pages) else ""
            # Clean author: strip "Author(s):" prefix if present
            author = re.sub(r'^Author\(s\):\s*', '', author, flags=re.IGNORECASE).strip()
            # Build full URL for the detail page
            full_url = urljoin(self.BASE + "/", detail_url.lstrip("/"))
            # fbc_id is the slug in the URL (e.g. Java-Notes-for-Professionals)
            fbc_id = detail_url.rsplit("/", 1)[-1].replace(".html", "")
            out.append({
                "title": title,
                "author": author,
                "pages": page_count,
                "detail_url": full_url,
                "fbc_id": fbc_id,
            })
        return out


    async def get_categories(self) -> list[tuple[str, str]]:
        """Returns the static list of (category_name, category_path) tuples."""
        return list(self.CATEGORIES)

    async def get_subtopics(self, category_path: str, *, force_refresh: bool = False) -> list[tuple[str, str]]:
        """Returns [(subtopic_name, subtopic_url), ...] for a category page. Cached weekly."""
        cached = self._subtopic_cache.get(category_path)
        if cached and not force_refresh and time.time() - cached[0] < self._CACHE_TTL:
            return cached[1]
        async with httpx.AsyncClient(follow_redirects=True, headers=self.HEADERS) as client:
            html = await self._fetch(client, category_path)
        subs = self._parse_subtopics(html, category_path)
        self._subtopic_cache[category_path] = (time.time(), subs)
        self._log("subtopics for %s: %d", category_path, len(subs))
        return subs

    async def get_top_books(self, subtopic_url: str, n: int = 5, *, force_refresh: bool = False) -> list[dict]:
        """Returns the top N books from a subtopic page. Cached weekly."""
        cached = self._book_cache.get(subtopic_url)
        if cached and not force_refresh and time.time() - cached[0] < self._CACHE_TTL:
            return cached[1][:n]
        async with httpx.AsyncClient(follow_redirects=True, headers=self.HEADERS) as client:
            html = await self._fetch(client, subtopic_url)
        books = self._parse_books(html, subtopic_url)
        self._book_cache[subtopic_url] = (time.time(), books)
        self._log("books for %s: %d total", subtopic_url, len(books))
        return books[:n]


class _FreeComputerBooksScraper:
    """Inlined scraper for freebookcentre.net.

    Used as a discovery layer: returns book titles + authors, which books.py then
    searches for in the 9 legal sources to deliver actual PDFs.
    """
    BASE = 'https://freecomputerbooks.com'
    HEADERS = {'User-Agent': 'CareerMateBot/1.0 (educational WhatsApp bot; free/legal books only)', 'Accept': 'text/html'}
    TIMEOUT = 20.0
    _CACHE_TTL = 7 * 86400  # 1 week

    CATEGORIES: list[tuple[str, str]] = [
    ("Computer Engineering", "/eeCategory.html"),
    ("Computer Languages",   "/langCategory.html"),
    ("Computer Networks",   "/networkCategory.html"),
    ("Computer Science",    "/compscCategory.html"),
    ("Data Science / Database", "/dbCategory.html"),
    ("Java and Jakarta EE", "/javaCategory.html"),
    ("Linux and Unix",      "/unixCategory.html"),
    ("Mathematics",         "/mathCategory.html"),
    ("Microsoft and .NET",  "/microsoftCategory.html"),
    ("Software Engineering","/softwareCategory.html"),
    ("Special Topics",     "/specialCategory.html"),
    ("Web Programming",     "/webCategory.html"),
    ]

    _subtopic_cache: dict = {}
    _book_cache: dict = {}

    @staticmethod
    def _log(msg, *args):
        log.info("fcb: " + msg, *args)

    async def _fetch(self, client: httpx.AsyncClient, path: str) -> str:
        url = urljoin(self.BASE + "/", path.lstrip("/"))
        r = await client.get(url, timeout=self.TIMEOUT)
        r.raise_for_status()
        return r.text

    def _parse_subtopics(self, html: str, category_path: str) -> list[tuple[str, str]]:
        """Extract (subtopic_name, subtopic_url) pairs from a category landing page.

        freecomputerbooks.com category pages have:
          1. A sidebar with cross-category nav (Math, Python, etc.) — we EXCLUDE these.
          2. A main content area listing actual subtopics for THIS category
             (e.g. for Java: Basic Java Programming -> /javaBasicBooks.html,
                                        Advanced Java -> /javaAdvancedBooks.html).

        We extract the real subtopics by:
          - Deriving the category prefix from the category_path (e.g. /javaCategory.html -> 'java')
          - Only keeping subtopic URLs whose slug starts with that prefix
            (e.g. /javaBasicBooks.html matches, /webHtmlBooks.html doesn't)
        """
        # Derive the category prefix from the category_path
        # /javaCategory.html -> 'java'
        # /langCategory.html -> 'lang'
        # /compscCategory.html -> 'compsc'
        # /eeCategory.html -> 'ee'
        cat_filename = category_path.rsplit("/", 1)[-1].lower()  # "javacategory.html"
        cat_prefix = cat_filename.replace("category.html", "").replace(".html", "")
        if not cat_prefix:
            return []

        pairs = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]{2,80})</a>', html)
        seen = set()
        out = []
        for href, text in pairs:
            text = text.strip()
            if not text or len(text) > 80:
                continue
            # Normalize: convert absolute URLs to relative
            if href.startswith(self.BASE):
                href = href[len(self.BASE):]
            href_lower = href.lower()
            # Skip external links
            if href.startswith("http"):
                continue
            # Skip category landing pages
            if href_lower.endswith("category.html"):
                continue
            # Skip footer / about / contact / privacy / sitemap
            if any(skip in href_lower for skip in ("about", "contact", "privacy", "terms", "sitemap", "tradepub", "amazon")):
                continue
            # Skip cross-promo links
            if any(skip in text.lower() for skip in ("magazine", "research library", "subscription", "tradepub", "bargin")):
                continue
            # Must be an .html subpage
            if not href_lower.endswith(".html"):
                continue
            # Must contain "books.html" (all subtopic pages do)
            if "books.html" not in href_lower:
                continue
            # KEY FILTER: the subtopic URL slug must start with this category's prefix
            # e.g. for category_prefix='java', accept '/javaBasicBooks.html' but reject '/webHtmlBooks.html'
            # We compare the leading part of the filename
            filename = href_lower.rsplit("/", 1)[-1].replace(".html", "")  # e.g. "javabasicbooks"
            if not filename.startswith(cat_prefix):
                continue
            key = (text, href)
            if key in seen:
                continue
            # Also dedupe by URL (some subtopic pages are linked multiple times
            # with different display text — keep the first, which is usually the
            # descriptive title)
            if href in {h for _, h in out}:
                continue
            seen.add(key)
            out.append((text, href))
        return out


    def _parse_books(self, html: str, subtopic_url: str) -> list[dict]:
        """Extract book entries from a subtopic page.

        Book entries on freecomputerbooks.com look like:
          <li>
            <div>
              <img src="covers/Book-Title_43x55.png"/>
              <a href="/Book-Title.html">Book Title (Author Name)</a>
              <p>Description...</p>
            </div>
          </li>

        We extract: title (with author if present in parentheses), detail_url.
        The author is often embedded in the title text after the title, like
        "Book Title (Author Name)" or "Book Title - Author Name".
        """
        out = []
        # Find each <li><div>...<img>...<a href="...">Title</a>...</div></li> block
        # Strategy: find all <a href="/Some-Book.html">Title Text</a> that come
        # after an <img> tag whose src starts with "covers/" — that's the book cover
        # and the anchor immediately after is the book title link.

        # Pattern: <img src="covers/..." ...> <a href="...">Title</a>
        # Allow whitespace/newlines between tags
        pattern = re.compile(
            r'<img[^>]+src="covers/[^"]+"[^>]*/?>\s*<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE | re.DOTALL
        )
        for match in pattern.finditer(html):
            detail_url = match.group(1).strip()
            title_text = match.group(2).strip()
            if not title_text or len(title_text) > 200:
                continue
            # The title often contains the author in parens: "Book Title (Author Name)"
            # or after a dash: "Book Title - Author Name"
            author = ""
            # Try parens first
            m = re.search(r"\(([^)]+)\)\s*$", title_text)
            if m:
                author = m.group(1).strip()
                title_text = title_text[:m.start()].strip()
            else:
                # Try em-dash or hyphen (but only if the part after looks like a name, not a subtitle)
                m = re.search(r"\s+[\u2013\u2014-]\s+([^-\u2013\u2014]{3,60})$", title_text)
                if m:
                    candidate = m.group(1).strip()
                    # Heuristic: a name has 1-4 words, no periods, no digits
                    words = candidate.split()
                    if 1 <= len(words) <= 4 and not re.search(r"\d", candidate) and "." not in candidate:
                        author = candidate
                        title_text = title_text[:m.start()].strip()
            # Build full URL
            full_url = urljoin(self.BASE + "/", detail_url.lstrip("/"))
            out.append({
                "title": title_text,
                "author": author,
                "pages": "",          # freecomputerbooks doesn't list page counts in the subtopic view
                "detail_url": full_url,
                "fcb_id": detail_url.rsplit("/", 1)[-1].replace(".html", ""),
            })
        return out


    async def get_categories(self) -> list[tuple[str, str]]:
        """Returns the static list of (category_name, category_path) tuples."""
        return list(self.CATEGORIES)

    async def get_subtopics(self, category_path: str, *, force_refresh: bool = False) -> list[tuple[str, str]]:
        """Returns [(subtopic_name, subtopic_url), ...] for a category page. Cached weekly."""
        cached = self._subtopic_cache.get(category_path)
        if cached and not force_refresh and time.time() - cached[0] < self._CACHE_TTL:
            return cached[1]
        async with httpx.AsyncClient(follow_redirects=True, headers=self.HEADERS) as client:
            html = await self._fetch(client, category_path)
        subs = self._parse_subtopics(html, category_path)
        self._subtopic_cache[category_path] = (time.time(), subs)
        self._log("subtopics for %s: %d", category_path, len(subs))
        return subs

    async def get_top_books(self, subtopic_url: str, n: int = 5, *, force_refresh: bool = False) -> list[dict]:
        """Returns the top N books from a subtopic page. Cached weekly."""
        cached = self._book_cache.get(subtopic_url)
        if cached and not force_refresh and time.time() - cached[0] < self._CACHE_TTL:
            return cached[1][:n]
        async with httpx.AsyncClient(follow_redirects=True, headers=self.HEADERS) as client:
            html = await self._fetch(client, subtopic_url)
        books = self._parse_books(html, subtopic_url)
        self._book_cache[subtopic_url] = (time.time(), books)
        self._log("books for %s: %d total", subtopic_url, len(books))
        return books[:n]


# Module-level instances of each scraper (the bot uses these everywhere)
_FBC = _FreeBookCentreScraper()
_FCB = _FreeComputerBooksScraper()
# ===========================================================================
# ===========================================================================
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
PROMPT = ("📚 *Free Books*\n\nChoose an option:\n"
          "*1.* Browse by topic (Categories → Subtopics → Top 5 books)\n"
          "*2.* Search by book name + author\n\n"
          "Reply with *1* or *2* (or send *exit* to cancel).\n\n"
          "I search OpenStax, Open Textbook Library, Project Gutenberg, OAPEN, DOAB,\n"
          "Google Books, Wikisource, Open Library and the Internet Archive for *free, legal* copies.")
_TOPIC_PROMPT = ("📚 *Browse by Topic*\n\n"
                  "Send the *number* of a category (or *back* for the menu, *exit* to cancel):\n\n")
_SUBTOPIC_PROMPT = ("📚 *{cat}*\n\n"
                    "Send the *number* of a subtopic (or *back* for categories, *exit* to cancel):\n\n")
_NCERT = re.compile(r"\b(ncert|cbse|class\s*(\d{1,2})|std\s*\d{1,2})\b", re.I)


def is_active(state: dict | None) -> bool:
    return bool(state) and state.get("mode") == MODE


async def start(phone: str, arg: str, say) -> bool:
    """/books [book name]. True = the job is over (state cleared)."""
    await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
    if arg and arg.strip():
        # If user typed `/books pride and prejudice`, treat it as a direct name search
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
        return await handle(phone, arg, say)
    await say(PROMPT)
    return False


def _pick_text(cands: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        lines.append(f"*{i}.* {c['title']}" + (f" - {c['author']}" if c.get("author") else "") +
                     f"\n    {c['fmt'].upper()} • {c['source']}" + (f" • {c['year']}" if c.get("year") else ""))
    return "I found these. Reply with the *number* of the one you want (or type another book name, or *exit*):\n\n" + "\n".join(lines)


# ----------------------------------------------------------------------- topic browsing
# State machine for topic mode:
#   step=menu        -> user picks 1 (topic) or 2 (search)
#   step=cat         -> user picks a category number (paginated 10/page)
#   step=sub         -> user picks a subtopic number (paginated 10/page)
#   step=topic_books -> bot fetches top 5 books, runs legal search() for each, delivers PDFs
_TOPIC_PAGE_SIZE = 10


async def _show_categories(phone: str, say, page: int = 0) -> None:
    """Send a paginated list of categories to the user.

    Categories are merged from BOTH discovery sources (freebookcentre.net and
    freecomputerbooks.com). Each entry is tagged with its source so the bot
    can route to the correct scraper when the user picks one.
    """
    cats = await _all_categories()
    if not cats:
        await say("⚠️ Topic browsing is not available right now (scraper modules missing). Send *2* to search by name, or *exit*.")
        return
    total_pages = (len(cats) + _TOPIC_PAGE_SIZE - 1) // _TOPIC_PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * _TOPIC_PAGE_SIZE
    chunk = cats[start_i:start_i + _TOPIC_PAGE_SIZE]
    lines = [_TOPIC_PROMPT]
    for i, (name, _, source) in enumerate(chunk, 1):
        src_tag = "📊 FBC" if source == "fbc" else "📗 FCB"
        lines.append(f"*{start_i + i}.* {name}  ({src_tag})")
    nav = []
    if page > 0:
        nav.append("Send *prev* for previous page")
    if page < total_pages - 1:
        nav.append("*next* for more")
    nav.append("*back* for menu")
    nav.append("*exit* to cancel")
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _all_categories() -> list[tuple[str, str, str]]:
    """Merge categories from both discovery sources.

    Returns a list of (category_name, category_path, source) tuples where
    source is either 'fbc' (freebookcentre.net) or 'fcb' (freecomputerbooks.com).
    Categories are sorted alphabetically, with duplicates between sources kept
    (the user can pick either one).
    """
    out: list[tuple[str, str, str]] = []
    # Both scrapers are inlined classes in this file, so they're always available.
    # We use try/except Exception (not ImportError) to handle network errors gracefully.
    try:
        for name, path in await _FBC.get_categories():
            out.append((name, path, "fbc"))
    except Exception as e:
        log.warning("_FBC.get_categories failed: %s", e)
    try:
        for name, path in await _FCB.get_categories():
            out.append((name, path, "fcb"))
    except Exception as e:
        log.warning("_FCB.get_categories failed: %s", e)
    # Sort alphabetically by name
    out.sort(key=lambda x: x[0].lower())
    return out


async def _show_subtopics(phone: str, say, cat_name: str, cat_path: str, source: str = "fbc", page: int = 0) -> None:
    """Send a paginated list of subtopics for a category from a given source.

    `source` is either 'fbc' (freebookcentre.net) or 'fcb' (freecomputerbooks.com).
    """
    # Both scrapers are inlined classes in this file, so they're always available.
    # Pick the right one based on the source tag.
    scraper = _FBC if source == "fbc" else _FCB
    await say(f"🔎 Loading subtopics for *{cat_name}*...")
    try:
        subs = await scraper.get_subtopics(cat_path)
    except Exception as e:
        log.exception("get_subtopics failed: %s", e)
        await say(f"⚠️ Couldn't load subtopics for *{cat_name}*. Try another category, or *back* for the menu.")
        return
    if not subs:
        await say(f"😕 No subtopics found for *{cat_name}*. Try another category, or *back*.")
        return
    total_pages = (len(subs) + _TOPIC_PAGE_SIZE - 1) // _TOPIC_PAGE_SIZE
    page = max(0, min(page, total_pages - 1))
    start_i = page * _TOPIC_PAGE_SIZE
    chunk = subs[start_i:start_i + _TOPIC_PAGE_SIZE]
    src_tag = "FBC" if source == "fbc" else "FCB"
    lines = [_SUBTOPIC_PROMPT.format(cat=f"{cat_name}  ({src_tag})")]
    for i, (name, _) in enumerate(chunk, 1):
        lines.append(f"*{start_i + i}.* {name}")
    nav = []
    if page > 0:
        nav.append("*prev* for previous")
    if page < total_pages - 1:
        nav.append("*next* for more")
    nav.append("*back* for categories")
    nav.append("*exit* to cancel")
    lines.append("\n" + " • ".join(nav))
    await say("\n".join(lines))


async def _deliver_topic_books(phone: str, say, cat_name: str, sub_name: str, sub_url: str, source: str = "fbc") -> None:
    """Fetch top 5 books from the chosen discovery source for a subtopic, run legal search()
    for each, and deliver the first PDF found per book. Books not found legally get a
    'not found' notice with the discovery site's detail-page link as a fallback.
    """
    # Both scrapers are inlined classes in this file, so they're always available.
    # Pick the right one based on the source tag.
    if source == "fbc":
        scraper = _FBC
        site_name = "freebookcentre.net"
    else:  # 'fcb'
        scraper = _FCB
        site_name = "freecomputerbooks.com"
    await say(f"🔎 Fetching top 5 *{sub_name}* books and searching legal libraries...\n"
              f"(This takes ~30s — I check 9 free legal sources for each book)")
    try:
        books = await scraper.get_top_books(sub_url, n=5)
    except Exception as e:
        log.exception("get_top_books failed: %s", e)
        await say("⚠️ Couldn't load books for that subtopic. Try another, or *back*.")
        return
    if not books:
        await say(f"😕 No books found under *{sub_name}*. Try another subtopic, or *back*.")
        return
    delivered = 0
    for i, b in enumerate(books, 1):
        title = b["title"]
        author = b.get("author", "")
        # Build a search query: title + author surname (if it's a personal name, not an institution)
        query = title
        if author and author.lower() not in ("na", "unknown", ""):
            first_author = re.split(r"[,;/]", author)[0].strip()
            if not re.search(r"(department|university|institute|college|school|professor|dept)", first_author, re.I):
                # Take just the first word of the author's name for a cleaner query
                query = f"{title} {first_author.split()[0]}"
        await say(f"\n📖 *Book {i}/{len(books)}:* {title}" + (f" — {author}" if author and author.lower() != "na" else ""))
        try:
            found, _ = await search(query)
        except Exception as e:
            log.warning("search for topic book failed: %s", e)
            found = []
        if found and found[0]["score"] >= CUTOFF:
            top = found[0]
            await say(f"✅ Found via *{top['source']}*. Sending...")
            try:
                await deliver(phone, top, say)
                delivered += 1
            except Exception as e:
                log.exception("deliver failed: %s", e)
                await say(f"⚠️ Couldn't send the file. Direct link: {top.get('url') or b['detail_url']}")
        else:
            # No legal source had this book — send the discovery site's detail page as a link
            await say(f"📎 Not in any free legal library. Browse/download here:\n{b['detail_url']}")
    await say(f"\n✅ Done! Delivered *{delivered}/{len(books)}* books as PDFs. "
              f"Send */books* for another, or */menu* for the menu.")


async def _recommend_books(query: str, n: int = 5) -> list[dict]:
    """Search BOTH discovery sites (freebookcentre.net + freecomputerbooks.com) for books
    that fuzzy-match the query, return top N as recommendations.

    Used when the legal `search()` returns nothing — we still want to give the user
    a few 'maybe you meant these' suggestions, each with a link to the detail page.
    """
    import difflib
    q_norm = _norm(query)
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return []
    candidates: list[dict] = []

    # Source 1: freebookcentre.net — iterate all categories' first subtopic (cheap heuristic)
    # In production we'd want a search endpoint, but freebookcentre doesn't have one.
    # Instead, we fetch a few popular category subtopic pages and fuzzy-match titles.
    # We cap at ~5 subtopic fetches so this stays under ~10 seconds.
    # (Scrapers are inlined classes, so no ImportError possible — only network errors.)
    try:
        # Pick a small set of categories likely to match the query (programming/CS first)
        priority_cats = [
            ("Java / Advanced Java", "/JavaTech/javaCategory.html"),
            ("Computer Science",      "/CompuScience/compscCategory.html"),
            ("Programming Languages", "/Language/langCategory.html"),
            ("Mathematics",           "/SpecialCat/Free-Mathematics-Books-Download.html"),
        ]
        all_cats = await _FBC.get_categories()
        # Filter priority_cats to only those that exist
        priority_cats = [c for c in priority_cats if c in all_cats]
        for cat_name, cat_path in priority_cats[:3]:
            try:
                subs = await _FBC.get_subtopics(cat_path)
                # Take the first 2 subtopics of each priority category
                for sub_name, sub_url in subs[:2]:
                    try:
                        books = await _FBC.get_top_books(sub_url, n=10)
                        for b in books:
                            # Score the title against the query
                            t_norm = _norm(b["title"])
                            if not t_norm:
                                continue
                            t_tokens = set(_tokens(b["title"]))
                            # Require at least one query token to appear in the title
                            if not (q_tokens & t_tokens):
                                continue
                            s = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()
                            if s >= 0.20:
                                b = dict(b)
                                b["score"] = s
                                b["source"] = "FBC"
                                candidates.append(b)
                    except Exception as e:
                        log.warning("fbc sub fetch %s failed: %s", sub_url, e)
                        continue
            except Exception as e:
                log.warning("fbc cat %s failed: %s", cat_path, e)
                continue
    except Exception as e:
        log.warning("_FBC recommendations failed: %s", e)

    # Source 2: freecomputerbooks.com — same approach
    try:
        priority_cats_fcb = [
            ("Java and Jakarta EE", "/javaCategory.html"),
            ("Computer Languages", "/langCategory.html"),
            ("Computer Science",    "/compscCategory.html"),
            ("Mathematics",        "/mathCategory.html"),
        ]
        all_cats_fcb = await _FCB.get_categories()
        priority_cats_fcb = [c for c in priority_cats_fcb if c in all_cats_fcb]
        for cat_name, cat_path in priority_cats_fcb[:3]:
            try:
                subs = await _FCB.get_subtopics(cat_path)
                for sub_name, sub_url in subs[:2]:
                    try:
                        books = await _FCB.get_top_books(sub_url, n=10)
                        for b in books:
                            t_norm = _norm(b["title"])
                            if not t_norm:
                                continue
                            t_tokens = set(_tokens(b["title"]))
                            if not (q_tokens & t_tokens):
                                continue
                            s = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()
                            if s >= 0.20:
                                b = dict(b)
                                b["score"] = s
                                b["source"] = "FCB"
                                candidates.append(b)
                    except Exception as e:
                        log.warning("fcb sub fetch %s failed: %s", sub_url, e)
                        continue
            except Exception as e:
                log.warning("fcb cat %s failed: %s", cat_path, e)
                continue
    except Exception as e:
        log.warning("_FCB recommendations failed: %s", e)

    # Sort by score, dedupe by title, return top N
    candidates.sort(key=lambda b: b["score"], reverse=True)
    seen_titles = set()
    out = []
    for b in candidates:
        key = _norm(b["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        out.append(b)
        if len(out) >= n:
            break
    return out


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
    cands = state.get("cands") or []
    try:
        # ---- TOPIC MODE: main menu (1 = browse by topic, 2 = search by name) ----
        if step == "menu":
            if text == "1":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "cat", "page": 0})
                await _show_categories(phone, say, page=0)
                return False
            if text == "2":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
                await say("📚 Send me the *book name* (add the author if you know it), or *back* for the menu, *exit* to cancel.")
                return False
            await say("🤔 Reply with *1* (browse by topic) or *2* (search by name), or *exit*.")
            return False

        # ---- TOPIC MODE: category picker (paginated) ----
        if step == "cat":
            page = state.get("page", 0)
            if low == "next":
                page += 1
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "cat", "page": page})
                await _show_categories(phone, say, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "cat", "page": page})
                await _show_categories(phone, say, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
                await say(PROMPT)
                return False
            if text.isdigit():
                cats = await _all_categories()
                n = int(text)
                if not 1 <= n <= len(cats):
                    await say(f"🤔 Please reply with a number from 1 to {len(cats)} (or *next* / *back* / *exit*).")
                    return False
                cat_name, cat_path, cat_source = cats[n - 1]
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "sub",
                                                       "cat_name": cat_name, "cat_path": cat_path,
                                                       "cat_source": cat_source, "page": 0})
                await _show_subtopics(phone, say, cat_name, cat_path, source=cat_source, page=0)
                return False
            await say("🤔 Reply with a category *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # ---- TOPIC MODE: subtopic picker (paginated) ----
        if step == "sub":
            cat_name = state.get("cat_name", "")
            cat_path = state.get("cat_path", "")
            cat_source = state.get("cat_source", "fbc")
            page = state.get("page", 0)
            if low == "next":
                page += 1
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "sub",
                                                       "cat_name": cat_name, "cat_path": cat_path,
                                                       "cat_source": cat_source, "page": page})
                await _show_subtopics(phone, say, cat_name, cat_path, source=cat_source, page=page)
                return False
            if low == "prev":
                page = max(0, page - 1)
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "sub",
                                                       "cat_name": cat_name, "cat_path": cat_path,
                                                       "cat_source": cat_source, "page": page})
                await _show_subtopics(phone, say, cat_name, cat_path, source=cat_source, page=page)
                return False
            if low == "back":
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "cat", "page": 0})
                await _show_categories(phone, say, page=0)
                return False
            if text.isdigit():
                scraper_mod = _FBC if cat_source == "fbc" else _FCB
                subs = await scraper_mod.get_subtopics(cat_path)
                n = int(text)
                if not 1 <= n <= len(subs):
                    await say(f"🤔 Please reply with a number from 1 to {len(subs)} (or *next* / *back* / *exit*).")
                    return False
                sub_name, sub_url = subs[n - 1]
                if not _quota_ok(phone):
                    await asyncio.to_thread(_save, phone, {})
                    await say(f"⏳ You've reached today's limit of {MAX_PER_DAY} books. Please try again tomorrow.")
                    return True
                await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "topic_books",
                                                       "cat_name": cat_name, "sub_name": sub_name, "sub_url": sub_url,
                                                       "cat_source": cat_source})
                await _deliver_topic_books(phone, say, cat_name, sub_name, sub_url, source=cat_source)
                await asyncio.to_thread(_save, phone, {})
                _log(phone, title=f"{cat_name} / {sub_name}", source="topic_browse")
                return True
            await say("🤔 Reply with a subtopic *number* (or *next* / *prev* / *back* / *exit*).")
            return False

        # ---- NAME SEARCH MODE: pick from previous results ----
        if step == "pick" and text.isdigit():
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

        # ---- NAME SEARCH MODE: free text ----
        if low == "back":
            await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "menu"})
            await say(PROMPT)
            return False
        if len(text) < 2:
            await say("🤔 Please send the book's name (or *back* for the menu, *exit* to cancel).")
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
        # Try to recommend 5 similar books from the discovery sites (freebookcentre + freecomputerbooks)
        await say("🔎 Looking for similar books you might like...")
        try:
            recs = await _recommend_books(text, n=5)
        except Exception as e:
            log.warning("_recommend_books failed: %s", e)
            recs = []
        if recs:
            rec_lines = ["\n\n💡 *Top 5 similar books you might like:*"]
            for i, r in enumerate(recs, 1):
                title = r["title"]
                author = r.get("author", "")
                src_tag = r.get("source", "")
                detail = r["detail_url"]
                line = f"\n*{i}.* {title}"
                if author and author.lower() not in ("na", "unknown", ""):
                    line += f" — {author}"
                if src_tag:
                    line += f"  ({src_tag})"
                line += f"\n    {detail}"
                rec_lines.append(line)
            extra += "\n".join(rec_lines)
            extra += "\n\n_These were not in the free legal libraries I check, but you can browse/download them at the source site._"
        await asyncio.to_thread(_save, phone, {"mode": MODE, "step": "ask"})
        await say("😕 I couldn't find a *free, legal* copy of that. It may still be under copyright (then it's only available to buy or borrow).\n"
                  "Try the exact title (and author), or send another book name, or *back* for the menu, or *exit*." + extra)
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
