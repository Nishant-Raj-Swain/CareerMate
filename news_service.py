import asyncio
import logging
import os
import re
import html
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger("career_bot")

# Tried in order until one returns headlines. Only headlines + links are used (no AI).
FEEDS = [
    "https://indianexpress.com/section/technology/feed/",
    "https://www.gadgets360.com/rss/news",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
]
RSS2JSON = "https://api.rss2json.com/v1/api.json?rss_url="
UA = {"User-Agent": "Mozilla/5.0 (compatible; CareerMateBot/1.0)", "Accept": "application/rss+xml, application/xml, text/xml, */*"}
_NS = {"media": "http://search.yahoo.com/mrss/", "content": "http://purl.org/rss/1.0/modules/content/"}


def _clean(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _image(item: ET.Element, raw_html: str) -> str:
    for tag in ("media:content", "media:thumbnail"):
        el = item.find(tag, _NS)
        if el is not None and el.get("url"):
            return el.get("url")
    enc = item.find("enclosure")
    if enc is not None and enc.get("url") and "image" in (enc.get("type") or "image"):
        return enc.get("url")
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', raw_html or "", re.IGNORECASE)
    return m.group(1) if m else ""


def parse_rss(xml_text: str, limit: int) -> list[dict]:
    """RSS 2.0 / Atom text -> [{title, link, image_url}] (headlines only)."""
    out = []
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    atom = False
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry"); atom = True
    for it in items:
        if atom:
            ns = "{http://www.w3.org/2005/Atom}"
            title = _clean(it.findtext(ns + "title") or "")
            ln = it.find(ns + "link")
            link = (ln.get("href") if ln is not None else "") or ""
            raw = it.findtext(ns + "content") or it.findtext(ns + "summary") or ""
        else:
            title = _clean(it.findtext("title") or "")
            link = (it.findtext("link") or "").strip()
            raw = (it.findtext("description") or "") + " " + (it.findtext("content:encoded", namespaces=_NS) or "")
        if title and link:
            out.append({"title": title, "link": link, "image_url": _image(it, raw)})
        if len(out) >= limit:
            break
    return out


class NewsService:
    """/news: sends the latest tech headlines (title + link, with the article image when the feed has one). No AI."""

    def __init__(self, wa_access_token: str = None, wa_phone_id: str = None):
        self.wa_access_token = (wa_access_token or os.getenv("WHATSAPP_ACCESS_TOKEN", "") or os.getenv("WHATSAPP_TOKEN", "")).strip()
        self.wa_phone_id = (wa_phone_id or os.getenv("PHONE_NUMBER_ID", "") or os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
                            or "1323339190853977").strip("/ ")

    async def fetch_news_articles(self, limit: int = 5) -> list[dict]:
        """Reads the RSS feeds directly (several fall-backs, then rss2json). Never raises."""
        async with httpx.AsyncClient(timeout=15.0, headers=UA, follow_redirects=True) as http:
            for feed in FEEDS:
                try:
                    r = await http.get(feed)
                    if r.status_code == 200:
                        arts = parse_rss(r.text, limit)
                        if arts:
                            return arts
                    logger.warning("news feed %s -> HTTP %s", feed, r.status_code)
                except Exception as e:
                    logger.warning("news feed %s failed: %s", feed, e)
            try:                                                    # last resort: the old JSON proxy
                r = await http.get(RSS2JSON + FEEDS[0])
                j = r.json() if r.status_code == 200 else {}
                if j.get("status") == "ok":
                    arts = [{"title": _clean(i.get("title", "")), "link": i.get("link", ""),
                             "image_url": i.get("thumbnail") or (i.get("enclosure") or {}).get("link", "")}
                            for i in j.get("items", []) if i.get("title") and i.get("link")]
                    return arts[:limit]
            except Exception as e:
                logger.error("rss2json fallback failed: %s", e)
        return []

    async def send_whatsapp_media_card(self, recipient_phone: str, image_url: str, caption: str):
        """Sends an image with a caption (or plain text when there is no image) via the WhatsApp Cloud API."""
        phone_id = (self.wa_phone_id or os.getenv("PHONE_NUMBER_ID", "")).strip("/ ")
        if not phone_id:
            logger.error("PHONE_NUMBER_ID is missing! Cannot send WhatsApp message.")
            return
        url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
        token = (self.wa_access_token or os.getenv("WHATSAPP_ACCESS_TOKEN", "") or os.getenv("WHATSAPP_TOKEN", "")).strip()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        base = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": recipient_phone}
        payload = ({**base, "type": "image", "image": {"link": image_url, "caption": caption[:1000]}} if image_url
                   else {**base, "type": "text", "text": {"body": caption, "preview_url": True}})
        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(url, headers=headers, json=payload, timeout=10.0)
                if res.status_code != 200 and image_url:            # image link unreachable for WhatsApp -> resend as text
                    logger.warning("image card rejected (%s); sending text instead", res.status_code)
                    payload = {**base, "type": "text", "text": {"body": caption, "preview_url": True}}
                    res = await client.post(url, headers=headers, json=payload, timeout=10.0)
                if res.status_code != 200:
                    logger.error(f"WhatsApp API error ({res.status_code}): {res.text}")
                else:
                    logger.info(f"Successfully sent news card to {recipient_phone}")
            except Exception as e:
                logger.error(f"WhatsApp API post failed: {e}")

    async def process_and_send_news(self, recipient_phone: str):
        """Fetch the latest headlines and send each as one card: headline + link."""
        articles = await self.fetch_news_articles(limit=5)
        if not articles:
            await self.send_whatsapp_media_card(recipient_phone, "", "📰 *Tech News Digest*\n\nUnable to retrieve news at the moment. Please try again later!")
            return
        for item in articles:
            caption = f"📰 *{item['title']}*\n\n🔗 *Read full story:* {item['link']}"
            await self.send_whatsapp_media_card(recipient_phone, item["image_url"], caption)
            await asyncio.sleep(1.5)                                 # image cards download slower than text; spacing keeps the order
        await asyncio.sleep(3.0)                                     # let the last image card land before main.py sends the menu
