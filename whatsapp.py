import asyncio
import logging
import os
import re
import time
import httpx

logger = logging.getLogger("career_bot")


class WhatsApp:

    # Meta throttles bursts to one user ("pair rate limit", error 131056 / 130429). Messages sent too fast can be
    # accepted by the API (HTTP 200) and still never reach the phone, so every outgoing message is paced.
    MIN_GAP = float(os.getenv("WA_MIN_GAP_SECONDS", "1.2"))        # seconds between two messages to the same user
    RATE_CODES = {130429, 131056, 131048, 80007, 4, 17, 613}

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._last = {}
        self._locks = {}

    async def _pace(self, phone):
        key = re.sub(r"[^\d]", "", str(phone))
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            wait = self._last.get(key, 0.0) + self.MIN_GAP - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[key] = time.monotonic()

    async def _send(self, payload: dict):
        """POST /messages, paced per recipient; retries once (after 6 s) when Meta says we are sending too fast."""
        await self._pace(payload.get("to", ""))
        for attempt in (1, 2):
            response = await self.client.post(f"{self.base_url}/messages", headers=self.headers, json=payload)
            if response.status_code == 200:
                return response
            try:
                code = int((response.json().get("error") or {}).get("code") or 0)
            except Exception:
                code = 0
            if attempt == 1 and (response.status_code == 429 or code in self.RATE_CODES):
                logger.warning("WhatsApp rate limit (code %s) for %s; retrying in 6 s", code, payload.get("to"))
                await asyncio.sleep(6)
                continue
            return response
        return response

    @property
    def phone_number_id(self) -> str:
        """Dynamically fetch and sanitize phone number ID to avoid empty or invalid base URLs."""
        phone_id = (
            os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
            or os.getenv("PHONE_NUMBER_ID", "")
        ).strip("/ ")
        
        # WARNING: Ensure this ID matches your LIVE Phone Number ID in Meta Developer Console!
        if not phone_id:
            logger.warning("PHONE_NUMBER_ID env variable missing! Ensure it is set in Render.")
        return phone_id

    @property
    def base_url(self) -> str:
        """Dynamically construct base URL for Meta Graph API v21.0."""
        return f"https://graph.facebook.com/v21.0/{self.phone_number_id}"

    @property
    def access_token(self) -> str:
        """Dynamically fetch token to ensure latest environment variables are used."""
        token = (
            os.getenv("WHATSAPP_TOKEN", "")
            or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        )
        return token.strip()

    @property
    def headers(self) -> dict:
        """Generate fresh authorization headers for each outgoing request."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def text(self, phone: str, body: str):
        if hasattr(body, "content"):
            clean_body = body.content
        else:
            clean_body = body

        if isinstance(clean_body, list):
            text_parts = []
            for block in clean_body:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            clean_body = "\n".join(text_parts)

        clean_body = str(clean_body)

        if len(clean_body) > 4000:
            clean_body = (
                clean_body[:3990] + "\n\n*(Message truncated due to length)*"
            )

        # SANITIZE PHONE NUMBER: Remove '+', spaces, and dashes
        clean_phone = re.sub(r"[^\d]", "", str(phone))

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_phone,
            "type": "text",
            "text": {"preview_url": False, "body": clean_body},
        }

        logger.info(f"Sending WhatsApp message to {clean_phone}")

        response = await self._send(payload)

        # PRINT/LOG THE EXACT OUTGOING META RESPONSE
        if response.status_code != 200:
            logger.error(
                f"Meta API Error ({response.status_code}): {response.text}"
            )
        else:
            logger.info(f"Meta API Success: {response.json()}")

        return response

    async def menu(self, to: str):
        menu_text = (
            "🤖 *CareerMate Command Menu*\n\n"
            "⚡ /news - Get today's top tech news\n"
            "🏆 /competitions - Latest competitions\n"
            "💼 /internships - Latest internships\n"
            "📝 /practice - helps to Practice quiz \n"
            "🎯 /ats <role> - Evaluate ATS score (e.g. /ats Devops)\n"
            "📄 /chat_pdf - Chat & ask questions with your PDF\n"
            "📚 /books - Discover top tech books\n"
            "📋 /pyq - Practice previous year questions\n"
            "📄 /send - Upload or review your saved resume\n"
            "🔍 /job_search - Search for open job roles\n"
            "📺 /utube_notes - Get detailed notes from a YouTube link\n"
            "🗺️ /roadmap <topic> - Get an official/AI roadmap\n"
            "💬 /assistance - Ask any question\n"
            "🚫 /cancel - Exit active quiz or operation\n"
            "🗑️ /delete - Clear saved session & uploaded resume\n"
            "💬 /feedback - Share your feedback \n\n"
            "Reply with any command to get started!"
        )
        await self.text(to, menu_text)

    async def download(self, media_id: str) -> bytes:
        res = await self.client.get(
            f"https://graph.facebook.com/v21.0/{media_id}",
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        media_url = res.json().get("url")
        doc_res = await self.client.get(
            media_url, headers={"Authorization": f"Bearer {self.access_token}"}
        )
        return doc_res.content

    async def upload_media(
        self, content: bytes, mime_type: str, filename: str
    ) -> str:
        url = f"{self.base_url}/media"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        files = {"file": (filename, content, mime_type)}
        data = {"messaging_product": "whatsapp"}
        res = await self.client.post(
            url, headers=headers, data=data, files=files
        )
        if res.status_code != 200:
            logger.error(f"Upload media error ({res.status_code}): {res.text}")
            return ""
        return res.json().get("id", "")

    async def document(
        self,
        to: str,
        content: bytes,
        filename: str,
        mime_type: str = "application/pdf",
    ):
        media_id = await self.upload_media(content, mime_type, filename)
        if not media_id:
            logger.error("Failed to upload document to WhatsApp Meta API")
            return None

        clean_phone = re.sub(r"[^\d]", "", str(to))

        payload = {
            "messaging_product": "whatsapp",
            "to": clean_phone,
            "type": "document",
            "document": {"id": media_id, "filename": filename},
        }
        response = await self._send(payload)
        return response

    async def send_roadmap_pdf(self, to: str, topic_name: str) -> bool:
        """Downloads a roadmap PDF from roadmap.sh using clean slug logic and sends it to user."""
        slug = topic_name.strip().lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)

        pdf_url = f"https://roadmap.sh/pdfs/roadmaps/{slug}.pdf"
        output_filename = f"{slug}.pdf"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        try:
            res = await self.client.get(
                pdf_url, headers=headers, follow_redirects=True
            )
            if res.status_code != 200:
                logger.error(
                    f"Failed to fetch roadmap PDF for '{slug}'. Status code: {res.status_code}"
                )
                await self.text(
                    to,
                    f"❌ Could not find a PDF roadmap for *{topic_name}*.\n\n"
                    f"Try popular topics like: `ai-engineer`, `devsecops`, `data-analyst`, `backend`, or `frontend`.",
                )
                return False

            pdf_bytes = res.content

        except Exception as e:
            logger.error(f"Exception while downloading roadmap PDF: {e}")
            await self.text(
                to, "❌ Failed to download roadmap PDF. Please try again."
            )
            return False

        try:
            await self.document(
                to=to,
                content=pdf_bytes,
                filename=output_filename,
                mime_type="application/pdf",
            )
            logger.info(f"Successfully sent {output_filename} to {to}")
            return True
        except Exception as e:
            logger.error(f"Failed to send PDF to WhatsApp: {e}")
            return False

    async def send_image_by_id(
        self, to: str, media_id: str, caption: str = ""
    ):
        clean_phone = re.sub(r"[^\d]", "", str(to))
        payload = {
            "messaging_product": "whatsapp",
            "to": clean_phone,
            "type": "image",
            "image": {"id": media_id, "caption": caption},
        }
        await self._send(payload)

    async def close(self):
        await self.client.aclose()
