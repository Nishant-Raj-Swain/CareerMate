import logging
import os
import re
import tempfile
import httpx

logger = logging.getLogger(__name__)

ROADMAP_SLUG_ALIASES = {
    # Data Analyst aliases
    "data analytics": "data-analyst",
    "data analyst": "data-analyst",
    "analytics": "data-analyst",
    # Data Scientist aliases
    "data science": "ai-data-scientist",
    "data scientist": "ai-data-scientist",
    "machine learning": "ai-data-scientist",
    "ml": "ai-data-scientist",
    # AI Engineer aliases
    "ai": "ai-engineer",
    "ai engineer": "ai-engineer",
    "artificial intelligence": "ai-engineer",
    # Web Dev aliases
    "web development": "full-stack",
    "fullstack": "full-stack",
    "full stack": "full-stack",
}


class WhatsAppMediaService:

    def __init__(self, access_token: str, phone_number_id: str):
        if not access_token or not access_token.strip():
            raise ValueError(
                "WHATSAPP_ACCESS_TOKEN is missing or empty! Check your .env file."
            )

        self.access_token = access_token.strip()
        self.phone_number_id = phone_number_id
        self.base_url = (
            f"https://graph.facebook.com/v21.0/{self.phone_number_id}"
        )
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def sanitize_slug(self, topic: str) -> str:
        """Normalizes user input and maps alias phrases to official roadmap.sh slugs."""
        raw_slug = topic.strip().lower()

        # Check alias dictionary first
        if raw_slug in ROADMAP_SLUG_ALIASES:
            return ROADMAP_SLUG_ALIASES[raw_slug]

        # Standard slugification fallback
        clean_text = re.sub(r"[^a-z0-9\s-]", "", raw_slug)
        return re.sub(r"[\s_]+", "-", clean_text)

    async def fetch_pdf(self, slug: str, save_path: str) -> bool:
        """Downloads the static PDF from roadmap.sh to local temporary storage."""
        pdf_url = f"https://roadmap.sh/pdfs/roadmaps/{slug}.pdf"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        }

        async with httpx.AsyncClient(follow_redirects=True) as client:
            res = await client.get(pdf_url, headers=headers)
            if res.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                logger.warning(
                    f"PDF missing on roadmap.sh for '{slug}' (HTTP {res.status_code})"
                )
                return False

    async def upload_to_tmpfiles(self, file_path: str) -> str | None:
        """Uploads local file to tmpfiles.org and returns direct download link."""
        url = "https://tmpfiles.org/api/v1/upload"

        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": f}
                res = await client.post(url, files=files)

            if res.status_code == 200:
                data = res.json()
                page_url = data.get("data", {}).get("url")
                if page_url:
                    # Convert page URL to direct download URL (requires /dl/)
                    return page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

            logger.error(f"tmpfiles.org upload failed: {res.text}")
            return None

    async def send_pdf_document(
        self, recipient_phone: str, topic: str
    ) -> bool:
        """Orchestrates PDF fetch, tmpfiles upload, and WhatsApp document dispatch."""
        slug = self.sanitize_slug(topic)
        temp_filename = f"{slug}-roadmap.pdf"

        # Safe cross-platform temporary file path (Windows & Linux compatible)
        local_path = os.path.join(tempfile.gettempdir(), temp_filename)

        try:
            # 1. Download PDF from roadmap.sh
            if not await self.fetch_pdf(slug, local_path):
                return False

            # 2. Upload PDF to tmpfiles.org to get a public direct download URL
            download_url = await self.upload_to_tmpfiles(local_path)
            if not download_url:
                return False

            # 3. Send Document Message via WhatsApp using the link parameter
            message_url = f"{self.base_url}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient_phone,
                "type": "document",
                "document": {
                    "link": download_url,
                    "filename": f"{slug.replace('-', ' ').title()} Roadmap.pdf",
                    "caption": (
                        f"📄 Here is your official PDF roadmap for"
                        f" *{topic.upper()}*!"
                    ),
                },
            }

            async with httpx.AsyncClient() as client:
                res = await client.post(
                    message_url, headers=self.headers, json=payload
                )
                return res.status_code == 200

        finally:
            # Clean up local temporary file safely
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError as e:
                    logger.warning(f"Failed to delete local temp file: {e}")