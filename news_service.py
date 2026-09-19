import logging
import os
import re
import asyncio
import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger("career_bot")


class NewsService:

    def __init__(self, wa_access_token: str = None, wa_phone_id: str = None):
        self.client = genai.Client()
        # Fallback to gemini-3.6-flash if GEMINI_MODEL is not set
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
        
        # Dynamically fetch tokens/IDs if missing or empty strings
        self.wa_access_token = (
            wa_access_token
            or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
            or os.getenv("WHATSAPP_TOKEN", "")
        ).strip()
        
        self.wa_phone_id = (
            wa_phone_id
            or os.getenv("PHONE_NUMBER_ID", "")
            or os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
            or "1323339190853977"  # Active fallback ID
        ).strip("/ ")

        self.proxy_url = "https://api.rss2json.com/v1/api.json?rss_url=https://indianexpress.com/section/technology/feed/"

    def extract_image_url(self, item: dict) -> str:
        """Extracts image URL from thumbnail, enclosure, or embedded <img> tags."""
        if item.get("thumbnail"):
            return item["thumbnail"]

        if isinstance(item.get("enclosure"), dict) and item["enclosure"].get("link"):
            return item["enclosure"]["link"]

        raw_html = item.get("description", "") + " " + item.get("content", "")
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', raw_html, re.IGNORECASE)
        if img_match:
            return img_match.group(1)

        return ""

    def clean_html(self, text: str) -> str:
        """Strips raw HTML tags for clean text snippets."""
        clean = re.sub(r"<[^>]+>", "", text)
        return clean.replace("&nbsp;", " ").strip()

    async def fetch_news_articles(self, limit: int = 3) -> list[dict]:
        """Fetches and parses top tech articles with extracted images."""
        articles = []
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            try:
                res = await http_client.get(self.proxy_url)
                if res.status_code == 200 and res.json().get("status") == "ok":
                    for item in res.json().get("items", [])[:limit]:
                        articles.append(
                            {
                                "title": item.get("title", ""),
                                "description": self.clean_html(item.get("description", ""))[:250],
                                "link": item.get("link", ""),
                                "image_url": self.extract_image_url(item),
                            }
                        )
            except Exception as e:
                logger.error(f"Failed to fetch news feed: {e}")
        return articles

    async def generate_caption_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generates content with automatic retry logic for 503/429 errors and model fallback."""
        models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-1.5-flash"]
        
        # Deduplicate models while keeping order
        models_to_try = list(dict.fromkeys(models_to_try))

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            # Disable function calling for news formatting to save latency/tokens
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                    )
                    return response.text.strip()
                except APIError as e:
                    if e.code in (503, 429):
                        logger.warning(f"Model {model} hit {e.code} error (Attempt {attempt+1}/{max_retries}). Retrying...")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"Gemini API Error with model {model}: {e}")
                        break
                except Exception as e:
                    logger.error(f"Unexpected error with model {model}: {e}")
                    break
                    
        raise RuntimeError("All Gemini models and retries failed.")

    async def send_whatsapp_media_card(self, recipient_phone: str, image_url: str, caption: str):
        """Dispatches an Image message with a formatted text caption via WhatsApp API."""
        phone_id = (
            self.wa_phone_id
            or os.getenv("PHONE_NUMBER_ID", "")
            or "1323339190853977"
        ).strip("/ ")

        if not phone_id:
            logger.error("PHONE_NUMBER_ID is missing! Cannot send WhatsApp message.")
            return

        url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
        access_token = (
            self.wa_access_token
            or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
            or os.getenv("WHATSAPP_TOKEN", "")
        ).strip()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        if image_url:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient_phone,
                "type": "image",
                "image": {"link": image_url, "caption": caption},
            }
        else:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient_phone,
                "type": "text",
                "text": {"body": caption},
            }

        async with httpx.AsyncClient() as client:
            try:
                res = await client.post(url, headers=headers, json=payload, timeout=10.0)
                if res.status_code != 200:
                    logger.error(f"WhatsApp API error ({res.status_code}): {res.text}")
                else:
                    logger.info(f"Successfully sent news card to {recipient_phone}")
            except Exception as e:
                logger.error(f"WhatsApp API post failed: {e}")

    async def process_and_send_news(self, recipient_phone: str):
        """Orchestrates fetching, Gemini editing, and sending individual media cards."""
        articles = await self.fetch_news_articles(limit=3)
        if not articles:
            await self.send_whatsapp_media_card(
                recipient_phone,
                image_url="",
                caption=(
                    "📰 *Tech News Digest*\n\n"
                    "Unable to retrieve news at the moment. Please try again later!"
                ),
            )
            return

        for item in articles:
            prompt = f"""
You are an expert Tech News Editor formatting a WhatsApp news caption.

Title: {item['title']}
Snippet: {item['description']}
Link: {item['link']}

Instructions:
1. Start with a bold catchy headline with a relevant emoji.
2. Provide a 2-sentence crisp summary.
3. End with a line: 🔗 *Read full story:* {item['link']}
4. Do NOT output image links or raw markdown code fences.
"""
            try:
                caption = await self.generate_caption_with_retry(prompt)
            except Exception as e:
                logger.error(f"Gemini fallback formatting triggered due to: {e}")
                caption = f"📰 *{item['title']}*\n\n{item['description']}\n\n🔗 {item['link']}"

            # Send as an individual visual card
            await self.send_whatsapp_media_card(
                recipient_phone=recipient_phone,
                image_url=item["image_url"],
                caption=caption,
            )