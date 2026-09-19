import json
import logging
import os
import re
from typing import Any, Dict, List, Union

import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger("career_bot")


class AIService:

    def __init__(self):
        # Initialize Gemini model (Updated default to gemini-3.6-flash)
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
        google_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not google_api_key:
            logger.warning("⚠️ GEMINI_API_KEY is missing in environment variables.")

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.3,
        )
        self.adzuna_app_id = os.getenv("ADZUNA_APP_ID", "").strip()
        self.adzuna_app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
        logger.info(f"✅ AIService initialized using model: {model_name}")

    def _parse_text(self, content: Union[str, List[Any]]) -> str:
        """Safely extracts raw text content from Gemini responses regardless of whether
        the model returns a string, list of blocks, or structured objects.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            return "".join(parts)

        return str(content)

    def _clean_json_str(self, raw_text: str) -> str:
        """Removes Markdown code fences (e.g., ```json ... ```) to extract raw JSON."""
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    async def score_resume_ats(
        self, resume_text: str, target_role: str = None
    ) -> str:
        """Evaluates a resume against Applicant Tracking System criteria using Gemini."""

        role_instruction = (
            f'Evaluate specifically for the target position: "{target_role}".'
            if target_role
            else "First, identify the candidate's target role dynamically from their resume header, summary, and projects (e.g., Full Stack Developer, Software Engineer, Data Analyst)."
        )

        prompt = f"""
You are an advanced Applicant Tracking System (ATS) parser and resume evaluator.
{role_instruction}

Evaluate the resume across these 4 categories (25 points each):
1. **Keywords & Technical Skills** (25 pts) — Evaluate tools and frameworks relevant strictly to THEIR target domain. DO NOT penalize roles for missing irrelevant tools (e.g., do not dock points on a Full Stack Developer resume for missing Data Analyst tools like Tableau or Pandas).
2. **Impact & Action Verbs** (25 pts) — Check for quantifiable metrics, business outcomes, and active phrasing.
3. **Structure & Completeness** (25 pts) — Check for standard resume sections, clear contact info, work history, and projects.
4. **Readability & Formatting** (25 pts) — Check for text extraction issues, spacing defects, date consistency, and bullet point structure.

Generate output matching EXACTLY the format below. Do NOT add extra sections or conversational intro text.

TARGET FORMAT:
🎯 *ATS RESUME SCORE CARD*
━━━━━━━━━━━━━━━━━━━
📊 *Overall ATS Score: [Total]/100*

*Breakdown:*
🛠️ *Keywords & Technical Skills:* [X]/25   
📈 *Impact & Action Verbs:* [X]/25   
🏗️ *Structure & Completeness:* [X]/25   
📄 *Readability & Formatting:* [X]/25   

💡 *Top 5 Key Issues to Rectify:*

1. **[Issue Title 1]:**
   [Specific, actionable fix.]

2. **[Issue Title 2]:**
   [Specific, actionable fix.]

3. **[Issue Title 3]:**
   [Specific, actionable fix.]

4. **[Issue Title 4]:**
   [Specific, actionable fix.]

5. **[Issue Title 5]:**
   [Specific, actionable fix.]

Resume Text:
{resume_text}
"""
        try:
            res = await self.model.ainvoke(prompt)
            return self._parse_text(res.content)
        except Exception as e:
            logger.error(f"Error performing ATS resume scoring: {e}")
            return "❌ Error generating ATS score. Please try again."

    async def generate_structured_roadmap(self, topic: str) -> Dict[str, Any]:
        """Generates structured roadmap sections and summary JSON for visual image rendering."""
        prompt = f"""
You are an expert career counselor and educator. Create a concise learning roadmap for: "{topic}".

Return ONLY a valid raw JSON object without any additional conversational text or markdown explanation.
The JSON must strictly follow this structure:
{{
  "sections": {{
    "1. Fundamentals": ["Topic A", "Topic B", "Topic C"],
    "2. Intermediate": ["Topic D", "Topic E", "Topic F"],
    "3. Advanced": ["Topic G", "Topic H", "Topic I"]
  }},
  "summary": "A brief 2-3 sentence overview explaining how to master {topic}."
}}
"""
        try:
            res = await self.model.ainvoke(prompt)
            raw_text = self._parse_text(res.content)
            clean_json = self._clean_json_str(raw_text)

            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON for roadmap '{topic}': {e}\nRaw output was:\n{res.content}"
            )
            return {
                "sections": {
                    "1. Core Basics": [f"Learn {topic} Fundamentals", "Core Concepts"],
                    "2. Practical Projects": [
                        "Build Real-world Apps",
                        "Best Practices",
                    ],
                    "3. Advanced Skills": ["Mastering Tools", "Optimization"],
                },
                "summary": f"A step-by-step pathway to learn {topic} from core basics to advanced mastery.",
            }
        except Exception as e:
            logger.error(
                f"Error generating roadmap for '{topic}': {str(e)}"
            )
            raise e

    async def analyze_resume(self, resume_text: str, target_role: str = None) -> str:
        """Analyzes resume text and provides a concise scorecard and key fixes."""
        return await self.score_resume_ats(
            resume_text=resume_text, target_role=target_role
        )

    async def fetch_adzuna_jobs(
        self, query: str, country_code: str = "in", results_per_page: int = 5
    ) -> list:
        """Fetch live jobs directly from Adzuna API."""
        if not self.adzuna_app_id or not self.adzuna_app_key:
            logger.warning("Adzuna API credentials missing in .env")
            return []

        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
        params = {
            "app_id": self.adzuna_app_id,
            "app_key": self.adzuna_app_key,
            "results_per_page": results_per_page,
            "what": query,
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("results", [])
                else:
                    logger.error(
                        f"Adzuna API Error: {response.status_code} - {response.text}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Exception during Adzuna API request: {e}")
            return []

    async def search_jobs(self, domain: str) -> str:
        """Fetch live structured job listings using Adzuna API with fallback."""
        raw_jobs = await self.fetch_adzuna_jobs(
            query=domain, country_code="in", results_per_page=5
        )

        if not raw_jobs:
            return (
                f"⚠️ Could not retrieve live jobs for *{domain}* right now.\n\n"
                f"🔗 Search directly on LinkedIn:\n"
                f"https://www.linkedin.com/jobs/search/?keywords={domain.replace(' ', '%20')}"
            )

        output = [f"🔍 *Live Job Openings for {domain.title()}:*\n"]

        for idx, job in enumerate(raw_jobs, start=1):
            title = job.get("title", "Job Title N/A").strip()
            company = job.get("company", {}).get("display_name", "N/A")
            location_name = job.get("location", {}).get("display_name", "India")
            redirect_url = job.get("redirect_url", "")

            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            salary_str = ""
            if salary_min and salary_max:
                salary_str = f"\n💰 **Salary:** ₹{int(salary_min):,} - ₹{int(salary_max):,}"

            output.append(
                f"📌 *{idx}. {title}*\n"
                f"🏢 **Company:** {company}\n"
                f"📍 **Location:** {location_name}"
                f"{salary_str}\n"
                f"🔗 [Apply Here]({redirect_url})\n"
            )

        return "\n".join(output)

    async def answer_document_question(
        self, doc_text: str, question: str
    ) -> str:
        """Answers questions based on uploaded document context."""
        prompt = f"""
Based on the following document context, answer the user's question clearly and accurately.

Document Context:
{doc_text[:4000]}

Question:
{question}
"""
        res = await self.model.ainvoke(prompt)
        return self._parse_text(res.content)