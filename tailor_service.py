import logging
from google import genai
from google.genai import types

logger = logging.getLogger("career_bot")


class ResumeTailorService:

    def __init__(self):
        # Initializes using google-genai SDK (reads GEMINI_API_KEY from environment)
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash"

    async def tailor_resume(
        self, resume_text: str, job_description: str
    ) -> str:
        """Analyzes a job description and current resume, returning a tailored version

        optimized for ATS alignment and missing key skills.
        """
        prompt = f"""
You are an expert ATS Resume Strategist and Senior Hiring Manager.

Your task is to tailor the user's existing resume specifically for the given Target Job Description / Role.

### Target Job Description / Role:
{job_description}

### User's Current Resume:
{resume_text}

---

### Instructions for Output:
1. **Targeted Executive Summary**: Write a compelling 3-4 sentence summary customized for this role.
2. **Key Missing Keywords & Skills**: List crucial hard/soft skills from the job description that should be emphasized.
3. **Optimized Bullet Points**: Rewrite 4-6 key experience/project bullet points from the original resume, incorporating strong action verbs and high-impact metrics tailored to the job description.
4. **Tailored Resume Draft**: Provide a clean, structured, complete Markdown draft of the tailored resume ready for copy-pasting.

Provide clear formatting using bold titles, bullet points, and clean Markdown structure.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Failed to tailor resume via Gemini: {e}")
            return (
                "❌ *Error tailoring resume.*\n\n"
                "An issue occurred while tailoring your resume. Please try again shortly."
            )