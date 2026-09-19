import io
import docx
import pypdf
from google import genai
from google.genai import types

client = genai.Client()

def extract_text(filename: str, content: bytes) -> str:
    filename = filename.lower()
    text = ""

    # 1. Native PDF text extraction using pypdf
    if filename.endswith(".pdf"):
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            extracted_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())
            text = "\n\n".join(extracted_pages).strip()
            
            # If native text extraction succeeded, return it directly without API calls
            if len(text) > 30:
                return text[:24000]
        except Exception:
            pass

    # 2. DOCX extraction
    elif filename.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if text.strip():
                return text.strip()[:24000]
        except Exception:
            pass

    # 3. TXT extraction
    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore").strip()[:24000]

    # 4. Fallback OCR using gemini-2.5-flash for scanned PDFs or image uploads
    if not text.strip():
        try:
            mime_type = "application/pdf" if filename.endswith(".pdf") else "image/png"
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_bytes(data=content, mime_type=mime_type),
                    "Extract and return all readable text from this document verbatim. Do not summarize or format into markdown."
                ]
            )
            text = response.text or ""
        except Exception as e:
            print(f"Gemini OCR Error: {e}")

    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Could not extract readable text from document.")

    return clean_text[:24000]


def make_docx(text: str) -> bytes:
    doc = docx.Document()
    for paragraph in text.split("\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()