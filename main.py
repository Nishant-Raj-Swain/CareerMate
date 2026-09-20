import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables relative to main.py location
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure explicit logging format to stdout for Render visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("career_bot")

# Environment Variable Resolution
VERIFY_TOKEN = (os.getenv("WEBHOOK_VERIFY_TOKEN") or "nishi7890").strip()
WHATSAPP_TOKEN = (os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN") or "").strip()
PHONE_NUMBER_ID = (os.getenv("WHATSAPP_PHONE_NUMBER_ID") or os.getenv("PHONE_NUMBER_ID") or "").strip()
APP_SECRET = os.getenv("META_APP_SECRET", "").strip()

import database as db
import pdf_service
import image_service
from ai_service import AIService
from agent import build_agent
from whatsapp import WhatsApp
import quiz_service
from news_service import NewsService  # Integrated NewsService
from opportunities_service import opportunities_service

wa = None
ai_service = None
news_service = None
locks = {}
langgraph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wa, ai_service, news_service, langgraph_app
    
    db.initialize()
    wa = WhatsApp()
    ai_service = AIService()
    news_service = NewsService(wa_access_token=WHATSAPP_TOKEN, wa_phone_id=PHONE_NUMBER_ID)

    if not WHATSAPP_TOKEN:
        logger.error("❌ WHATSAPP_TOKEN is missing or empty in environment! Outgoing messages will fail.")
    else:
        logger.info(f"✅ WHATSAPP_TOKEN loaded successfully! (Length: {len(WHATSAPP_TOKEN)})")
        
    if not PHONE_NUMBER_ID:
        logger.error("❌ PHONE_NUMBER_ID is missing or empty in environment! API requests will fail.")
    else:
        logger.info(f"✅ PHONE_NUMBER_ID loaded successfully: {PHONE_NUMBER_ID}")

    checkpointer = MemorySaver()
    langgraph_app = build_agent(checkpointer)

    yield

    await wa.close()


app = FastAPI(title="WhatsApp Career Assistant", lifespan=lifespan)


async def send_next_question(phone: str, state: dict):
    """Formats and sends the current quiz question."""
    idx = state.get("current_index", 0)
    questions = state.get("questions", [])
    
    if idx >= len(questions):
        score = state.get("score", 0)
        total = len(questions)
        topic = state.get("topic", "Quiz").title()
        
        await wa.text(
            phone, 
            f"🎉 *{topic} Quiz Completed!*\n\n"
            f"📊 *Final Score:* {score} / {total}\n"
            f"Percentage: {(score/total)*100:.1f}%\n\n"
            f"Type `/prepare <topic>` to try another topic!"
        )
        db.save_state(phone, {})
        await wa.menu(phone)
        return

    q = questions[idx]
    opts_text = ""
    opt_labels = ["A", "B", "C", "D"]
    for i, opt in enumerate(q.get("options", [])):
        label = opt_labels[i] if i < len(opt_labels) else str(i+1)
        opts_text += f"\n*{label}.* {opt}"

    msg = (
        f"📝 *{state.get('topic', 'Quiz').title()} Test* (Q{idx + 1}/{len(questions)})\n\n"
        f"{q.get('question', '')}\n"
        f"{opts_text}\n\n"
        f"_Reply with A, B, C, or D (or send *exit* to quit early)_"
    )
    await wa.text(phone, msg)


async def handle_roadmap_generation(phone: str, topic: str):
    """Generates an official PDF roadmap or falls back to an AI visual roadmap card."""
    await wa.text(phone, f"🔍 Searching roadmap.sh for official *{topic.title()}* PDF...")
    
    pdf_sent = await wa.send_roadmap_pdf(to=phone, topic_name=topic)

    if pdf_sent:
        db.save_state(phone, {})
        await wa.menu(phone)
        return

    await wa.text(phone, "Official PDF not found on roadmap.sh. Generating custom AI visual roadmap...")
    roadmap_data = await ai_service.generate_structured_roadmap(topic)
    image_bytes = image_service.generate_roadmap_card(topic, roadmap_data.get("sections", {}))
    
    media_id = await wa.upload_media(image_bytes, "image/jpeg", "roadmap.jpg")
    await wa.send_image_by_id(phone, media_id, caption=f"🗺️ {topic.title()} Roadmap")
    db.save_state(phone, {})
    
    if roadmap_data.get("summary"):
        await wa.text(phone, roadmap_data.get("summary", ""))
        
    await wa.menu(phone)


async def handle_text(phone: str, text: str):
    user = db.get_user(phone)
    state = user.get("state", {})
    mode = state.get("mode")

    # --- QUIZ INTERACTION MODE ---
    if mode == "quiz":
        ans = text.strip().lower()
        
        if ans in ("exit", "quit", "stop", "cancel", "/cancel", "/exit"):
            score = state.get("score", 0)
            attempted = state.get("current_index", 0)
            topic = state.get("topic", "Quiz").title()
            
            await wa.text(
                phone,
                f"🛑 *Test Ended Early*\n\n"
                f"Topic: *{topic}*\n"
                f"Score: *{score} / {attempted}* questions attempted.\n\n"
                f"Session cleared."
            )
            db.save_state(phone, {})
            await wa.menu(phone)
            return

        ans_upper = text.strip().upper()
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        
        if ans_upper not in mapping:
            await wa.text(phone, "⚠️ Please reply with option **A**, **B**, **C**, **D**, or send **exit** to quit the test.")
            return

        user_choice = mapping[ans_upper]
        idx = state.get("current_index", 0)
        questions = state.get("questions", [])
        current_q = questions[idx] if idx < len(questions) else None

        if current_q:
            correct_idx = current_q.get("correct", 0)
            if user_choice == correct_idx:
                state["score"] = state.get("score", 0) + 1
                await wa.text(phone, "✅ *Correct!*")
            else:
                opt_labels = ["A", "B", "C", "D"]
                correct_label = opt_labels[correct_idx] if correct_idx < len(opt_labels) else "N/A"
                await wa.text(phone, f"❌ *Incorrect.* Correct Answer: *{correct_label}*")

        state["current_index"] = idx + 1
        db.save_state(phone, state)
        await send_next_question(phone, state)
        return

    # --- ATS SCORE MODE ---
    if mode == "ats_score":
        db.save_resume(phone, text)
        target_role = state.get("target_role", "Data Analyst / Software Role")
        await wa.text(phone, f"🎯 Evaluating resume for target role: *{target_role}*...")
        
        score_report = await ai_service.score_resume_ats(resume_text=text, target_role=target_role)
        db.save_state(phone, {})
        await wa.text(phone, score_report)
        await wa.menu(phone)
        return

    if mode == "resume":
        db.save_resume(phone, text)
        await wa.text(phone, "Resume saved. Generating review...")
        review = await ai_service.analyze_resume(text)
        db.save_state(phone, {})
        await wa.text(phone, review)
        await wa.menu(phone)
        return

    if mode == "job_search":
        await wa.text(phone, f"🔍 Searching for recent *{text}* job postings...")
        job_results = await ai_service.search_jobs(text)
        db.save_state(phone, {})
        await wa.text(phone, job_results)
        await wa.menu(phone)
        return

    if mode == "roadmap":
        await handle_roadmap_generation(phone, text)
        return

    if mode == "assistance":
        config = {"configurable": {"thread_id": phone}}
        input_msg = f"User Resume:\n{user['resume']}\n\nQuestion:\n{text}" if user.get("resume") else text
        
        output = await asyncio.to_thread(
            langgraph_app.invoke, 
            {"messages": [HumanMessage(content=input_msg)]}, 
            config
        )
        
        if isinstance(output, dict) and "messages" in output and len(output["messages"]) > 0:
            last_msg = output["messages"][-1]
            text_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        elif hasattr(output, 'content'):
            text_content = output.content
        else:
            text_content = str(output)
            
        if len(text_content) > 4000:
            text_content = text_content[:3990] + "\n\n*(Message truncated due to length)*"
            
        await wa.text(phone, text_content)
        return

    await wa.text(phone, "Please select a command from the menu.")
    await wa.menu(phone)


async def command(phone: str, text: str):
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    
    if cmd in ("/start", "/help", "/menu", "/cancel"):
        db.save_state(phone, {})
        await wa.menu(phone)
        return
        
    if cmd == "/delete":
        db.delete_user(phone)
        await wa.text(phone, "Local user session deleted.")
        await wa.menu(phone)
        return

    if cmd == "/news":
        db.save_state(phone, {})
        await wa.text(phone, "⚡ *Fetching today's top tech news cards...*")
        await news_service.process_and_send_news(recipient_phone=phone)
        await wa.menu(phone)
        return

    if cmd in ("/competitions", "/competition"):
        db.save_state(phone, {})
        await wa.text(phone, "🔍 *Fetching active competitions from Unstop...*")
        result = await opportunities_service.fetch_opportunities("competitions", limit=5)
        await wa.text(phone, result)
        await wa.menu(phone)
        return

    if cmd in ("/internships", "/internship"):
        db.save_state(phone, {})
        await wa.text(phone, "🔍 *Fetching active internships from Unstop...*")
        result = await opportunities_service.fetch_opportunities("internships", limit=5)
        await wa.text(phone, result)
        await wa.menu(phone)
        return

    if cmd == "/ats_score":
        target_role = parts[1].strip() if len(parts) > 1 else "Data Analyst / Software Role"
        user = db.get_user(phone)
        existing_resume = user.get("resume")

        if existing_resume:
            await wa.text(phone, f"🎯 Scoring existing resume against: *{target_role}*...")
            score_report = await ai_service.score_resume_ats(resume_text=existing_resume, target_role=target_role)
            db.save_state(phone, {})
            await wa.text(phone, score_report)
            await wa.menu(phone)
        else:
            db.save_state(phone, {"mode": "ats_score", "target_role": target_role})
            await wa.text(
                phone,
                f"🎯 *ATS Resume Evaluation*\n\n"
                f"Target Role: *{target_role}*\n\n"
                f"Please upload your resume (PDF/DOCX) or paste your full resume text below."
            )
        return

    if cmd == "/prepare":
        topic = parts[1].strip().lower() if len(parts) > 1 else "python"
        await wa.text(phone, f"⏳ Generating 20 practice questions for *{topic.title()}* from IndiaBIX...")
        
        questions = await quiz_service.fetch_indiabix_questions(topic=topic, total_needed=20)
        
        if not questions:
            await wa.text(
                phone, 
                "❌ Couldn't load questions for this topic right now.\n"
                "Try topics like `/prepare aptitude`, `/prepare reasoning`, `/prepare python`, `/prepare java`, or `/prepare sql`."
            )
            return

        quiz_state = {
            "mode": "quiz",
            "topic": topic,
            "questions": questions,
            "current_index": 0,
            "score": 0
        }
        db.save_state(phone, quiz_state)
        await send_next_question(phone, quiz_state)
        return

    if cmd == "/job_search":
        if len(parts) > 1:
            domain_query = parts[1].strip()
            db.save_state(phone, {})
            await wa.text(phone, f"🔍 Searching for recent *{domain_query}* job postings...")
            job_results = await ai_service.search_jobs(domain_query)
            await wa.text(phone, job_results)
            await wa.menu(phone)
        else:
            db.save_state(phone, {"mode": "job_search"})
            await wa.text(phone, "🔍 *Job Search Mode*\n\nPlease reply with the role or domain you are searching for (e.g., Data Analyst, Python Developer).")
        return

    if cmd == "/roadmap":
        if len(parts) > 1:
            topic = parts[1].strip()
            db.save_state(phone, {})
            await handle_roadmap_generation(phone, topic)
        else:
            db.save_state(phone, {"mode": "roadmap"})
            await wa.text(phone, "🗺️ *Roadmap Mode*\n\nEnter learning topic (e.g., DevSecOps, AI Engineer, Data Science).")
        return

    if cmd in ("/send", "/assistance"):
        mode = cmd.lstrip("/")
        if mode == "send":
            mode = "resume"

        db.save_state(phone, {"mode": mode})
        
        prompts = {
            "resume": "Upload PDF/DOCX resume or paste text.",
            "assistance": "Ask any career question."
        }
        await wa.text(phone, prompts.get(mode, "Send details."))


async def process_message(message: dict):
    phone = message.get("from", "")
    msg_id = message.get("id", "")
    
    logger.info(f"Processing incoming message [{msg_id}] from user: {phone}")
    
    # Check duplicate processing via DB claim check (Allowlist filter removed for Live production)
    if not db.claim_message(msg_id):
        logger.info(f"Message ID [{msg_id}] already processed. Skipping duplicate.")
        return

    lock = locks.setdefault(phone, asyncio.Lock())
    async with lock:
        msg_type = message.get("type")
        if msg_type == "document":
            user = db.get_user(phone)
            state = user.get("state", {})
            mode = state.get("mode")
            doc = message["document"]
            content = await wa.download(doc["id"])
            
            try:
                extracted = pdf_service.extract_text(doc.get("filename", "file.pdf"), content)
                if not extracted or len(extracted.strip()) == 0:
                    raise ValueError("No extractable plain text found in file.")

            except Exception as e:
                logger.error(f"Failed to extract document text for {phone}: {e}")
                await wa.text(
                    phone,
                    "❌ *Could not read text from this document.*\n\n"
                    "Please ensure the file is a text-based PDF/DOCX file (not an image scan) or paste the text directly."
                )
                return

            if mode == "ats_score":
                db.save_resume(phone, extracted)
                target_role = state.get("target_role", "Data Analyst / Software Role")
                await wa.text(phone, f"🎯 Document uploaded. Evaluating ATS score for *{target_role}*...")
                
                score_report = await ai_service.score_resume_ats(resume_text=extracted, target_role=target_role)
                db.save_state(phone, {})
                await wa.text(phone, score_report)
                await wa.menu(phone)
            else:
                db.save_resume(phone, extracted)
                await wa.text(phone, "✅ Resume saved successfully. Generating review...")
                
                review = await ai_service.analyze_resume(extracted)
                db.save_state(phone, {})
                await wa.text(phone, review)
                await wa.menu(phone)
            return

        # Text handling with default fallback
        text = (message.get("text", {}) or {}).get("body", "").strip()
        if text.startswith("/"):
            await command(phone, text)
        elif text:
            await handle_text(phone, text)


@app.get("/webhook")
async def verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode") or params.get("hub_mode")
    token = params.get("hub.verify_token") or params.get("hub_verify_token")
    challenge = params.get("hub.challenge") or params.get("hub_challenge")

    logger.info(f"Verify Request - Mode: {mode} | Token: {token}")

    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        logger.info("Webhook verification succeeded.")
        return Response(content=str(challenge), media_type="text/plain", status_code=200)

    logger.warning("Webhook verification failed.")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook(request: Request, bg_tasks: BackgroundTasks):
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > 1024 * 1024:
            raise HTTPException(status_code=413)

    sig = request.headers.get("x-hub-signature-256", "")
    
    if APP_SECRET:
        expected = "sha256=" + hmac.new(APP_SECRET.encode(), bytes(buffer), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            logger.warning("Invalid webhook signature from Meta.")
            raise HTTPException(status_code=403)

    payload = json.loads(bytes(buffer))
    
    # Process valid WhatsApp entry changes
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                bg_tasks.add_task(process_message, msg)

    return {"status": "accepted"}
