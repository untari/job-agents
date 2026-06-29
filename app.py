import io
import json
import os
from pathlib import Path

import google.generativeai as genai
import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

SYSTEM_PROMPT = """You are an expert career coach and professional resume writer with deep knowledge of how applicant tracking systems (ATS) work and what hiring managers look for.

Your job is to help software engineering candidates tailor their application materials to a specific job description. Be specific, honest, and actionable. Focus on what will actually get someone past the resume screen."""


def get_model():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_PROMPT)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ValueError("Could not extract text from PDF. Try a non-scanned PDF.")
    return text


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/tailor")
async def tailor(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set. Export it in your terminal before running.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    pdf_bytes = await resume.read()
    try:
        resume_text = extract_pdf_text(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user_message = f"""Please analyze my resume against this job description and give me three things:

## MATCH SCORE
Give me a score from 0-100 and 2-3 sentences explaining exactly where I fit and where I fall short. Be honest.

## TAILORED RESUME BULLETS
Give me 6-8 bullet points I can use or adapt for my resume. These should highlight my existing experience but reframe it to speak directly to what this role needs. Use strong action verbs. Quantify results where you can infer them. Start each bullet with a dash (–).

## COVER LETTER
Write a compelling 3-paragraph cover letter:
- Paragraph 1: A strong opening hook that shows I understand what they're building and why I'm excited
- Paragraph 2: Connect 2-3 specific things from my background directly to their needs
- Paragraph 3: Confident close, mention I'd love to discuss further

---

MY RESUME:
{resume_text}

---

JOB DESCRIPTION:
{job_description}"""

    async def stream_response():
        try:
            model = get_model()
            response = await model.generate_content_async(
                user_message,
                stream=True,
                generation_config=genai.GenerationConfig(max_output_tokens=4000),
            )
            async for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            yield "data: [DONE]\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Something went wrong: {e}'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
