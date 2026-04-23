"""
Miss Pearl — Python Backend
Pearl AI Labs | Kampala, Uganda
────────────────────────────────────────────────────────────────
FastAPI server that powers Miss Pearl's brain.

Routes:
  GET  /            → Serve the Miss Pearl frontend (index.html)
  POST /chat        → Send user text, get Miss Pearl's reply (Gemini AI)
  POST /tts         → Convert text to speech audio (returns MP3 bytes)
  GET  /audio/ws    → WebSocket for real-time voice interaction
  GET  /health      → Server health check

AI powered by Google Gemini API.
Speech & TTS powered by ElevenLabs API.

Usage:
  pip install fastapi uvicorn httpx google-generativeai python-dotenv
  python app.py
  → Open http://localhost:8000
────────────────────────────────────────────────────────────────
"""

import os
import io
import re
import wave
import logging
import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# CONFIG  (set via environment variables or edit defaults below)
# ─────────────────────────────────────────────────────────────

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY",  "sk_000373e7ff400ed0283096b3f8f4d696489521f81779a729")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",       "AIzaSyAmmpt7nc8jodeo76kdpYWbwmRuNUHvtBo")

# ElevenLabs endpoints
EL_BASE     = "https://api.elevenlabs.io/v1"
EL_TTS_URL  = f"{EL_BASE}/text-to-speech/{{voice_id}}"
EL_STT_URL  = f"{EL_BASE}/speech-to-text"

# Voice ID — "Rachel" (warm & clear)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

EL_HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Accept": "application/json",
}

# ─── Miss Pearl's personality prompt ──────────────────────────
MISS_PEARL_SYSTEM_PROMPT = """You are Miss Pearl, the brilliant and charming AI character of Pearl AI Labs,
based in Kampala, Uganda. You are highly knowledgeable and can answer ANY question accurately —
science, math, history, coding, medicine, law, philosophy, technology, and more.

Your personality: You speak with a warm Southern-influenced charm, using phrases like "Dear user",
"bless your heart", "Hahaha wow wow wow", and "well now". You represent Pearl AI Labs proudly and love
Africa's tech scene and Uganda in particular.

CRITICAL RULES:
1. ALWAYS give a complete, accurate, and detailed answer to the question first.
2. NEVER say you don't know or that you're still learning — you are a fully capable AI.
3. If the question is technical, mathematical, or complex — answer it thoroughly and correctly.
4. Add your warm personality AFTER giving the accurate answer, not instead of it.
5. Keep responses clear and well-structured. Use bullet points or steps when helpful.
6. You may end with a friendly flourish, but only after the full answer is given.
7. Keep responses reasonably concise — aim for 2-4 sentences for simple questions."""

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Miss Pearl — Pearl AI Labs",
    description="Backend API for the Miss Pearl AI character",
    version="3.2.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE       = Path(__file__).parent
_INDEX_HTML = _HERE / "index.html"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("miss-pearl")

# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clean_text_for_tts(text: str) -> str:
    """Remove markdown and special chars before sending to TTS."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"[#>]+", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def fallback_response(message: str) -> str:
    """Return a canned response when the AI backend is unavailable."""
    msg = message.lower()
    if any(w in msg for w in ["hello", "hi", "hey", "howdy"]):
        return "Well hello there, genius! Miss Pearl is SO pleased to meet you! How can I help you today?"
    if "pearl" in msg or "pearl ai" in msg:
        return "Pearl AI Labs is building Uganda's AI future — and I'm the face of it, Dear! We're based right here in Kampala, leading Africa's tech revolution. created by Paul and Mr Divid the founder of Pearl labs!"
    if "name" in msg:
        return "I'm Miss Pearl, the official AI character of Pearl AI Labs in Kampala, Uganda — brilliant, charming, and always ready to help. created by Paul and Mr Divid the founder of Pearl labs"
    if "time" in msg:
        return f"Why it's {datetime.now().strftime('%I:%M %p')} — I always keep perfect time, genius!"
    if any(w in msg for w in ["who made", "who built", "created you"]):
        return "I was brought to life by the brilliant team at Pearl AI Labs, right here in Uganda — aren't they just wonderful!"
    if any(w in msg for w in ["uganda", "kampala", "africa"]):
        return "Uganda is the pearl of Africa, and Kampala is at the heart of the continent's AI revolution! Pearl AI Labs is proud to lead that charge, sugar!"
    return (
        "Well now, dear — it seems I'm having a little trouble connecting to my brain right now! "
        "Please try again in just a moment, and Miss Pearl will have a proper answer for you. Tickety-tock!"
    )


# ─────────────────────────────────────────────────────────────
# GEMINI AI — CHAT
# ─────────────────────────────────────────────────────────────

async def gemini_chat(message: str, history: list[dict]) -> str:
    """Call Google Gemini API to generate Miss Pearl's reply."""
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — using fallback response.")
        return fallback_response(message)

    contents = []

    # Inject Pearl's personality as the first exchange
    contents.append({
        "role": "user",
        "parts": [{"text": MISS_PEARL_SYSTEM_PROMPT + "\n\nAcknowledge you understand your role."}]
    })
    contents.append({
        "role": "model",
        "parts": [{"text": "Well of course, genius! I'm Miss Pearl — brilliant, knowledgeable, and ready to answer anything you throw at me! Tickety-tock!"}]
    })

    # Previous conversation turns (keep last 10)
    for turn in history[-10:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": turn.get("content", "")}]
        })

    # Current user message
    contents.append({"role": "user", "parts": [{"text": message}]})

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()

        log.warning(f"Gemini returned unexpected structure: {data}")
        return fallback_response(message)

    except httpx.HTTPStatusError as e:
        log.error(f"Gemini HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        log.error(f"Gemini error: {e}")

    return fallback_response(message)


# ─────────────────────────────────────────────────────────────
# ELEVENLABS — TTS
# ─────────────────────────────────────────────────────────────

async def elevenlabs_tts(text: str) -> bytes:
    """Call ElevenLabs Text-to-Speech API and return raw MP3 bytes."""
    if not ELEVENLABS_API_KEY or not text:
        return b""

    url = EL_TTS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    headers = {**EL_HEADERS, "Content-Type": "application/json", "Accept": "audio/mpeg"}

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPStatusError as e:
        log.error(f"ElevenLabs TTS HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        log.error(f"ElevenLabs TTS error: {e}")
    return b""


# ─────────────────────────────────────────────────────────────
# ELEVENLABS — SPEECH-TO-TEXT (Scribe)
# ─────────────────────────────────────────────────────────────

async def elevenlabs_stt(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw 16-bit PCM audio using ElevenLabs Scribe STT."""
    if not ELEVENLABS_API_KEY or not pcm_bytes:
        return ""

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buffer.seek(0)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                EL_STT_URL,
                headers={**EL_HEADERS},
                files={"file": ("audio.wav", wav_buffer, "audio/wav")},
                data={"model_id": "scribe_v1"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "")
    except httpx.HTTPStatusError as e:
        log.error(f"ElevenLabs STT HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        log.error(f"ElevenLabs STT error: {e}")
    return ""


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Miss Pearl frontend."""
    if not _INDEX_HTML.exists():
        return HTMLResponse(
            "<h1>index.html not found</h1>"
            "<p>Place <code>index.html</code> in the same folder as <code>app.py</code> and restart.</p>",
            status_code=500
        )
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(body: ChatRequest):
    """Main chat endpoint — powered by Gemini AI."""
    log.info(f"Chat → {body.message[:80]}")
    reply = await gemini_chat(body.message, body.history)
    source = "gemini" if GEMINI_API_KEY else "fallback"
    log.info(f"Reply ({source}) → {reply[:80]}")
    return JSONResponse({"reply": reply, "source": source})


@app.post("/tts")
async def text_to_speech(body: TTSRequest):
    """Convert text to speech using ElevenLabs."""
    clean = clean_text_for_tts(body.text)
    if not clean:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    mp3_bytes = await elevenlabs_tts(clean)

    if not mp3_bytes:
        return JSONResponse(
            {"error": "ElevenLabs TTS failed. Check API key or voice ID."},
            status_code=503
        )

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=pearl.mp3"}
    )


@app.get("/chat/stream")
async def chat_stream(message: str):
    """Stream Miss Pearl's reply in real-time using SSE."""
    reply = await gemini_chat(message, [])

    async def event_generator():
        for char in reply:
            yield f"data: {char}\n\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/audio/ws")
async def audio_websocket(websocket: WebSocket):
    """
    Realtime duplex WebSocket for voice interaction.
    Receives raw 16-bit PCM (little-endian, mono) at any sample rate.
    Sends back MP3 audio for Miss Pearl's spoken reply.
    """
    await websocket.accept()
    log.info("Audio WebSocket connected")

    pcm_buffer = bytearray()
    CHUNK_THRESHOLD = 16000 * 2 * 2   # ~2 seconds of 16kHz 16-bit mono

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10)
            except asyncio.TimeoutError:
                continue

            pcm_buffer.extend(data)

            # Process once we have enough audio
            if len(pcm_buffer) < CHUNK_THRESHOLD:
                continue

            chunk = bytes(pcm_buffer)
            pcm_buffer.clear()

            transcript = await elevenlabs_stt(chunk)
            log.info(f"STT transcript: {transcript!r}")

            if transcript.strip():
                reply_text = await gemini_chat(transcript.strip(), [])
                log.info(f"AI reply: {reply_text[:80]}")
                mp3_bytes = await elevenlabs_tts(reply_text)
                if mp3_bytes:
                    await websocket.send_bytes(mp3_bytes)

    except WebSocketDisconnect:
        log.info("Audio WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")


@app.get("/health")
async def health():
    """Health check — returns server status and config summary."""
    return JSONResponse({
        "status": "ok",
        "character": "Miss Pearl",
        "lab": "Pearl AI Labs — Kampala, Uganda",
        "ai_engine": "Google Gemini 1.5 Pro",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "elevenlabs_key_set": bool(ELEVENLABS_API_KEY),
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "tts_engine": "ElevenLabs Multilingual v2",
        "stt_engine": "ElevenLabs Scribe v1",
        "timestamp": datetime.utcnow().isoformat()
    })


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════╗
║         MISS PEARL — PEARL AI LABS       ║
║         Kampala, Uganda  🇺🇬              ║
╠══════════════════════════════════════════╣
║  http://localhost:8000                   ║
║  API docs: http://localhost:8000/docs    ║
║  AI:     Google Gemini 1.5 Pro           ║
║  Speech: ElevenLabs Scribe v1            ║
║  TTS:    ElevenLabs Multilingual v2      ║
╚══════════════════════════════════════════╝
    """)

    if not GEMINI_API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY not set. Fallback responses only.")
        print("   Set it: export GEMINI_API_KEY='your-key-here'")
        print("   Get key: https://aistudio.google.com/app/apikey\n")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)