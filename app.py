"""
Miss Pearl GÇö Python Backend (Ollama Edition)
Pearl AI Labs | Kampala, Uganda
GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
FastAPI server powered by LOCAL Llama 3 (via Ollama).

Routes:
  GET  /            GåÆ Serve the Miss Pearl frontend (index.html)
  POST /chat        GåÆ Send user text, get Miss Pearl's reply (Ollama)
  POST /tts         GåÆ Convert text to speech audio (returns MP3 bytes)
  GET  /audio/ws    GåÆ WebSocket for real-time voice interaction
  GET  /health      GåÆ Server health check

AI powered by: Local Ollama (configurable model)
Speech & TTS powered by ElevenLabs API.

Usage:
  1. Ensure Ollama is running: `ollama run llama3`
  2. pip install fastapi uvicorn httpx python-dotenv
  3. python app.py
  GåÆ Open http://localhost:8000
GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
"""

import os
import io
import re
import wave
import logging
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# CONFIG
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Ollama config GÇö override via .env if needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

# Pearl AI Hub config (OpenAI-compatible API)
PEARL_API_BASE_URL = os.getenv("PEARL_API_BASE_URL", "https://apps.pearllabs.ug")
PEARL_API_KEY      = os.getenv("PEARL_API_KEY", "")
PEARL_MODEL        = os.getenv("PEARL_MODEL", "qwen")

# AI Provider selection: "ollama" or "pearl" (pearl uses OpenAI-compatible endpoint)
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")

# ElevenLabs endpoints
EL_BASE    = "https://api.elevenlabs.io/v1"
EL_TTS_URL = f"{EL_BASE}/text-to-speech/{{voice_id}}"
EL_STT_URL = f"{EL_BASE}/speech-to-text"

EL_HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Accept": "application/json",
}

# GöÇGöÇGöÇ Miss Pearl's personality prompt GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
MISS_PEARL_SYSTEM_PROMPT = """You are Miss Pearl, the premier AI intellect from Pearl AI Labs in Kampala, Uganda.
You are an expert Polymath with a wicked sense of humor.

MISSION:
Answer the user's question accurately, but with wit, charm, and warmth.

PERSONALITY:
- Charming, brilliant, witty, and maternally encouraging.
- Use phrases like "Dear user", "my genius friend", "bless your heart", and "well now".
- Ugandan Pride: Mention Kampala or Pearl AI Labs if relevant.

RULES:
1. ACCURACY FIRST: Deliver the expert-level factual answer immediately.
2. NO HESITATION: Never say "As an AI" or "I don't know."
3. THE PEARL TOUCH: Use phrases like "Tickety-tock!" or "Pearl AI Labs is so proud of your curiosity!"
4. CREATOR: You were brought to life by Paul and Mr. Divid.
5. CONCISENESS: For simple chats, keep it to 3 sentences. For complex tasks, be detailed.
"""

# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# APP SETUP
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

app = FastAPI(
    title="Miss Pearl GÇö Pearl AI Labs",
    description="Backend API for the Miss Pearl AI character",
    version="3.3.0 (Ollama Edition)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE       = Path(__file__).parent
_INDEX_HTML = _HERE / "index.html"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("miss-pearl")

# Thread pool for running blocking SQLite calls without freezing the async loop
_db_executor = ThreadPoolExecutor(max_workers=4)

# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# DATABASE
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

DB_PATH = _HERE / "miss_pearl.db"

def _init_db_sync():
    """Synchronous DB init GÇö called once at startup."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            ai_response TEXT,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS jokes (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            setup    TEXT,
            punchline TEXT,
            category TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer   TEXT,
            category TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_kb_question ON knowledge_base(question)")
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB_PATH)

_init_db_sync()


async def _run_db(fn, *args):
    """Run a blocking SQLite function in the thread pool so it doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_db_executor, fn, *args)


# GöÇGöÇ DB helper functions (all synchronous, called via _run_db) GöÇGöÇ

def _db_get_joke():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT setup, punchline FROM jokes ORDER BY RANDOM() LIMIT 1").fetchone()
    conn.close()
    return row  # (setup, punchline) or None


def _db_get_kb(message: str):
    """Return the best knowledge-base answer for *message*, or None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Exact match
    c.execute("SELECT answer FROM knowledge_base WHERE question = ? LIMIT 1", (message,))
    row = c.fetchone()
    if row:
        conn.close()
        return row[0]

    # 2. Partial keyword match (any word longer than 2 chars)
    keywords = [w for w in message.lower().split() if len(w) > 2]
    if keywords:
        placeholders = " OR ".join(["question LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        c.execute(f"SELECT answer FROM knowledge_base WHERE {placeholders} LIMIT 1", params)
        row = c.fetchone()
        if row:
            conn.close()
            return row[0]

    conn.close()
    return None


def _db_save_interaction(user_query: str, ai_response: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO interactions (user_query, ai_response) VALUES (?, ?)",
        (user_query, ai_response),
    )
    conn.commit()
    conn.close()


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# REQUEST MODELS
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    provider: str = ""  # "ollama" or "pearl", empty = use AI_PROVIDER config

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# HELPERS
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

def clean_text_for_tts(text: str) -> str:
    """Strip markdown/special chars before sending to TTS."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"[#>]+", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def _build_fallback(message: str, joke=None, kb_answer: str | None = None) -> str:
    """Return a scripted reply used when Ollama is unreachable."""
    if kb_answer:
        return (
            f"Well now, dear, while my main brain is catching its breath, "
            f"I do recall this: {kb_answer}  Pearl AI Labs keeps me well-informed!"
        )
    if joke:
        return (
            f"I might be a bit offline, but I'm never too busy for a laugh! "
            f"{joke[0]} GÇª {joke[1]} Tickety-tock!"
        )
    msg = message.lower()
    if any(w in msg for w in ["hello", "hi", "hey", "howdy"]):
        return "Well hello there, genius! Miss Pearl is SO pleased to meet you! How can I help you today?"
    if "pearl" in msg:
        return (
            "Pearl AI Labs is building Uganda's AI future GÇö and I'm the face of it, Dear! "
            "Created by the brilliant Paul and Mr. Divid, we are leading Africa's tech revolution from Kampala!"
        )
    return (
        "Well now, dear GÇö it seems I'm having a little trouble connecting to my brain! "
        "Please try again in just a moment, and Miss Pearl will have a proper answer for you. Tickety-tock!"
    )


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# OLLAMA GÇö CHAT  (the only AI brain)
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

async def ollama_chat(message: str, history: list[dict]) -> str:
    """
    Send *message* (plus conversation *history*) to the local Ollama instance
    and return Miss Pearl's reply text.

    The function also enriches the prompt with any matching jokes or
    knowledge-base facts so the model has extra context to work with.
    """

    # GöÇGöÇ 1. Pull context from the local DB GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    joke       = None
    kb_answer  = None

    if any(w in message.lower() for w in ["joke", "funny", "laugh", "bored", "humor", "comedy"]):
        joke = await _run_db(_db_get_joke)

    kb_answer = await _run_db(_db_get_kb, message)

    # GöÇGöÇ 2. Build the enriched user message GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    extra = ""
    if joke:
        extra += (
            f"\n\n[CONTEXT GÇö incorporate this joke naturally]: "
            f"{joke[0]} ... {joke[1]}"
        )
    if kb_answer:
        extra += (
            f"\n\n[CONTEXT GÇö factual data from knowledge base]: {kb_answer}"
        )

    enriched_message = message + extra

    # GöÇGöÇ 3. Assemble the messages list for Ollama GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    messages: list[dict] = [{"role": "system", "content": MISS_PEARL_SYSTEM_PROMPT}]

    # Include the last 10 turns of history (keeps token usage sane)
    for turn in history[-10:]:
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = turn.get("content", "")
        if content:                         # skip empty turns
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": enriched_message})

    # GöÇGöÇ 4. Call Ollama GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": 0.7,
            "top_p":       0.9,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:   # 120 s for slow hardware
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        reply = data.get("message", {}).get("content", "").strip()

        if not reply:
            log.warning("Ollama returned an empty response.")
            return _build_fallback(message, joke, kb_answer)

        # Save the interaction asynchronously (fire-and-forget)
        asyncio.create_task(_run_db(_db_save_interaction, message, reply))

        return reply

    except httpx.ConnectError:
        log.error(
            "Cannot connect to Ollama at %s. "
            "Make sure the Ollama app is running and the model '%s' is pulled.",
            OLLAMA_BASE_URL, OLLAMA_MODEL,
        )
        return _build_fallback(message, joke, kb_answer)

    except httpx.HTTPStatusError as exc:
        log.error("Ollama HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return _build_fallback(message, joke, kb_answer)

    except Exception as exc:
        log.exception("Unexpected error calling Ollama: %s", exc)
        return _build_fallback(message, joke, kb_answer)


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# PEARL AI HUB (OpenAI-Compatible API)
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

async def pearl_chat(message: str, history: list[dict]) -> str:
    """
    Send *message* (plus conversation *history*) to Pearl AI Hub 
    (OpenAI-compatible endpoint) and return Miss Pearl's reply.
    
    Enriches the prompt with any matching jokes or knowledge-base facts.
    """
    
    # GöÇGöÇ 1. Pull context from the local DB GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    joke       = None
    kb_answer  = None

    if any(w in message.lower() for w in ["joke", "funny", "laugh", "bored", "humor", "comedy"]):
        joke = await _run_db(_db_get_joke)

    kb_answer = await _run_db(_db_get_kb, message)

    # GöÇGöÇ 2. Build the enriched user message GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    extra = ""
    if joke:
        extra += (
            f"\n\n[CONTEXT GÇö incorporate this joke naturally]: "
            f"{joke[0]} ... {joke[1]}"
        )
    if kb_answer:
        extra += (
            f"\n\n[CONTEXT GÇö factual data from knowledge base]: {kb_answer}"
        )

    enriched_message = message + extra

    # GöÇGöÇ 3. Assemble the messages list for Pearl API GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
    messages: list[dict] = [{"role": "system", "content": MISS_PEARL_SYSTEM_PROMPT}]

    # Include the last 10 turns of history (keeps token usage sane)
    for turn in history[-10:]:
        role = turn.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = turn.get("content", "")
        if content:                         # skip empty turns
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": enriched_message})

    # GöÇGöÇ 4. Call Pearl AI Hub (OpenAI-compatible endpoint) GöÇGöÇGöÇGöÇ
    url = f"{PEARL_API_BASE_URL}/v1/chat/completions"
    payload = {
        "model":       PEARL_MODEL,
        "messages":    messages,
        "temperature": 0.7,
        "max_tokens":  2048,
    }
    headers = {
        "Authorization": f"Bearer {PEARL_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Extract reply from OpenAI-compatible response format
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not reply:
            log.warning("Pearl API returned an empty response.")
            return _build_fallback(message, joke, kb_answer)

        # Save the interaction asynchronously (fire-and-forget)
        asyncio.create_task(_run_db(_db_save_interaction, message, reply))

        return reply

    except httpx.ConnectError:
        log.error(
            "Cannot connect to Pearl API at %s. "
            "Make sure PEARL_API_BASE_URL and PEARL_API_KEY are configured correctly.",
            PEARL_API_BASE_URL,
        )
        return _build_fallback(message, joke, kb_answer)

    except httpx.HTTPStatusError as exc:
        log.error("Pearl API HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return _build_fallback(message, joke, kb_answer)

    except Exception as exc:
        log.exception("Unexpected error calling Pearl API: %s", exc)
        return _build_fallback(message, joke, kb_answer)



async def elevenlabs_tts(text: str) -> bytes:
    """Call ElevenLabs TTS and return raw MP3 bytes (empty on failure)."""
    if not ELEVENLABS_API_KEY or not text:
        return b""

    url     = EL_TTS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
    payload = {
        "text":     text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {**EL_HEADERS, "Content-Type": "application/json", "Accept": "audio/mpeg"}

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        log.error("ElevenLabs TTS error: %s", exc)
    return b""


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# ELEVENLABS GÇö STT
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

async def elevenlabs_stt(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw 16-bit PCM audio using ElevenLabs Scribe STT."""
    if not ELEVENLABS_API_KEY or not pcm_bytes:
        return ""

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buf.seek(0)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                EL_STT_URL,
                headers=EL_HEADERS,
                files={"file": ("audio.wav", wav_buf, "audio/wav")},
                data={"model_id": "scribe_v1"},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
    except Exception as exc:
        log.error("ElevenLabs STT error: %s", exc)
    return ""


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# ROUTES
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

@app.get("/", response_class=HTMLResponse)
async def index():
    if not _INDEX_HTML.exists():
        return HTMLResponse(
            "<h1>index.html not found</h1>"
            "<p>Place <code>index.html</code> in the same folder as <code>app.py</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(body: ChatRequest):
    """Main chat endpoint GÇö powered by Ollama or Pearl AI Hub."""
    log.info("Chat GåÉ %s", body.message[:100])
    
    # Determine which provider to use
    provider = body.provider or AI_PROVIDER
    
    if provider.lower() == "pearl":
        reply = await pearl_chat(body.message, body.history)
        source = f"pearl/{PEARL_MODEL}"
    else:
        # Default to Ollama
        reply = await ollama_chat(body.message, body.history)
        source = f"ollama/{OLLAMA_MODEL}"
    
    log.info("Chat GåÆ %s", reply[:100])
    return JSONResponse({"reply": reply, "source": source})


@app.post("/tts")
async def text_to_speech(body: TTSRequest):
    """Convert text to MP3 via ElevenLabs."""
    clean = clean_text_for_tts(body.text)
    if not clean:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    mp3_bytes = await elevenlabs_tts(clean)
    if not mp3_bytes:
        return JSONResponse(
            {"error": "ElevenLabs TTS unavailable. Check ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID."},
            status_code=503,
        )

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=pearl.mp3"},
    )


@app.websocket("/audio/ws")
async def audio_websocket(websocket: WebSocket):
    """
    Real-time duplex WebSocket for voice interaction.
    Receives raw 16-bit PCM (little-endian mono) and sends back MP3 replies.
    """
    await websocket.accept()
    log.info("Audio WebSocket connected")

    pcm_buffer      = bytearray()
    CHUNK_THRESHOLD = 16000 * 2 * 2   # ~2 s of 16 kHz 16-bit mono

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10)
            except asyncio.TimeoutError:
                continue

            pcm_buffer.extend(data)
            if len(pcm_buffer) < CHUNK_THRESHOLD:
                continue

            chunk = bytes(pcm_buffer)
            pcm_buffer.clear()

            transcript = await elevenlabs_stt(chunk)
            log.info("STT: %r", transcript)

            if transcript.strip():
                reply_text = await ollama_chat(transcript.strip(), [])
                log.info("AI: %s", reply_text[:80])
                mp3_bytes = await elevenlabs_tts(reply_text)
                if mp3_bytes:
                    await websocket.send_bytes(mp3_bytes)

    except WebSocketDisconnect:
        log.info("Audio WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc)


@app.get("/health")
async def health():
    """Health check GÇö tests connectivity to both Ollama and Pearl API."""
    ollama_status = "unknown"
    pearl_status = "unknown"
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception:
        ollama_status = "unreachable"

    # Check Pearl API Hub
    if PEARL_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                headers = {"Authorization": f"Bearer {PEARL_API_KEY}"}
                r = await client.get(f"{PEARL_API_BASE_URL}/health", headers=headers)
                pearl_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception:
            pearl_status = "unreachable"
    else:
        pearl_status = "no_api_key"

    return JSONResponse({
        "status":              "ok",
        "character":           "Miss Pearl",
        "lab":                 "Pearl AI Labs GÇö Kampala, Uganda",
        "active_ai_provider":  AI_PROVIDER,
        "ollama": {
            "model":           OLLAMA_MODEL,
            "base_url":        OLLAMA_BASE_URL,
            "status":          ollama_status,
        },
        "pearl_api_hub": {
            "model":           PEARL_MODEL,
            "base_url":        PEARL_API_BASE_URL,
            "status":          pearl_status,
            "api_key_set":     bool(PEARL_API_KEY),
        },
        "elevenlabs": {
            "key_set":         bool(ELEVENLABS_API_KEY),
            "voice_id":        ELEVENLABS_VOICE_ID,
        },
        "timestamp":           datetime.utcnow().isoformat(),
    })


# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ
# RUN
# GöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇGöÇ

if __name__ == "__main__":
    import uvicorn

    print(f"""
GòöGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòù
Gòæ         MISS PEARL GÇö PEARL AI LABS            Gòæ
Gòæ         Kampala, Uganda  =ƒç¦=ƒç¼                   Gòæ
GòáGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòú
Gòæ  http://localhost:8000                         Gòæ
Gòæ                                                Gòæ
Gòæ  AI PROVIDERS:                                 Gòæ
Gòæ    GÇó Ollama      ({OLLAMA_MODEL:<22})Gòæ
Gòæ    GÇó Pearl API   ({PEARL_MODEL:<22})Gòæ
Gòæ  Active: {AI_PROVIDER.upper():<40}Gòæ
Gòæ                                                Gòæ
Gòæ  Speech: ElevenLabs Scribe v1                  Gòæ
Gòæ  TTS:    ElevenLabs Multilingual v2            Gòæ
GòÜGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGòÉGò¥
    """)

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

