# Miss Pearl — Multi-Provider AI Backend

Miss Pearl now supports **dual AI knowledge sources**: local Ollama models and Pearl AI Hub (OpenAI-compatible cloud API).

## Overview

The backend can seamlessly switch between:
- **Ollama** (local LLM)—fast, private, no internet required
- **Pearl AI Hub** (cloud, OpenAI-compatible)—access to advanced models, broader knowledge

Both providers maintain the same Miss Pearl personality and knowledge enrichment (jokes, knowledge base).

---

## Configuration

### Set Default Provider

In `.env`:
```bash
# Options: "ollama" or "pearl"
AI_PROVIDER=pearl
```

- `ollama` = Use local Ollama (requires Ollama running locally)
- `pearl` = Use Pearl API Hub (requires internet and valid API key)

### Required Environment Variables

**For Pearl AI Hub (OpenAI-compatible):**
```bash
PEARL_API_BASE_URL=https://apps.pearllabs.ug
PEARL_API_KEY=pearl__k3zN-ihPfuTfWeiNMcXDDpCogbCcJSGnZi2yK_dT64
PEARL_MODEL=qwen
```

**For Ollama (optional, defaults shown):**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## Usage

### 1. Default Provider (Server Config)

The server uses the provider set in `AI_PROVIDER`:

```bash
# Terminal 1: Start the backend
python app.py

# Terminal 2: Send a chat request (uses default AI_PROVIDER)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'

# Response will show which provider was used:
{
  "reply": "...",
  "source": "pearl/qwen"  # or "ollama/llama3"
}
```

### 2. Override Provider Per Request

Clients can override the default by sending `"provider"` in the request:

```bash
# Use Pearl API Hub (even if default is ollama)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is quantum computing?",
    "provider": "pearl",
    "history": []
  }'

# Use Ollama (even if default is pearl)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is quantum computing?",
    "provider": "ollama",
    "history": []
  }'
```

### 3. Check Health & Status

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "character": "Miss Pearl",
  "active_ai_provider": "pearl",
  "ollama": {
    "model": "llama3",
    "base_url": "http://localhost:11434",
    "status": "unreachable"  // or "ok"
  },
  "pearl_api_hub": {
    "model": "qwen",
    "base_url": "https://apps.pearllabs.ug",
    "status": "ok",
    "api_key_set": true
  },
  "elevenlabs": {
    "key_set": true,
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  }
}
```

---

## How It Works

Both providers follow the same enrichment pipeline:

```
User Message
    ↓
[Load from local DB]
  • Search for matching jokes
  • Search for matching knowledge base answers
    ↓
[Enrich Message with Context]
  • Append jokes (if request mentions humor)
  • Append KB answers (if matching)
    ↓
[Build Conversation]
  • Miss Pearl system prompt
  • Last 10 turns of history
  • Enriched user message
    ↓
[Call AI Provider]
  ├─ Ollama: POST /api/chat
  └─ Pearl: POST /v1/chat/completions (OpenAI-compatible)
    ↓
[Return Reply + Source]
  • Response: "..."
  • Source: "pearl/qwen" or "ollama/llama3"
    ↓
[Save to Interaction DB]
```

### Knowledge Enrichment

Both providers automatically enrich prompts with:
- **Jokes**: If message contains keywords like "joke", "funny", "laugh"
- **Knowledge Base**: If message keywords match stored KB answers

This happens transparently—the local knowledge is always available regardless of provider.

---

## Pearl API Hub Endpoints

The backend uses the OpenAI-compatible endpoint:

```
POST /v1/chat/completions
```

**Request:**
```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "You are Miss Pearl..."},
    {"role": "user", "content": "What is AI?"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Well, dear..."
      }
    }
  ]
}
```

---

## Performance & Latency

| Provider | Setup | Latency | Cost | Privacy |
|----------|-------|---------|------|---------|
| **Ollama** | Local LLM | ~500ms–2s | Free | ✅ Full |
| **Pearl API** | Cloud (OpenAI-compatible) | ~1–3s | Metered | 🌐 Cloud |

- Choose **Ollama** for fast local responses and privacy
- Choose **Pearl API** for advanced model capabilities and broader knowledge

---

## Switching Providers

### Option 1: Change Default in `.env`
```bash
# Restart the server to take effect
AI_PROVIDER=pearl
```

### Option 2: Override Per Request
```bash
curl ... -d '{"message": "...", "provider": "pearl"}'
```

### Option 3: Programmatic (Frontend)
When building chat UI, send `provider` parameter:
```typescript
const response = await fetch('/chat', {
  method: 'POST',
  body: JSON.stringify({
    message: userInput,
    provider: selectedProvider, // 'pearl' or 'ollama'
    history: chatHistory
  })
});
```

---

## Error Handling

### Pearl API Unavailable
If `PEARL_API_KEY` is missing or Pearl API is unreachable, the fallback triggers:
- Returns scripted response
- Logs error
- Uses KB + jokes if available

### Ollama Unavailable
If Ollama is not running, the fallback triggers:
- Returns scripted response
- Logs error
- Uses KB + jokes if available

Check `/health` endpoint to diagnose connection issues.

---

## API Reference

### `POST /chat`

**Request Body:**
```json
{
  "message": "What is quantum computing?",
  "provider": "pearl",  // Optional: "pearl" or "ollama"
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

**Response:**
```json
{
  "reply": "Well, quantum computing is...",
  "source": "pearl/qwen"
}
```

**Status Codes:**
- `200` – Success
- `400` – Invalid request
- `500` – Server error (fallback triggered)

---

## Examples

### Python Client
```python
import httpx

async def chat_with_pearl(message: str, provider: str = "pearl"):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/chat",
            json={"message": message, "provider": provider}
        )
        data = resp.json()
        print(f"Reply: {data['reply']}")
        print(f"Source: {data['source']}")

# Usage
asyncio.run(chat_with_pearl("Tell me about AI", provider="pearl"))
```

### TypeScript/Frontend
```typescript
async function askPearl(message: string, provider = "pearl") {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, provider })
  });
  const data = await res.json();
  console.log(`${data.source}: ${data.reply}`);
}

// Usage
askPearl("How can I learn AI?", "pearl");
```

---

## Troubleshooting

### "Cannot connect to Pearl API"
- Verify `PEARL_API_BASE_URL` is correct
- Check `PEARL_API_KEY` is valid
- Ensure internet connection is available

### "Cannot connect to Ollama"
- Start Ollama: `ollama run llama3`
- Verify `OLLAMA_BASE_URL` is correct
- Check Ollama is listening on the configured port

### Provider not switching
- Check `.env` for correct `AI_PROVIDER` value
- Restart server after changing `.env`
- Verify `/health` endpoint shows correct status

---

## Architecture

```
┌──────────────────────────────────────┐
│    Miss Pearl Frontend               │
│  (index.html / React / Flutter)      │
└────────────┬─────────────────────────┘
             │ /chat requests
             │ {message, history, provider?}
             ↓
┌──────────────────────────────────────┐
│    FastAPI Backend (app.py)          │
│                                      │
│  ├─ /chat endpoint                  │
│  │  ├─ Load DB (jokes, KB)          │
│  │  ├─ Enrich message               │
│  │  └─ Route to provider:           │
│  │     ├─ ollama_chat()             │
│  │     └─ pearl_chat()              │
│  │                                  │
│  ├─ /health endpoint                │
│  ├─ /tts endpoint (ElevenLabs)      │
│  └─ /audio/ws (WebSocket)           │
└────┬─────────────────┬──────────────┘
     │                 │
     ↓                 ↓
┌──────────────┐  ┌──────────────────────┐
│   Ollama     │  │  Pearl API Hub       │
│   (Local)    │  │  (Cloud OpenAI-compat)
│ /api/chat    │  │  /v1/chat/completions
└──────────────┘  └──────────────────────┘
```

---

## See Also

- [README.md](README.md) — Setup and general info
- [.env](.env) — Configuration file
- [app.py](app.py) — Backend source
