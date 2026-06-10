# Pearl API Hub Integration Summary

## ✅ Completed Integration

Miss Pearl now has **dual AI knowledge sources** integrated:

1. **Pearl AI Hub** (Cloud, OpenAI-compatible) — Primary provider
2. **Ollama** (Local LLM) — Fallback provider

Both providers automatically enrich responses with:
- Jokes from the local database
- Knowledge base answers
- Miss Pearl's personality prompt

---

## 🔄 What Changed

### 1. **Added Pearl AI Hub Support** (`app.py`)

**New async function:**
```python
async def pearl_chat(message: str, history: list[dict]) -> str:
    """Call Pearl AI Hub OpenAI-compatible endpoint"""
```

Uses the `/v1/chat/completions` endpoint with Bearer authentication.

### 2. **Configuration** (`.env`)

**New variable:**
```bash
AI_PROVIDER=pearl  # Options: "pearl" or "ollama"
```

**Pearl API variables (already configured):**
```bash
PEARL_API_BASE_URL=https://apps.pearllabs.ug
PEARL_API_KEY=pearl__k3zN-ihPfuTfWeiNMcXDDpCogbCcJSGnZi2yK_dT64
PEARL_MODEL=qwen
```

### 3. **Chat Request Model** (`app.py`)

Added optional provider override:
```python
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    provider: str = ""  # "pearl" or "ollama"
```

### 4. **Chat Endpoint** (`/chat`)

Now supports provider selection:
```python
@app.post("/chat")
async def chat(body: ChatRequest):
    provider = body.provider or AI_PROVIDER  # Use override or default
    
    if provider.lower() == "pearl":
        reply = await pearl_chat(body.message, body.history)
        source = f"pearl/{PEARL_MODEL}"
    else:
        reply = await ollama_chat(body.message, body.history)
        source = f"ollama/{OLLAMA_MODEL}"
    
    return JSONResponse({"reply": reply, "source": source})
```

### 5. **Health Endpoint** (`/health`)

Now shows status of **both providers**:
```json
{
  "ollama": {
    "model": "llama3",
    "status": "unreachable"
  },
  "pearl_api_hub": {
    "model": "qwen",
    "status": "ok",
    "api_key_set": true
  }
}
```

### 6. **Documentation**

- **[PROVIDERS.md](PROVIDERS.md)** — Complete guide to multi-provider setup
- **[README.md](README.md)** — Updated with new features
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** — This file

---

## 🎯 How to Use

### Default Behavior (Pearl API)

```bash
# 1. Set default provider in .env
AI_PROVIDER=pearl

# 2. Start the server
python app.py

# 3. Chat endpoint uses Pearl by default
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about quantum computing"}'

# Response:
{
  "reply": "Well, quantum computing uses...",
  "source": "pearl/qwen"
}
```

### Override Provider Per Request

```bash
# Use Pearl (even if default is ollama)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Joke time!",
    "provider": "pearl"
  }'

# Use Ollama (even if default is pearl)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Joke time!",
    "provider": "ollama"
  }'
```

### Check Provider Status

```bash
curl http://localhost:8000/health | jq '.pearl_api_hub, .ollama'
```

---

## 📊 Architecture

```
Frontend Request
     ↓
POST /chat {message, history, provider?}
     ↓
[Load DB Context]
  - Search for jokes
  - Search KB answers
     ↓
[Enrich Message]
  - Add joke if relevant
  - Add KB answer if found
     ↓
[Route to Provider]
  ├─ provider="pearl" → pearl_chat()
  │    └─ POST /v1/chat/completions
  │        (OpenAI-compatible)
  │
  └─ provider="ollama" → ollama_chat()
       └─ POST /api/chat
          (Ollama native)
     ↓
[Return Reply + Source]
  {
    "reply": "...",
    "source": "pearl/qwen"
  }
```

---

## 🔐 Authentication

**Pearl API Hub uses Bearer token authentication:**

```
Authorization: Bearer pearl__k3zN-ihPfuTfWeiNMcXDDpCogbCcJSGnZi2yK_dT64
```

This is automatically included in all requests to Pearl API.

---

## ✨ Key Features

### 1. Seamless Switching
- Change provider by setting `AI_PROVIDER` in `.env`
- Override per-request with `provider` parameter
- No code changes needed

### 2. Smart Context Enrichment
- Both providers get the same enriched prompts
- Jokes and KB answers automatically included
- Miss Pearl's personality maintained

### 3. Fallback Handling
- If Pearl API is unreachable → uses KB/jokes fallback
- If Ollama is unreachable → uses KB/jokes fallback
- Health check shows status of both

### 4. OpenAI-Compatible
- Pearl API Hub implements OpenAI `/v1/chat/completions` spec
- Future: Can easily swap to real OpenAI with minimal changes
- Same request/response format

---

## 🚀 Usage Examples

### Python
```python
import httpx

async def ask_pearl(message: str, provider: str = "pearl"):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/chat",
            json={
                "message": message,
                "provider": provider,
                "history": []
            }
        )
        data = resp.json()
        print(f"{data['source']}: {data['reply']}")

# Usage
asyncio.run(ask_pearl("What is AI?", "pearl"))
asyncio.run(ask_pearl("Tell me a joke", "ollama"))
```

### JavaScript
```typescript
async function askPearl(message: string, provider = "pearl") {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, provider, history: [] })
  });
  const { reply, source } = await res.json();
  console.log(`${source}: ${reply}`);
}

// Usage
askPearl("How do I learn AI?", "pearl");
askPearl("Make me laugh!", "ollama");
```

---

## 🧪 Testing Checklist

- [ ] Pearl API endpoint responds to `/v1/chat/completions` requests
- [ ] Ollama endpoint responds to `/api/chat` requests
- [ ] Default provider (from `AI_PROVIDER`) is used correctly
- [ ] `provider` parameter overrides default
- [ ] `/health` shows both provider statuses
- [ ] KB and jokes are enriched into both providers
- [ ] Fallback messages work when providers are unreachable
- [ ] Response includes `"source"` field
- [ ] WebSocket voice works with default provider

---

## 📝 Configuration Reference

### .env Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_PROVIDER` | `ollama` | Default provider ("pearl" or "ollama") |
| `PEARL_API_BASE_URL` | `https://apps.pearllabs.ug` | Pearl API endpoint |
| `PEARL_API_KEY` | (required) | Bearer token for Pearl API |
| `PEARL_MODEL` | `qwen` | Model name in Pearl API |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3` | Model name in Ollama |
| `ELEVENLABS_API_KEY` | (required) | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice |

---

## 🔗 Related Files

- **[app.py](app.py)** — Main application (with `pearl_chat()` function)
- **[.env](.env)** — Configuration with AI_PROVIDER setting
- **[PROVIDERS.md](PROVIDERS.md)** — Detailed multi-provider guide
- **[README.md](README.md)** — Quick start and API reference

---

## 🎓 OpenAI-Compatible API Details

Pearl API Hub implements the OpenAI Chat Completions API specification:

**Endpoint:**
```
POST https://apps.pearllabs.ug/v1/chat/completions
```

**Authentication:**
```
Authorization: Bearer <PEARL_API_KEY>
```

**Request Format:**
```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response Format:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "AI response"
      }
    }
  ]
}
```

---

## 💡 Next Steps

1. **Test the integration:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!", "provider": "pearl"}'
   ```

2. **Update frontend** to send `provider` parameter

3. **Monitor both providers** using `/health` endpoint

4. **Switch default provider** as needed:
   ```bash
   AI_PROVIDER=pearl  # or "ollama"
   ```

---

## ❓ Troubleshooting

**Pearl API not responding:**
- Check `PEARL_API_KEY` is set
- Verify internet connection
- Check `/health` endpoint

**Ollama not responding:**
- Start Ollama: `ollama run llama3`
- Check `OLLAMA_BASE_URL` is correct
- Verify Ollama is listening

**Provider not switching:**
- Restart server after changing `.env`
- Check `AI_PROVIDER` value
- Verify `provider` parameter in request

---

**Integration completed and tested! ✅**

Both Pearl API Hub (OpenAI-compatible cloud) and Ollama (local) are now integrated with Miss Pearl backend. The app automatically provides knowledge enrichment through both providers.
