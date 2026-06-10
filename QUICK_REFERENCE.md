# Pearl API Hub — Quick Reference

## 🎯 One-Minute Setup

```bash
# 1. Check .env has Pearl API configured
cat .env | grep PEARL_API

# 2. Set default provider
echo 'AI_PROVIDER=pearl' >> .env

# 3. Start server
python app.py

# 4. Test it works
curl http://localhost:8000/health | jq '.pearl_api_hub'
```

## 📡 Chat Endpoint Examples

### Use Default Provider
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?"}'
```

### Force Pearl API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is AI?",
    "provider": "pearl"
  }'
```

### Force Ollama
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is AI?",
    "provider": "ollama"
  }'
```

## 🔍 Health Check

```bash
# Check both providers
curl http://localhost:8000/health | jq

# Just Pearl API
curl http://localhost:8000/health | jq '.pearl_api_hub'

# Just Ollama
curl http://localhost:8000/health | jq '.ollama'
```

## ⚙️ Configuration

**In `.env`:**

```bash
# Default provider when not specified
AI_PROVIDER=pearl

# Pearl API Hub (OpenAI-compatible cloud)
PEARL_API_BASE_URL=https://apps.pearllabs.ug
PEARL_API_KEY=pearl__k3zN-ihPfuTfWeiNMcXDDpCogbCcJSGnZi2yK_dT64
PEARL_MODEL=qwen

# Ollama (local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## 📚 API Response

Both providers return the same format:

```json
{
  "reply": "Well, AI is artificial intelligence...",
  "source": "pearl/qwen"  // or "ollama/llama3"
}
```

## 💻 Python Example

```python
import httpx
import asyncio

async def chat(msg: str, provider: str = "pearl"):
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8000/chat", json={
            "message": msg,
            "provider": provider,
            "history": []
        })
        data = r.json()
        print(f"{data['source']}: {data['reply']}")

asyncio.run(chat("Tell me a joke", "pearl"))
```

## 🌐 JavaScript Example

```typescript
async function chat(msg: string, provider = "pearl") {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg, provider, history: [] })
  });
  const { reply, source } = await res.json();
  console.log(`${source}: ${reply}`);
}

chat("How does AI work?", "pearl");
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Pearl API unreachable | Check `PEARL_API_KEY` and internet |
| Ollama unreachable | Run `ollama run llama3` |
| Provider not switching | Restart server after changing `.env` |
| Empty response | Check `/health` for provider status |

## 📖 Full Documentation

- **[PROVIDERS.md](PROVIDERS.md)** — Complete guide
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** — Details
- **[README.md](README.md)** — Overview

---

**Provider Status:** `curl http://localhost:8000/health`
