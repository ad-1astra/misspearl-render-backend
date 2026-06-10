# Miss Pearl Render Backend

FastAPI backend for Miss Pearl AI character powered by **dual AI providers**: local Ollama + Pearl AI Hub (OpenAI-compatible).

## 🚀 Features

- **Dual AI Providers**: Switch between local Ollama and cloud Pearl API Hub
- **OpenAI-Compatible API**: Pearl API Hub provides OpenAI knowledge via standard endpoints
- **Smart Enrichment**: Automatic joke & knowledge-base integration
- **Voice Support**: ElevenLabs TTS/STT for speech interaction
- **Real-time WebSocket**: Live voice conversation
- **Health Monitoring**: Endpoint status for both providers

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- (Optional) Ollama running locally: `ollama run llama3`

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env
cp .env.example .env
# Edit .env and set:
#   - AI_PROVIDER=pearl (or "ollama")
#   - PEARL_API_KEY=your-token
#   - ELEVENLABS_API_KEY=your-key

# 3. Run the server
python app.py
```

### Verify Setup
```bash
curl http://localhost:8000/health
```

## 🧠 AI Providers

### Option 1: Pearl API Hub (Cloud, Default)
**Pros**: Broader knowledge, always available, no setup needed  
**Cons**: Requires internet, metered usage

```bash
AI_PROVIDER=pearl
PEARL_API_KEY=pearl__k3zN-ihPfuTfWeiNMcXDDpCogbCcJSGnZi2yK_dT64
```

### Option 2: Ollama (Local LLM)
**Pros**: Fast, private, free, no internet  
**Cons**: Requires local setup, limited to available models

```bash
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## 📡 API Endpoints

### Chat (with provider selection)
```bash
POST /chat
Content-Type: application/json

{
  "message": "Tell me about AI",
  "provider": "pearl",  # Optional: override default
  "history": []
}
```

Response:
```json
{
  "reply": "Well, AI is...",
  "source": "pearl/qwen"
}
```

### Health Check
```bash
GET /health
```

Returns status of both Ollama and Pearl API Hub.

### Text-to-Speech
```bash
POST /tts
Content-Type: application/json

{ "text": "Hello world" }
```

### Voice WebSocket
```
GET /audio/ws
```

Real-time duplex for PCM audio input → MP3 output.

## 🌐 Environment Variables

See [.env](.env) for complete configuration. Key variables:

```bash
# AI Provider Selection
AI_PROVIDER=pearl  # or "ollama"

# Pearl API Hub (OpenAI-compatible)
PEARL_API_BASE_URL=https://apps.pearllabs.ug
PEARL_API_KEY=your-token-here
PEARL_MODEL=qwen

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Voice (ElevenLabs)
ELEVENLABS_API_KEY=your-key-here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

## 📚 Documentation

- **[PROVIDERS.md](PROVIDERS.md)** — Detailed multi-provider guide
- **[app.py](app.py)** — Source code with inline docs

## 🧪 Testing

```bash
# Test Pearl provider
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "provider": "pearl"}'

# Test Ollama provider
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "provider": "ollama"}'

# Check health
curl http://localhost:8000/health | jq
```

## 📦 Deployment

### To Render.com
1. Connect this GitHub repo to Render
2. Set environment variables in Render dashboard:
   - `AI_PROVIDER=pearl`
   - `PEARL_API_KEY=your-token`
   - `ELEVENLABS_API_KEY=your-key`
3. Deploy! (Render will run `python app.py`)

### Files to include
- `app.py`
- `requirements.txt`
- `index.html` (optional)
- `.env` (via Render environment variables)

## 🔧 Troubleshooting

### Pearl API connection fails
- Check `PEARL_API_KEY` is set and valid
- Verify internet connection
- Test: `curl https://apps.pearllabs.ug/health`

### Ollama connection fails
- Start Ollama: `ollama run llama3`
- Check `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- Test: `curl http://localhost:11434/api/tags`

### Voice not working
- Verify `ELEVENLABS_API_KEY` is set
- Check speaker volume
- Test TTS: `curl -X POST http://localhost:8000/tts -d '{"text":"hello"}'`

## 📝 License

Pearl AI Labs — Kampala, Uganda

Created by: Paul & Mr. Divid

---

## See Also
- Pearl AI Labs: https://pearllabs.ug
- Ollama: https://ollama.ai
- ElevenLabs: https://elevenlabs.io

