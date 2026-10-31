# Miss Pearl — Face, Lip-Sync & Backend

**Live face:** https://ad-1astra.github.io/misspearl-render-backend/

Miss Pearl is the animated face of Pearl AI Labs (Kampala). She can say **any** word with
viseme-accurate lip-sync, and her face reads and acts out the emotion of every sentence
(brows, eyelids, gaze, cheeks, smile, head motion). The face runs entirely in the browser,
so it works on GitHub Pages. `app.py` adds the LLM brain and ElevenLabs voice when deployed.

## How the face works

```
text ─┬─ [emotion] tags from the LLM ──────────────┐
      ├─ emotion network (trained) ─────────────────┤→ expression engine → brows · lids · cheeks · smile · head
      └─ normalise → CMU lexicon / neural G2P ─→ phones ─→ viseme timeline ─┐
                                                                            ├→ coarticulation → mouth (15 visemes)
audio ──── audio→viseme network (trained) ─→ per-frame viseme probabilities ┘
```

| Piece | File | What it does |
|---|---|---|
| Face rig | `pearl/face.js` | SVG face with 15 Meta/Oculus viseme mouth shapes (teeth, tongue, lip tuck, rounding), 12 emotions, blinks, micro-saccades, gaze, brow flashes on stressed syllables, laugh/wink/nod/blush gestures. Everything is spring-smoothed. |
| Text → visemes | `pearl/text.js` | Emotion-tag parsing, number/date/money/abbreviation normalisation, CMUdict lexicon (125k words) + neural G2P for unknown words, learned phone durations, viseme timelines, re-timing from TTS character alignment. |
| Speech | `pearl/speech.js` | **Browser voice**: text timeline re-anchored on every word-boundary event and rate-calibrated per voice. **ElevenLabs** (`/tts/timed`): character timestamps place each phoneme exactly. **Any audio**: the trained network reads visemes straight from the sound. |
| NN runtime | `pearl/nn.js`, `pearl/audio.js` | Dependency-free JS inference for all three models + log-mel features identical to training. |

## The trained models (`models/`)

Trained in Google Colab with [`training/MissPearl_LipSync_Training.ipynb`](training/MissPearl_LipSync_Training.ipynb);
everything (corpus, checkpoints, exports) is saved to Google Drive → `MyDrive/MissPearl-LipSync/`.

| Model | Trained on | Result (held-out data) | Size |
|---|---|---|---|
| **audio → viseme** (`audio2viseme`) — 8-block dilated TCN on 64-band log-mel, ±310 ms context, 100 fps | 6.12 h of speech rendered by Kokoro-82M in 24 voices × random speeds, labelled frame-by-frame from Kokoro's own phoneme durations (3,832 utterances in total), with noise / reverb / gain / band-limit augmentation | **83.6%** frame accuracy over 15 visemes, **91.0%** within ±20 ms, macro-F1 0.835 — on 547 utterances from **4 voices never heard in training**; 77.7% at 15 dB SNR noise | 228,695 params |
| **G2P** (`g2p`) — 4-layer Transformer, CTC, letters×3 | CMU Pronouncing Dictionary (121,179 words) | phone error rate **8.9%** (6.9% ignoring stress), **62.6%** of 3,747 unseen words perfect | 818,630 params |
| **emotion** (`emotion`) — bag-of-n-grams MLP, int8 | 114,906 sentences (GoEmotions, dair-ai/emotion, OpenAssistant, Wikipedia) soft-labelled by `j-hartmann/emotion-english-distilroberta-base` | **72.2%** agreement with the RoBERTa teacher; 82.3% on human-labelled dair-ai/emotion (teacher: 89.4%) | ~1 MB |
| durations (`durations.json`) | median phone & pause lengths measured in the Kokoro corpus | drives the text-only timeline | — |


Each exporter checks that the exported maths matches PyTorch. In the browser runtime:

```
node tests/verify_models.mjs     # JS features/logits/phones/emotions == PyTorch (max |Δ| ≈ 5e-5)
node tests/verify_streaming.mjs  # live-mic streaming == whole-clip analysis (bit-exact)
```
`tests/face-sheet.html` renders every viseme and emotion side by side (`#mouths` zooms on the mouth).

### Retrain
Open the notebook in Colab (T4 GPU) → *Runtime ▸ Run all*. It clones this repo, builds a
phoneme-aligned speech corpus with Kokoro-82M (an isolated Python 3.12 env is created automatically),
trains the three models and downloads `misspearl_face_models.zip`. Unzip it into `models/`.

## Using it

* **GitHub Pages / offline:** open the live link. *Say* speaks anything you type (try `[sad]`, `[surprised]`, `*laughs*`);
  *Mirror* lip-syncs your microphone live; *Audio* lip-syncs any recording you drop in.
  The expression buttons preview every emotion. The *Face brain* panel shows the live viseme, word, sync and emotion readings.
* **With the backend:** `⚙ Settings → Backend URL` (or `?api=https://your-app.onrender.com`). Chat replies come from the
  LLM with per-sentence `[emotion]` tags, and *Voice engine → ElevenLabs* gives sample-accurate lip-sync.

## Backend (`app.py`)

```
pip install -r requirements.txt
python app.py            # → http://localhost:8000 (serves the face too)
```

| Route | |
|---|---|
| `GET /` | the face (also `/pearl/*`, `/models/*`) |
| `POST /chat` | `{message, history}` → `{reply, tagged, source}`. `tagged` keeps the LLM's `[emotion]` tags |
| `POST /tts` | MP3 |
| `POST /tts/timed` | `{audio_base64, alignment}` from ElevenLabs `/with-timestamps` |
| `GET /health` | provider status, ElevenLabs key presence, face models |

> **ElevenLabs note:** the default voice (`21m00Tcm4TlvDq8ikWAM`) is a *library* voice, and ElevenLabs refuses
> library voices over the API on the free plan (`402 paid_plan_required`). Use a paid plan/credits or a voice you own.
> Until then the face falls back to the browser voice automatically and says why in the status line.

Environment: `AI_PROVIDER` (`ollama`/`pearl`), `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `PEARL_API_BASE_URL`,
`PEARL_API_KEY`, `PEARL_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`. Keep them in `.env`, which is git-ignored.

### Deploy to Render
New Web Service → this repo → start command `uvicorn app:app --host 0.0.0.0 --port $PORT` → add the environment variables.
