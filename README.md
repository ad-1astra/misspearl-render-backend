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
| Speech | `pearl/speech.js` | **Browser voice**: text timeline re-anchored on every word-boundary event and rate-calibrated per voice. **ElevenLabs** (`/tts/timed`): character timestamps give every word its true span, then the trained audio→viseme network force-aligns each mouth shape on the real audio (see the benchmark below). **Any audio**: the trained network reads visemes straight from the sound. |
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

### Lip-sync timing benchmark
How accurately the face places each mouth shape when it gets what a timestamped TTS gives it (audio + word/character
times). 80 sentences written for the test, spoken by the 4 Kokoro voices held out of training, scored against
Kokoro's exact phoneme timing (`models/metrics/lipsync_timing.json`):

| Method | Frames exact | Within ±20 ms | P/B/M lip closures within 30 ms |
|---|---|---|---|
| timestamps only (phones spread inside each word) | 64.1% | 76.2% | 78% |
| **timestamps + audio network (what ElevenLabs mode uses)** | **78.1%** | **91.5%** | **93%** |
| same, with ±40 ms noise added to every timestamp | 77.8% | 91.2% | 93% |
| audio network alone (no text) | 80.1% | 87.8% | 96% |

Forced alignment keeps the mouth-shape *sequence* from the text and takes the *timing* from the sound, so it is the
most accurate within ±20 ms and barely changes when the timestamps are noisy. The test voices are Kokoro voices, the
same kind of audio the network was trained on; ElevenLabs voices are a different source, so don't read the numbers
as an exact score for them.

```
python training/make_align_eval.py --out tests/align_eval.json   # needs kokoro (see colab_setup.py)
node tests/eval_alignment.mjs tests/align_eval.json
```

### Retrain
Open the notebook in Colab (T4 GPU) → *Runtime ▸ Run all*. It clones this repo, builds a
phoneme-aligned speech corpus with Kokoro-82M (an isolated Python 3.12 env is created automatically),
trains the three models and downloads `misspearl_face_models.zip`. Unzip it into `models/`.

## Using it

* **GitHub Pages / offline:** open the live link. *Say* speaks anything you type (try `[sad]`, `[surprised]`, `*laughs*`);
  *Mirror* lip-syncs your microphone live; *Audio* lip-syncs any recording you drop in.
  The expression buttons preview every emotion. The *Face brain* panel shows the live viseme, word, sync and emotion readings.
* **With the backend:** `⚙ Settings → Backend URL` (or `?api=https://your-app.onrender.com`). Chat replies come from the
  LLM with per-sentence `[emotion]` tags, and the *Auto* voice engine (default) speaks with ElevenLabs, with the
  audio-aligned lip-sync above.

## Backend (`app.py`)

```
pip install -r requirements.txt
python app.py            # → http://localhost:8000 (serves the face too)
```

| Route | |
|---|---|
| `GET /` | the face (also `/pearl/*`, `/models/*`) |
| `POST /chat` | `{message, history}` → `{reply, tagged, source}`. `tagged` keeps the LLM's `[emotion]` tags |
| `POST /tts` | `{text, voice_id?}` → MP3 |
| `POST /tts/timed` | `{text, voice_id?}` → `{audio_base64, alignment, voice_id, voice_name}` from ElevenLabs `/with-timestamps` |
| `GET /voices` | ElevenLabs voices this backend can use |
| `GET /health` | provider status, ElevenLabs key + voice in use, face models |

> **ElevenLabs voices:** the free plan can't use *library* voices over the API (`402 paid_plan_required`; that
> includes the old default "Rachel", `21m00Tcm4TlvDq8ikWAM`). ElevenLabs' own default voices work on every plan. The
> backend tries the requested voice, then `ELEVENLABS_VOICE_ID`, then the defaults (Jessica, Sarah, Matilda, Alice,
> Laura, Lily) and remembers which ones were refused, so she always speaks with *some* voice. Pick one in
> ⚙ Settings → ElevenLabs voice. If ElevenLabs fails completely (no key, quota used up), the face falls back to the
> browser voice and says why in the status line.

Environment: `AI_PROVIDER` (`ollama`/`pearl`), `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `PEARL_API_BASE_URL`,
`PEARL_API_KEY`, `PEARL_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`. Keep them in `.env`, which is git-ignored.

### Deploy to Render
New Web Service → this repo → start command `uvicorn app:app --host 0.0.0.0 --port $PORT` → add the environment variables.
