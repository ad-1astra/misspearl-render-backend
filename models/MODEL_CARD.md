# Miss Pearl — trained face models

Face models trained 2026-10-31 16:58 UTC, song model 2026-11-01 07:25 UTC, on Colab
({'python': '3.13.16', 'torch': '2.11.0+cu130', 'cuda': True, 'gpu': 'Tesla T4'}).

| Model | What it does | Result |
|---|---|---|
| audio2viseme | speech audio → 15 visemes @100 fps | 0.836 frame accuracy, 0.9099 within ±20 ms, on 547 utterances from **voices never seen in training** (6.12 h of training audio, 228695 params) |
| g2p | any English word → ARPAbet phones | phone error rate 0.0895 (no-stress 0.0694), word accuracy 0.6256 on 3747 held-out words |
| emotion | sentence → 7 emotions | 0.7217 agreement with the DistilRoBERTa teacher; 0.823 accuracy on human-labelled dair-ai/emotion |
| sing2viseme | song mix (voice + music) → 15 visemes of the singing voice + voice activity @100 fps | with voice and music equally loud: 0.738 of frames within ±20 ms (speech model 0.407), voice F1 0.910, mouth shut on 0.934 of music-only frames (speech model 0.149); real songs by unseen artists: voice F1 0.829 (701368 params; 4.82 h sung + 4.73 h speech + 8.8 h of music) |

Viseme set (Meta/Oculus): sil PP FF TH DD kk CH SS nn RR aa E ih oh ou.

Song-model data: lyric-like lines sung by Kokoro-82M (public-domain poetry from `biglam/gutenberg-poetry-corpus`, everyday
phrases from wikitext / OpenAssistant, vocables), with a melody and held vowels rendered through Kokoro's own decoder
(`training/sing_synth.py`); instrumentals and real songs from the Free Music Archive (`fma_small`, Creative Commons),
separated with Demucs htdemucs. Train/validation split by voice and by artist.
