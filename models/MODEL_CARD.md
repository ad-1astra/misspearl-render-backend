# Miss Pearl — trained face models

Trained 2026-10-31 16:58 UTC on {'python': '3.13.16', 'torch': '2.11.0+cu130', 'cuda': True, 'gpu': 'Tesla T4'}.

| Model | What it does | Result |
|---|---|---|
| audio2viseme | speech audio → 15 visemes @100 fps | 0.836 frame accuracy, 0.9099 within ±20 ms, on 547 utterances from **voices never seen in training** (6.12 h of training audio, 228695 params) |
| g2p | any English word → ARPAbet phones | phone error rate 0.0895 (no-stress 0.0694), word accuracy 0.6256 on 3747 held-out words |
| emotion | sentence → 7 emotions | 0.7217 agreement with the DistilRoBERTa teacher; 0.823 accuracy on human-labelled dair-ai/emotion |

Viseme set (Meta/Oculus): sil PP FF TH DD kk CH SS nn RR aa E ih oh ou.
