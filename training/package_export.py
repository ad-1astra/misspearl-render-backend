"""
Step 5 — collect everything the browser needs into one folder + zip.

export/
  audio2viseme.json/.bin   trained audio→viseme network
  g2p.json/.bin            trained grapheme→phoneme network
  emotion.json/.bin        trained emotion network
  lexicon.txt              CMUdict pronunciations (compact)
  durations.json           learned phone & pause durations
  metrics/ tests/          evaluation results and JS test vectors
  MODEL_CARD.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import env_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    data, export = Path(args.data), Path(args.export)

    if (data / "durations.json").exists():
        shutil.copy(data / "durations.json", export / "durations.json")

    def load(p):
        p = export / "metrics" / p
        return json.loads(p.read_text()) if p.exists() else {}

    a2v, g2p, emo = load("audio2viseme.json"), load("g2p.json"), load("emotion.json")
    fa = a2v.get("final_unseen_voices", {})
    card = f"""# Miss Pearl — trained face models

Trained {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} on {env_info()}.

| Model | What it does | Result |
|---|---|---|
| audio2viseme | speech audio → 15 visemes @100 fps | {fa.get('viseme_acc', '?')} frame accuracy, {fa.get('viseme_acc_tol20ms', '?')} within ±20 ms, on {a2v.get('val_utts', '?')} utterances from **voices never seen in training** ({a2v.get('train_hours', '?')} h of training audio, {a2v.get('params', '?')} params) |
| g2p | any English word → ARPAbet phones | phone error rate {g2p.get('final_heldout', {}).get('per', '?')} (no-stress {g2p.get('final_heldout', {}).get('per_no_stress', '?')}), word accuracy {g2p.get('final_heldout', {}).get('word_acc', '?')} on {g2p.get('val_words', '?')} held-out words |
| emotion | sentence → 7 emotions | {emo.get('final', {}).get('teacher_agreement', '?')} agreement with the DistilRoBERTa teacher; {emo.get('final', {}).get('gold_acc_dair', '?')} accuracy on human-labelled dair-ai/emotion |

Viseme set (Meta/Oculus): sil PP FF TH DD kk CH SS nn RR aa E ih oh ou.
"""
    (export / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    zp = Path(args.zip)
    zp.parent.mkdir(parents=True, exist_ok=True)
    out = shutil.make_archive(str(zp.with_suffix("")), "zip", export)
    print(card)
    print("packaged →", out, f"({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
