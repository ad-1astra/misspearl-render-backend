"""
Build a small lip-sync timing benchmark with exact ground truth.

Kokoro-82M renders each sentence in the 4 voices held out from training. Its
predicted phoneme durations give the true viseme for every 10 ms frame, and its
word timestamps play the role of the word/character timestamps that ElevenLabs
returns. tests/eval_alignment.mjs then measures how accurately the browser
places every mouth shape:

  * timestamps only  — phones spread inside each word in proportion to typical durations
  * timestamps + audio network — forced alignment against the trained audio->viseme net

Usage (needs a Python env with `kokoro`, see colab_setup.py):
  python training/make_align_eval.py --out tests/align_eval.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import num_frames  # noqa: E402
from gen_audio_data import (REPO, VAL_VOICES, frame_labels, misaki_to_segments,  # noqa: E402
                            segments_to_visemes)

KOKORO_TS_LEAD = 0.075

# Fresh sentences (not taken from the training corpus): plosives, rounding,
# fricatives, names, questions and long words.
SENTENCES = [
    "Peter Piper picked a peck of pickled peppers.",
    "Which witch watched the big black bat?",
    "Seven shy sheep thought five fat frogs were funny.",
    "Good morning, my brilliant friend, shall we begin today's lesson?",
    "Bright blue balloons bounced above the busy market.",
    "Would you like some warm porridge or a cup of cocoa?",
    "Thank you so much for the wonderful surprise party!",
    "The quick brown fox jumps over the lazy dog.",
    "Mathematics, music and medicine all reward careful practice.",
    "Please remember to bring your umbrella when it rains.",
    "She sells seashells by the seashore every summer.",
    "Our robot carefully measured the very small yellow bottle.",
    "Why do zebras have such beautiful stripes?",
    "I am honestly thrilled to finally meet you in person.",
    "Fresh mangoes and pineapples were piled high on the table.",
    "Thinking deeply about difficult problems makes the mind stronger.",
    "Oh no, I think we missed the last bus home!",
    "Every child deserves a chance to learn and play.",
    "Moonlight shimmered over the quiet lake at midnight.",
    "Would you believe that octopuses have three hearts?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/align_eval.json")
    ap.add_argument("--per-voice", type=int, default=len(SENTENCES), help="sentences per held-out voice")
    args = ap.parse_args()

    import torch
    from scipy.signal import resample_poly
    from kokoro import KModel, KPipeline

    torch.manual_seed(0)
    model = KModel(repo_id=REPO).eval()
    pipes = {lc: KPipeline(lang_code=lc, repo_id=REPO, model=model) for lc in ("a", "b")}
    from collections import Counter, defaultdict
    stats ={"phone": defaultdict(list), "pause": defaultdict(list), "gap": [], "unknown": Counter()}

    utts = []
    for vi, voice in enumerate(VAL_VOICES):
        for k in range(args.per_voice):
            text = SENTENCES[(vi * args.per_voice + k) % len(SENTENCES)]
            for r in pipes[voice[0]](text, voice=voice, speed=1.0, split_pattern=None):
                out = r.output
                audio = out.audio.detach().float().cpu().numpy().reshape(-1)
                dur = out.pred_dur.detach().cpu().numpy().reshape(-1).astype(np.float64)
                kept = [c for c in r.phonemes if model.vocab.get(c) is not None]
                if len(dur) != len(kept) + 2:
                    print("  ! length mismatch, skipped:", text)
                    continue
                spu = (len(audio) / dur.sum()) / 24000.0
                segs = misaki_to_segments(kept, dur, spu, 1.0, stats)
                vsegs = segments_to_visemes(segs)
                wav = resample_poly(audio, 2, 3).astype(np.float32)
                peak = float(np.abs(wav).max()) or 1.0
                if peak > 0.98:
                    wav = wav / peak * 0.7
                n = num_frames(len(wav))
                vis, _ = frame_labels(segs, vsegs, n)
                pcm = np.clip(np.round(wav * 32767), -32768, 32767).astype(np.int16)
                # Kokoro's token timestamps lead the audio by 3 duration units (3 × 25 ms):
                # kokoro.pipeline.join_timestamps starts at 2*(pred_dur[0]-3)/80 s.
                words = [{"text": t.text, "start": float(t.start_ts) + KOKORO_TS_LEAD,
                          "end": float(t.end_ts) + KOKORO_TS_LEAD}
                         for t in (r.tokens or [])
                         if any(c.isalpha() for c in t.text) and t.start_ts is not None and t.end_ts is not None]
                utts.append({"voice": voice, "text": text, "words": words,
                             "vis": [int(x) for x in vis],
                             "pcm16_b64": base64.b64encode(pcm.tobytes()).decode()})
                print(f"  {voice:12s} {len(pcm) / 16000:4.1f}s  {len(words):2d} words  {text}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"sr": 16000, "fps": 100, "utts": utts}))
    print(f"wrote {len(utts)} utterances -> {args.out}")


if __name__ == "__main__":
    main()
