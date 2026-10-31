"""
Step 1 — build a phoneme-aligned speech corpus with Kokoro-82M.

Kokoro is a duration-based TTS: for every phoneme it predicts how many
frames to hold it before the decoder renders audio.  Those predicted
durations ARE the alignment of the audio it produces, so every
synthesized sentence comes with exact phoneme timing.  We render many
sentences across all English voices and speaking rates, map phonemes to
visemes, and store frame-level labels at 100 fps.

Outputs (in --data):
  audio.npy  int16 16 kHz, all utterances concatenated
  vis.npy    uint8 viseme label per 10 ms frame
  ph.npy     uint8 aux phone label per frame
  index.json per-utterance offsets, voice, speed, text, phonemes
  durations.json  learned phone / pause durations (used by the browser's
                  text-driven timeline when only text is available)
  samples/*.wav   a few rendered examples
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (AUX_PHONES, HOP, MISAKI_LENGTH, MISAKI_PUNCT, MISAKI_STRESS,  # noqa: E402
                    MISAKI_TO_ARPA, PHONE_TO_VISEME, SR, VIS_INDEX, VISEMES, VOWELS, WIN,
                    num_frames, save_json)

REPO = "hexgrad/Kokoro-82M"
VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric",
    "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable",
    "bm_george", "bm_lewis",
]
VAL_VOICES = ["af_nova", "am_echo", "bf_isabella", "bm_lewis"]

PEARL_LINES = [
    "Well hello there, my genius friend!", "Tickety-tock, let's get to work.",
    "Bless your heart, that is a wonderful question.", "Pearl AI Labs is so proud of your curiosity!",
    "Well now, dear, let me explain.", "Oh my goodness, that is amazing!",
    "I'm so sorry to hear that.", "Hmm, let me think about that for a moment.",
    "Absolutely, I would love to help you with that.", "Kampala is a beautiful city.",
    "Photosynthesis turns sunlight, water and carbon dioxide into sugar.",
    "The quick brown fox jumps over the lazy dog.", "Peter Piper picked a peck of pickled peppers.",
    "She sells seashells by the seashore.", "How much wood would a woodchuck chuck?",
    "Five fine friends found four fresh fish.", "Thank you very much, you are too kind!",
    "Really? I did not expect that at all!", "That makes me a little sad, to be honest.",
    "Please don't do that again, it's not safe.", "Yes! We did it!", "No, no, no. Try again.",
    "Are you ready? Let's begin.", "Good morning! How are you feeling today?",
    "Mathematics is the language of the universe.", "Uganda is the pearl of Africa.",
    "Would you like to hear a joke?", "Why did the chicken cross the road?",
    "My favourite number is three hundred and sixty five.", "Wow, that is incredible!",
]

_ok = re.compile(r"^[A-Za-z0-9 ,.'!?;:\-]+$")


def _clean(s: str) -> str:
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r"\s+([,.;:!?'])", r"\1", s)
    s = s.replace("( ", "(").replace(" )", ")").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def _sentences(text: str):
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = _clean(s)
        n = len(s.split())
        if 3 <= n <= 28 and _ok.match(s) and not s.startswith("="):
            yield s


def load_corpus(n_target: int, seed: int):
    from datasets import load_dataset
    rng = random.Random(seed)
    wiki, chat = [], []
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        for row in ds:
            wiki.extend(_sentences(row["text"]))
        print(f"  wikitext sentences: {len(wiki):,}")
    except Exception as e:  # pragma: no cover
        print("  ! wikitext unavailable:", e)
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        for row in ds:
            if row["role"] == "assistant" and row["lang"] == "en":
                chat.extend(_sentences(row["text"]))
        print(f"  oasst1 assistant sentences: {len(chat):,}")
    except Exception as e:  # pragma: no cover
        print("  ! oasst1 unavailable:", e)
    rng.shuffle(wiki)
    rng.shuffle(chat)
    half = n_target // 2
    pool = chat[:half] + wiki[: n_target - min(half, len(chat))]
    pool = list(dict.fromkeys(pool))  # dedupe, keep order
    rng.shuffle(pool)
    return PEARL_LINES * 2 + pool


def misaki_to_segments(kept, dur, spu, speed, stats):
    """kept: misaki symbols accepted by the model; dur: pred_dur incl. BOS/EOS.
    Returns [(phone|'SIL', stress, t0, t1)] in seconds."""
    items = []
    t = float(dur[0]) * spu
    items.append(["SIL", "", 0.0, t, "bos"])
    for c, d in zip(kept, dur[1:-1]):
        dt = float(d) * spu
        items.append([c, "", t, t + dt, "raw"])
        t += dt
    items.append(["SIL", "", t, t + float(dur[-1]) * spu, "eos"])

    out = []
    pending_stress, carry_t0 = "", None
    i = 0
    while i < len(items):
        sym, _, t0, t1, kind = items[i]
        if kind != "raw":
            out.append(["SIL", "", t0, t1])
        elif sym in MISAKI_STRESS:
            pending_stress = MISAKI_STRESS[sym]
            carry_t0 = t0 if carry_t0 is None else carry_t0
        elif sym in MISAKI_LENGTH:
            if out:
                out[-1][3] = t1
        elif sym == " ":
            if t1 - t0 >= 0.075:
                out.append(["SIL", "", t0, t1])
            else:
                stats["gap"].append((t1 - t0) * speed)
                mid = (t0 + t1) / 2
                if out:
                    out[-1][3] = mid
                carry_t0 = mid if carry_t0 is None else carry_t0
        elif sym in MISAKI_PUNCT:
            stats["pause"][sym].append((t1 - t0) * speed)
            out.append(["SIL", "", t0, t1])
        elif sym in MISAKI_TO_ARPA:
            ph = MISAKI_TO_ARPA[sym]
            stress = ""
            if ph in VOWELS:
                stress = pending_stress or "0"
                pending_stress = ""
            start = carry_t0 if carry_t0 is not None else t0
            carry_t0 = None
            out.append([ph, stress, start, t1])
        else:
            stats["unknown"][sym] += 1
            if out:
                out[-1][3] = t1
        i += 1
    # merge consecutive SILs and fix any carried start time
    merged = []
    for seg in out:
        if merged and seg[0] == "SIL" and merged[-1][0] == "SIL":
            merged[-1][3] = seg[3]
        else:
            if merged:
                seg[2] = merged[-1][3]
            merged.append(seg)
    for ph, st, t0, t1 in merged:
        if ph != "SIL":
            stats["phone"][ph + st].append((t1 - t0) * speed)
    return merged


def segments_to_visemes(segs):
    """Expand phones to viseme segments (diphthongs split, HH -> next)."""
    vis = []
    for k, (ph, _, t0, t1) in enumerate(segs):
        v = PHONE_TO_VISEME[ph]
        if v is None:                                   # HH takes the next shape
            nxt = "sil"
            for ph2, _, _, _ in segs[k + 1:]:
                v2 = PHONE_TO_VISEME[ph2]
                if v2 is not None:
                    nxt = v2[0] if isinstance(v2, tuple) else v2
                    break
            v = nxt
        if isinstance(v, tuple):
            mid = t0 + (t1 - t0) * 0.5
            vis.append((v[0], t0, mid))
            vis.append((v[1], mid, t1))
        else:
            vis.append((v, t0, t1))
    return vis


def frame_labels(segs, vsegs, n_frames):
    centers = (np.arange(n_frames) * HOP + WIN / 2) / SR
    vends = np.array([s[2] for s in vsegs])
    vidx = np.minimum(np.searchsorted(vends, centers, side="right"), len(vsegs) - 1)
    vis = np.array([VIS_INDEX[vsegs[i][0]] for i in vidx], dtype=np.uint8)
    pends = np.array([s[3] for s in segs])
    pidx = np.minimum(np.searchsorted(pends, centers, side="right"), len(segs) - 1)
    aux = {p: i for i, p in enumerate(AUX_PHONES)}
    ph = np.array([aux[segs[i][0]] for i in pidx], dtype=np.uint8)
    return vis, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-utts", type=int, default=6000)
    ap.add_argument("--minutes", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="render 6 sentences and check alignment")
    args = ap.parse_args()

    data = Path(args.data)
    if (data / "index.json").exists() and not args.smoke:
        print(f"Dataset already exists at {data} — skipping generation.")
        return

    # Kokoro may live in a separate interpreter (see colab_setup.py)
    try:
        import kokoro  # noqa: F401
    except ImportError:
        import os
        kpy = os.environ.get("KOKORO_PY")
        if kpy and os.path.realpath(kpy) != os.path.realpath(sys.executable) and os.path.exists(kpy):
            print(f"re-launching under {kpy} (Kokoro environment)", flush=True)
            os.execv(kpy, [kpy] + sys.argv)
        raise

    import torch
    from scipy.signal import resample_poly
    from kokoro import KModel, KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kokoro-82M on {device} …")
    model = KModel(repo_id=REPO).to(device).eval()
    pipes = {lc: KPipeline(lang_code=lc, repo_id=REPO, model=model) for lc in ("a", "b")}

    voices = []
    for v in VOICES:
        try:
            pipes[v[0]].load_voice(v)
            voices.append(v)
        except Exception as e:
            print(f"  ! voice {v} unavailable: {e}")
    print(f"  voices: {len(voices)} ({sum(v in VAL_VOICES for v in voices)} held out for validation)")

    sents = PEARL_LINES[:6] if args.smoke else load_corpus(args.max_utts, args.seed)
    rng = random.Random(args.seed)
    stats = {"phone": defaultdict(list), "pause": defaultdict(list), "gap": [],
             "unknown": Counter()}
    audio_chunks, vis_chunks, ph_chunks, index = [], [], [], []
    a_off = f_off = 0
    ratios = []
    t_start = time.time()
    (data / "samples").mkdir(parents=True, exist_ok=True)

    order = voices[:]
    for i, text in enumerate(sents):
        if i >= args.max_utts or (time.time() - t_start) > args.minutes * 60:
            break
        if i % len(order) == 0:
            rng.shuffle(order)
        voice = order[i % len(order)]
        speed = 1.0 if rng.random() < 0.25 else round(rng.uniform(0.8, 1.25), 3)
        try:
            results = list(pipes[voice[0]](text, voice=voice, speed=speed, split_pattern=None))
        except Exception as e:
            print(f"  ! synth failed ({voice}): {e}")
            continue
        for r in results:
            out = getattr(r, "output", None)
            audio = getattr(out, "audio", None) if out is not None else getattr(r, "audio", None)
            dur = getattr(out, "pred_dur", None) if out is not None else getattr(r, "pred_dur", None)
            if audio is None or dur is None:
                continue
            audio = audio.detach().float().cpu().numpy().reshape(-1)
            dur = dur.detach().cpu().numpy().reshape(-1).astype(np.float64)
            kept = [c for c in r.phonemes if model.vocab.get(c) is not None]
            if len(dur) != len(kept) + 2 or dur.sum() <= 0:
                print(f"  ! length mismatch: {len(dur)} durs vs {len(kept)} phonemes — skipped")
                continue
            ratio = len(audio) / dur.sum()          # samples per duration unit (expect 600)
            ratios.append(ratio)
            spu = ratio / 24000.0
            segs = misaki_to_segments(kept, dur, spu, speed, stats)
            vsegs = segments_to_visemes(segs)
            wav = resample_poly(audio, 2, 3).astype(np.float32)
            peak = float(np.abs(wav).max()) or 1.0
            wav = wav / peak * 0.7 if peak > 0.98 else wav
            n = num_frames(len(wav))
            if n < 20:
                continue
            vis, ph = frame_labels(segs, vsegs, n)
            pcm = np.clip(np.round(wav * 32767), -32768, 32767).astype(np.int16)
            audio_chunks.append(pcm)
            vis_chunks.append(vis)
            ph_chunks.append(ph)
            index.append({"a": a_off, "n": len(pcm), "f": f_off, "nf": n, "voice": voice,
                          "speed": speed, "text": r.graphemes if hasattr(r, "graphemes") else text,
                          "phonemes": r.phonemes, "val": voice in VAL_VOICES})
            a_off += len(pcm)
            f_off += n
            if len(index) <= 8 or args.smoke:
                import soundfile as sf
                sf.write(data / "samples" / f"{len(index):02d}_{voice}.wav", wav, SR)
            if args.smoke:
                print(f"  [{voice} x{speed}] {text}\n    {r.phonemes}\n    "
                      f"ratio={ratio:.1f} visemes={' '.join(v for v, _, _ in vsegs[:24])}")
        if len(index) % 250 == 0 and index:
            hrs = a_off / SR / 3600
            el = time.time() - t_start
            print(f"  {len(index):5d} utts | {hrs:5.2f} h audio | {el / 60:5.1f} min | "
                  f"{a_off / SR / max(el, 1e-6):5.1f}x realtime", flush=True)

    if not index:
        raise SystemExit("No audio generated — check the Kokoro install.")

    print(f"Samples-per-duration-unit: mean {np.mean(ratios):.2f} (Kokoro hop is 600 @24 kHz)")
    if stats["unknown"]:
        print("Unmapped misaki symbols (merged into neighbours):", dict(stats["unknown"].most_common(20)))
    if args.smoke:
        print("Smoke test OK.")
        return

    np.save(data / "audio.npy", np.concatenate(audio_chunks))
    np.save(data / "vis.npy", np.concatenate(vis_chunks))
    np.save(data / "ph.npy", np.concatenate(ph_chunks))
    save_json(data / "index.json", {"sr": SR, "hop": HOP, "win": WIN, "visemes": VISEMES,
                                    "aux_phones": AUX_PHONES, "utts": index})

    # learned durations for the text-driven timeline in the browser
    med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    phones = {k: {"median": round(med(v), 4), "mean": round(float(np.mean(v)), 4), "n": len(v)}
              for k, v in sorted(stats["phone"].items()) if len(v) >= 5}
    pauses = {k: round(med(v), 4) for k, v in stats["pause"].items() if len(v) >= 5}
    save_json(data / "durations.json", {
        "source": "Kokoro-82M predicted durations, normalised to speed 1.0",
        "phones": phones, "pauses": pauses, "word_gap": round(med(stats["gap"]) or 0.0, 4),
        "utterances": len(index), "hours": round(a_off / SR / 3600, 3)})
    vis_all = np.concatenate(vis_chunks)
    dist = {VISEMES[i]: round(float((vis_all == i).mean()), 4) for i in range(len(VISEMES))}
    print(f"Saved {len(index):,} utterances, {a_off / SR / 3600:.2f} h → {data}")
    print("Viseme distribution:", dist)


if __name__ == "__main__":
    main()
