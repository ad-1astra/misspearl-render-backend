"""
Singing data for Miss Pearl's song lip-sync.

  python gen_song_data.py sung  --data DIR        # Kokoro sings lyric-like lines (exact viseme labels)
  python gen_song_data.py music --out DIR         # real songs → vocals / instrumental stems (Demucs)

`sung` writes the same format as gen_audio_data.py (audio.npy, vis.npy, ph.npy,
index.json): lines from public-domain poetry, everyday sentences and song
vocables ("oh", "yeah", "la la") are *sung* by all Kokoro voices with a melody,
held vowels and vibrato (see sing_synth.py). Voices in VAL_VOICES are held out.

`music` takes Creative-Commons songs from the Free Music Archive (fma_small),
splits every clip with Demucs (htdemucs) into the singing voice and the
accompaniment (drums + bass + other), and stores mixture / accompaniment /
vocals at 16 kHz. Training mixes the accompaniment ("the beat") with sung
lines, and learns from the real mixtures where the voice is. Clips are split
train/val by artist. Any audio files in --test-dir are separated the same way
and kept apart as test songs (never trained on).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import HOP, SR, VISEMES, num_frames, save_json  # noqa: E402

VOCABLES = [
    "oh oh oh", "yeah yeah yeah", "la la la la", "na na na na", "ooh baby", "oh yeah", "hey hey hey",
    "whoa oh oh", "mm mm mm", "oh oh, yeah", "ooh ooh", "la di da", "do do do do", "ah ah ah",
    "hey yeah", "oh my love", "whoa whoa", "yeah, oh yeah", "uh huh, uh huh", "baby, baby",
]
_ok = re.compile(r"^[A-Za-z ,.'!?;:\-]+$")


def _relaunch_in_kokoro_env():
    try:
        import kokoro  # noqa: F401
    except ImportError:
        kpy = os.environ.get("KOKORO_PY")
        if kpy and os.path.realpath(kpy) != os.path.realpath(sys.executable) and os.path.exists(kpy):
            print(f"re-launching under {kpy} (Kokoro environment)", flush=True)
            os.execv(kpy, [kpy] + sys.argv)
        raise


def lyric_lines(n_target: int, seed: int):
    """Short, singable lines: public-domain poetry (CC0), everyday sentences, vocables."""
    from datasets import load_dataset
    rng = random.Random(seed)
    poetry, modern = [], []
    try:
        ds = load_dataset("biglam/gutenberg-poetry-corpus", split="train")
        for i in rng.sample(range(len(ds)), min(len(ds), n_target * 6)):
            s = ds[i]["line"].strip().strip("_").replace("’", "'")
            w = s.split()
            if 4 <= len(w) <= 11 and _ok.match(s) and not s.isupper() and sum(c.isupper() for c in s) <= 3:
                poetry.append(s)
        print(f"  poetry lines: {len(poetry):,}")
    except Exception as e:  # pragma: no cover
        print("  ! poetry corpus unavailable:", e)
    try:
        from gen_audio_data import load_corpus
        for s in load_corpus(n_target, seed):
            for part in re.split(r"(?<=[,;:])\s+", s):
                if 3 <= len(part.split()) <= 11:
                    modern.append(part)
        print(f"  everyday phrases: {len(modern):,}")
    except Exception as e:  # pragma: no cover
        print("  ! everyday phrases unavailable:", e)
    rng.shuffle(poetry)
    rng.shuffle(modern)
    out = []
    while len(out) < n_target and (poetry or modern):
        r = rng.random()
        if r < 0.12:
            out.append(rng.choice(VOCABLES))
        elif r < 0.55 and poetry:
            out.append(poetry.pop())
        elif modern:
            out.append(modern.pop())
        elif poetry:
            out.append(poetry.pop())
        if rng.random() < 0.12 and out:                         # chorus-style repeat / ad-lib
            out[-1] = f"{out[-1]}, {rng.choice(VOCABLES)}" if rng.random() < 0.5 else f"{out[-1]}, {out[-1]}"
    return out


def cmd_sung(args):
    _relaunch_in_kokoro_env()
    import torch
    from scipy.signal import resample_poly
    from kokoro import KModel, KPipeline

    from gen_audio_data import REPO, VAL_VOICES, VOICES, frame_labels, misaki_to_segments, segments_to_visemes
    from sing_synth import sing

    data = Path(args.data)
    if (data / "index.json").exists() and not args.smoke:
        print(f"Sung corpus already exists at {data} — skipping.")
        return
    (data / "samples").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kokoro-82M on {device} …")
    model = KModel(repo_id=REPO).to(device).eval()
    pipes = {lc: KPipeline(lang_code=lc, repo_id=REPO, model=model) for lc in ("a", "b")}
    packs = {}
    for v in VOICES:
        try:
            packs[v] = pipes[v[0]].load_voice(v).to(device)
        except Exception as e:
            print(f"  ! voice {v} unavailable: {e}")
    voices = list(packs)
    lines = VOCABLES[:3] + ["I walked along the river where the morning light was gold"] if args.smoke \
        else lyric_lines(args.max_lines, args.seed)
    rng = np.random.default_rng(args.seed)
    stats = {"phone": defaultdict(list), "pause": defaultdict(list), "gap": [], "unknown": Counter()}
    audio_chunks, vis_chunks, ph_chunks, index = [], [], [], []
    a_off = f_off = 0
    t0 = time.time()
    order = voices[:]
    for i, text in enumerate(lines):
        if (time.time() - t0) > args.minutes * 60:
            print("time budget reached")
            break
        if i % len(order) == 0:
            random.Random(args.seed + i).shuffle(order)
        voice = order[i % len(order)]
        try:
            ps, _ = pipes[voice[0]].g2p(text)
            if not ps or len(ps) > 400:
                continue
            speed = float(rng.uniform(0.85, 1.15))
            audio, dur, kept, info = sing(model, ps, packs[voice], rng, speed=speed)
        except Exception as e:
            print(f"  ! sing failed ({voice}): {e}")
            continue
        if len(dur) != len(kept) + 2:
            continue
        spu = len(audio) / dur.sum() / 24000.0
        segs = misaki_to_segments(kept, dur, spu, 1.0, stats)
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
        index.append({"a": a_off, "n": len(pcm), "f": f_off, "nf": n, "voice": voice, "speed": round(speed, 3),
                      "text": text, "phonemes": ps, "syllables": info["syllables"], "base_hz": info["base_hz"],
                      "kind": "sung", "val": voice in VAL_VOICES})
        a_off += len(pcm)
        f_off += n
        if len(index) <= 10 or args.smoke:
            import soundfile as sf
            sf.write(data / "samples" / f"{len(index):02d}_{voice}.wav", wav, SR)
        if len(index) % 250 == 0:
            el = time.time() - t0
            print(f"  {len(index):5d} sung lines | {a_off / SR / 3600:5.2f} h | {el / 60:5.1f} min | "
                  f"{a_off / SR / max(el, 1e-6):5.1f}x realtime", flush=True)
    if not index:
        raise SystemExit("No sung audio generated.")
    np.save(data / "audio.npy", np.concatenate(audio_chunks))
    np.save(data / "vis.npy", np.concatenate(vis_chunks))
    np.save(data / "ph.npy", np.concatenate(ph_chunks))
    from common import AUX_PHONES, WIN
    save_json(data / "index.json", {"sr": SR, "hop": HOP, "win": WIN, "visemes": VISEMES, "aux_phones": AUX_PHONES,
                                    "kind": "sung", "utts": index})
    vis_all = np.concatenate(vis_chunks)
    print(f"Saved {len(index):,} sung lines, {a_off / SR / 3600:.2f} h → {data}")
    print("Viseme distribution:", {VISEMES[i]: round(float((vis_all == i).mean()), 4) for i in range(len(VISEMES))})


# ─────────────────────────────────────────────────────────────
# music: Free Music Archive → Demucs stems
# ─────────────────────────────────────────────────────────────

def ffmpeg_decode(data: bytes | str, sr=44100, ch=2):
    src = data if isinstance(data, str) else "pipe:0"
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "pipe:1"],
                       input=None if isinstance(data, str) else data, capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:200])
    return np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, ch).T.copy()


class Separator:
    def __init__(self, name="htdemucs"):
        import torch
        from demucs.pretrained import get_model
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(name).to(self.dev).eval()
        self.vi = self.model.sources.index("vocals")
        print(f"Demucs {name} on {self.dev}: sources {self.model.sources}")

    def __call__(self, wav44):
        """wav44 [2, N] float → (vocals, accompaniment) as mono 16 kHz float32."""
        from demucs.apply import apply_model
        from scipy.signal import resample_poly
        torch = self.torch
        x = torch.tensor(wav44, device=self.dev)
        ref = x.mean(0)
        m, s = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(self.model, ((x - m) / s)[None], device=self.dev, shifts=0, split=True,
                              overlap=0.25, progress=False)[0] * s + m
        voc = out[self.vi].mean(0)
        acc = (out.sum(0) - out[self.vi]).mean(0)
        to16 = lambda a: resample_poly(a.float().cpu().numpy(), 160, 441).astype(np.float32)  # noqa: E731
        return to16(voc), to16(acc)


def cmd_music(args):
    # Demucs checkpoints are full pickles from Meta's own server; torch ≥2.6 loads weights-only by default
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from scipy.signal import resample_poly      # runs in the Demucs python (see colab_setup_singing.py)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sep = None
    t0 = time.time()

    def pack(rows, stem):
        mix = np.concatenate([r.pop("mix") for r in rows])
        acc = np.concatenate([r.pop("acc") for r in rows])
        voc = np.concatenate([r.pop("voc") for r in rows])
        np.save(out / f"{stem}_mix.npy", mix)
        np.save(out / f"{stem}_acc.npy", acc)
        np.save(out / f"{stem}_voc.npy", voc)
        return rows

    def separate_clip(wav44, meta):
        nonlocal sep
        if sep is None:
            sep = Separator(args.model)
        voc, acc = sep(wav44)
        mix = resample_poly(wav44.mean(0), 160, 441).astype(np.float32)
        n = min(len(mix), len(voc), len(acc))
        peak = max(float(np.abs(mix[:n]).max()), 1e-4)
        g = 0.89 / peak
        q = lambda a: np.clip(np.round(a[:n] * g * 32767), -32768, 32767).astype(np.int16)  # noqa: E731
        return {**meta, "n": n, "mix": q(mix), "acc": q(acc), "voc": q(voc)}

    # test songs (never trained on)
    tdir = Path(args.test_dir) if args.test_dir else None
    if tdir and tdir.exists() and not (out / "test_index.json").exists():
        rows, off = [], 0
        for f in sorted(tdir.iterdir()):
            if f.suffix.lower() not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"):
                continue
            try:
                r = separate_clip(ffmpeg_decode(str(f)), {"name": f.name})
            except Exception as e:
                print("  ! test song failed", f.name, e)
                continue
            r["a"] = off
            off += r["n"]
            rows.append(r)
            print(f"  test song separated: {f.name} ({r['n'] / SR:.1f} s)")
        if rows:
            save_json(out / "test_index.json", {"sr": SR, "songs": pack(rows, "test")})

    if (out / "index.json").exists():
        print(f"Music stems already exist at {out} — skipping FMA.")
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows, off, n_ok, n_bad = [], 0, 0, 0
    for k in [int(s) for s in args.shards.split(",")]:
        fn = hf_hub_download("benjamin-paine/free-music-archive-small", f"data/train-{k:05d}-of-00015.parquet",
                             repo_type="dataset")
        tab = pq.read_table(fn, columns=["audio", "artist", "title", "genres"])
        print(f"  shard {k}: {tab.num_rows} clips ({time.time() - t0:.0f} s)", flush=True)
        for i in range(tab.num_rows):
            if args.max_clips and n_ok >= args.max_clips:
                break
            a = tab.column("audio")[i].as_py()
            artist = str(tab.column("artist")[i].as_py() or "")
            try:
                wav44 = ffmpeg_decode(a["bytes"])
                if wav44.shape[1] < 44100 * 8:
                    continue
                r = separate_clip(wav44, {"artist": artist, "title": str(tab.column("title")[i].as_py() or ""),
                                          "genres": tab.column("genres")[i].as_py() or []})
            except Exception as e:
                n_bad += 1
                if n_bad < 5:
                    print("  ! clip failed:", e)
                continue
            r["a"] = off
            r["val"] = int(hashlib.md5(artist.encode()).hexdigest(), 16) % 10 == 0
            off += r["n"]
            rows.append(r)
            n_ok += 1
            if n_ok % 100 == 0:
                el = time.time() - t0
                print(f"  {n_ok:5d} clips separated | {off / SR / 3600:5.2f} h | {el / 60:5.1f} min", flush=True)
        if args.max_clips and n_ok >= args.max_clips:
            break
    if not rows:
        raise SystemExit("No music clips separated.")
    rows = pack(rows, "fma")
    save_json(out / "index.json", {"sr": SR, "source": "Free Music Archive (fma_small, CC licences) via "
                                   "benjamin-paine/free-music-archive-small", "separator": args.model, "clips": rows})
    nv = sum(r["val"] for r in rows)
    print(f"Saved {len(rows)} clips ({off / SR / 3600:.2f} h; {nv} val clips by artist) → {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sung")
    s.add_argument("--data", required=True)
    s.add_argument("--max-lines", type=int, default=9000)
    s.add_argument("--minutes", type=float, default=18.0)
    s.add_argument("--seed", type=int, default=21)
    s.add_argument("--smoke", action="store_true")
    m = sub.add_parser("music")
    m.add_argument("--out", required=True)
    m.add_argument("--shards", default="0,1")
    m.add_argument("--max-clips", type=int, default=0)
    m.add_argument("--model", default="htdemucs")
    m.add_argument("--test-dir", default="")
    args = ap.parse_args()
    {"sung": cmd_sung, "music": cmd_music}[args.cmd](args)


if __name__ == "__main__":
    main()
