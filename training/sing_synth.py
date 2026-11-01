"""
Make Kokoro-82M *sing* with exact phoneme timing.

Kokoro (StyleTTS2 + iSTFTNet) predicts a duration for every phoneme and a pitch
(F0) curve, and its decoder renders audio from the two. Both can be replaced:

  * durations — vowels are stretched into held notes (phrase-final notes most),
    consonants stay short, exactly as singers do;
  * pitch    — the predicted speech intonation is replaced by a melody: one note
    per syllable on a musical scale, with portamento between notes and vibrato
    that blooms on long notes.

Because we choose the durations, every rendered line comes with its true
phoneme → time alignment, i.e. frame-exact viseme labels for *sung* audio.
"""
from __future__ import annotations

import math

import numpy as np

from common import MISAKI_TO_ARPA, VOWELS

VOWEL_SYMS = {s for s, a in MISAKI_TO_ARPA.items() if a in VOWELS}
PAUSE_SYMS = set(",.!?;:—…")
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "penta": [0, 2, 4, 7, 9],
    "minpenta": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
}
F0_HOP_S = 300 / 24000          # Kokoro's F0 curve runs at 2× the duration frame rate (12.5 ms)


def _degree_to_semitone(scale, deg):
    n = len(scale)
    octave, k = divmod(deg, n)
    return 12 * octave + scale[k]


def plan_song_line(kept, rng, style=None):
    """Per-token duration factors + one melody note (semitones) per syllable."""
    style = style or {}
    n = len(kept)
    fac = np.ones(n + 2)
    is_v = np.array([k in VOWEL_SYMS for k in kept])
    vidx = np.flatnonzero(is_v)
    hold = style.get("hold", rng.uniform(0.8, 1.6))            # how "legato" this singer is
    for j, i in enumerate(vidx):
        if rng.random() < 0.12:
            f = rng.uniform(0.9, 1.2)                           # quick passing syllable
        else:
            f = math.exp(rng.uniform(math.log(1.15), math.log(3.2))) * hold
        nxt = kept[i + 1:i + 3]
        phrase_end = j == len(vidx) - 1 or any(c in PAUSE_SYMS for c in nxt)
        if phrase_end:
            f = rng.uniform(2.5, 6.5) * max(hold, 1.0)          # held note at the end of a phrase
        fac[i + 1] = f
        if i + 1 < n and kept[i + 1] == "ː":
            fac[i + 2] = f
    for i, k in enumerate(kept):
        if not is_v[i] and k != "ː":
            fac[i + 1] = rng.uniform(0.6, 1.3) if k in PAUSE_SYMS or k == " " else rng.uniform(0.8, 1.25)

    scale = SCALES[style.get("scale") or rng.choice(list(SCALES))]
    deg = int(rng.choice([0, 2, 4]))
    notes = []
    for _ in range(len(vidx)):
        notes.append(_degree_to_semitone(scale, deg))
        deg = int(np.clip(deg + rng.choice([-2, -1, -1, 0, 0, 1, 1, 1, 2, 3]), -3, 2 * len(scale)))
    # token → note: consonants take the note of the next vowel (onset), trailing ones the last
    tok_note = np.zeros(n + 2)
    if len(vidx):
        nv = 0
        for i in range(n):
            while nv < len(vidx) - 1 and vidx[nv] < i:
                nv += 1
            tok_note[i + 1] = notes[nv] if i <= vidx[-1] else notes[-1]
        tok_note[0], tok_note[-1] = notes[0], notes[-1]
    return fac, tok_note, len(vidx)


def melody_f0(f0_pred, tok_of_frame, tok_note, rng, style=None):
    """Replace speech F0 with the melody (Hz), keeping the voiced/unvoiced pattern."""
    style = style or {}
    f0 = f0_pred.copy()
    voiced = f0 > 55
    base = float(np.median(f0[voiced])) if voiced.any() else 200.0
    base_semi = 12 * math.log2(base / 440.0) + style.get("transpose", rng.uniform(-2.0, 4.0))
    target = base_semi + tok_note[tok_of_frame]
    # portamento: one-pole glide between notes
    tau = rng.uniform(0.02, 0.07)
    a = 1 - math.exp(-F0_HOP_S / tau)
    sm = np.empty_like(target)
    sm[0] = target[0]
    for t in range(1, len(target)):
        sm[t] = sm[t - 1] + a * (target[t] - sm[t - 1])
    # vibrato that grows on held notes
    depth = rng.uniform(0.0, 0.6)          # semitones (0–60 cents)
    rate = rng.uniform(4.6, 6.6)
    since = np.zeros(len(target))
    for t in range(1, len(target)):
        since[t] = since[t - 1] + F0_HOP_S if target[t] == target[t - 1] else 0.0
    ramp = np.clip((since - 0.18) / 0.35, 0, 1)
    tt = np.arange(len(target)) * F0_HOP_S
    sm = sm + depth * ramp * np.sin(2 * math.pi * rate * tt + rng.uniform(0, 6.28))
    # slight random drift / jitter like a real voice
    sm = sm + 0.08 * np.cumsum(rng.standard_normal(len(sm))) / math.sqrt(len(sm) + 1)
    hz = 440.0 * 2 ** (sm / 12.0)
    f0[voiced] = hz[voiced]
    return f0, base


def sing(model, ps, pack, rng, speed=1.0, style=None):
    """Render phoneme string `ps` as a sung line.
    Returns (audio float32 @24 kHz, pred_dur incl. BOS/EOS, kept symbols, info)."""
    import torch

    dev = model.device
    kept = [p for p in ps if model.vocab.get(p) is not None]
    ids = [model.vocab[p] for p in kept]
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(dev)
    ref_s = pack[len(ps) - 1].to(dev)
    with torch.no_grad():
        L = input_ids.shape[-1]
        input_lengths = torch.full((1,), L, device=dev, dtype=torch.long)
        text_mask = torch.arange(L, device=dev)[None] + 1 > input_lengths[:, None]
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)[0] / speed
        fac, tok_note, n_syl = plan_song_line(kept, rng, style)
        dur = duration.detach().cpu().numpy() * fac
        dur = np.maximum(1, np.round(dur)).astype(np.int64)
        if dur.sum() > 1400:                         # keep renders ≤ ~35 s
            dur = np.maximum(1, np.round(dur * 1400 / dur.sum())).astype(np.int64)
        pred_dur = torch.tensor(dur, device=dev)
        indices = torch.repeat_interleave(torch.arange(L, device=dev), pred_dur)
        aln = torch.zeros((L, indices.shape[0]), device=dev)
        aln[indices, torch.arange(indices.shape[0], device=dev)] = 1
        aln = aln[None]
        en = d.transpose(-1, -2) @ aln
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        f0 = F0_pred[0].detach().cpu().numpy()
        tok_of_frame = np.repeat(np.arange(L), dur * 2)[: len(f0)]
        if len(tok_of_frame) < len(f0):
            tok_of_frame = np.pad(tok_of_frame, (0, len(f0) - len(tok_of_frame)), mode="edge")
        f0_new, base = melody_f0(f0, tok_of_frame, tok_note, rng, style)
        F0 = torch.tensor(f0_new, dtype=F0_pred.dtype, device=dev)[None]
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln
        audio = model.decoder(asr, F0, N_pred, ref_s[:, :128]).squeeze()
    info = {"syllables": n_syl, "base_hz": round(base, 1), "f0_target": f0_new, "f0_speech": f0}
    return audio.detach().float().cpu().numpy().reshape(-1), dur.astype(np.float64), kept, info
