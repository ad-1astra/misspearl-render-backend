"""
Miss Pearl lip-sync training — shared definitions.

Everything the browser runtime must agree on lives here: phone / viseme
inventories, the misaki(Kokoro) -> ARPAbet mapping, log-mel feature
extraction and the model architectures.  The exporters write these
constants into the model manifests so `pearl/*.js` never hard-codes them.
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Phones & visemes
# ─────────────────────────────────────────────────────────────

VOWELS = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"]
CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
              "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
BASE_PHONES = sorted(VOWELS + CONSONANTS)            # 39 ARPAbet phones
AUX_PHONES = ["SIL"] + BASE_PHONES                   # 40 classes for the aux head

# G2P / lexicon tokens: consonants + vowels with lexical stress 0/1/2
G2P_TOKENS = sorted(CONSONANTS + [v + s for v in VOWELS for s in "012"])  # 69
G2P_CODE_ALPHABET = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "0123456789!#$%&()*+-./:;<=>?@[]^_{}~")
assert len(G2P_TOKENS) == 69 and len(G2P_CODE_ALPHABET) >= 69

# Meta/Oculus 15-viseme standard (also used by Azure / Ready Player Me rigs)
VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR",
           "aa", "E", "ih", "oh", "ou"]
VIS_INDEX = {v: i for i, v in enumerate(VISEMES)}

# ARPAbet -> viseme. Diphthongs are (start, end) pairs: the first half of the
# segment uses the first viseme, the second half the second.
PHONE_TO_VISEME = {
    "AA": "aa", "AE": "aa", "AH": "aa", "AO": "oh", "AW": ("aa", "ou"),
    "AY": ("aa", "ih"), "EH": "E", "ER": "RR", "EY": ("E", "ih"), "IH": "ih",
    "IY": "ih", "OW": ("oh", "ou"), "OY": ("oh", "ih"), "UH": "ou", "UW": "ou",
    "B": "PP", "M": "PP", "P": "PP", "F": "FF", "V": "FF", "TH": "TH",
    "DH": "TH", "T": "DD", "D": "DD", "K": "kk", "G": "kk", "NG": "kk",
    "CH": "CH", "JH": "CH", "SH": "CH", "ZH": "CH", "S": "SS", "Z": "SS",
    "N": "nn", "L": "nn", "R": "RR", "W": "ou", "Y": "ih",
    "HH": None,          # aspiration: mouth already shaped for the next sound
    "SIL": "sil",
}

# misaki (Kokoro's G2P) symbols -> ARPAbet base phones
MISAKI_TO_ARPA = {
    "A": "EY", "I": "AY", "W": "AW", "Y": "OY", "O": "OW", "Q": "OW",
    "æ": "AE", "a": "AE", "ɑ": "AA", "ɒ": "AA", "ɐ": "AH", "ə": "AH",
    "ᵊ": "AH", "ʌ": "AH", "ᵻ": "IH", "ɪ": "IH", "i": "IY", "ɛ": "EH",
    "e": "EH", "ɜ": "ER", "ɚ": "ER", "ɝ": "ER", "ɔ": "AO", "ʊ": "UH",
    "u": "UW", "o": "OW",
    "b": "B", "d": "D", "f": "F", "h": "HH", "j": "Y", "k": "K", "l": "L",
    "m": "M", "n": "N", "p": "P", "s": "S", "t": "T", "v": "V", "w": "W",
    "z": "Z", "ɡ": "G", "g": "G", "ŋ": "NG", "ɹ": "R", "r": "R", "ʃ": "SH",
    "ʒ": "ZH", "ð": "DH", "θ": "TH", "ʤ": "JH", "ʧ": "CH", "ɾ": "T", "T": "T",  # misaki 'T' = US flap
    "ʔ": "T", "x": "HH", "ç": "HH", "ɬ": "L", "ʍ": "W", "ɫ": "L",
}
MISAKI_STRESS = {"ˈ": "1", "ˌ": "2"}
MISAKI_LENGTH = {"ː"}
MISAKI_PUNCT = set(";:,.!?—…\"()“”'«»-–")


# ─────────────────────────────────────────────────────────────
# Audio features (must match pearl/audio.js exactly)
# ─────────────────────────────────────────────────────────────

SR = 16000
WIN = 400          # 25 ms
HOP = 160          # 10 ms  -> 100 frames / s
N_FFT = 512
N_MELS = 64
FMIN, FMAX = 50.0, 8000.0
LOG_EPS = 1e-5
FPS = SR / HOP


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """HTK-style triangular filterbank, area-normalised (Slaney norm)."""
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    mpts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hpts = mel_to_hz(mpts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        lo, ce, hi = hpts[m], hpts[m + 1], hpts[m + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
        fb[m] *= 2.0 / (hi - lo)
    return fb.astype(np.float32)


def hann_window(n=WIN):
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * math.pi * k / n)).astype(np.float32)


def num_frames(n_samples: int) -> int:
    return 0 if n_samples < WIN else 1 + (n_samples - WIN) // HOP


def logmel_torch(wave, fb, window, lowpass_bins=None):
    """wave: [B, N] float tensor -> [B, n_mels, T] log-mel (un-normalised)."""
    import torch
    frames = wave.unfold(-1, WIN, HOP)                    # [B, T, WIN]
    frames = frames * window
    spec = torch.fft.rfft(frames, n=N_FFT)                # [B, T, 257]
    power = spec.real ** 2 + spec.imag ** 2
    if lowpass_bins is not None:                          # augmentation: band-limit
        power = power * lowpass_bins[:, None, :]
    mel = torch.matmul(power, fb.t())                     # [B, T, n_mels]
    return torch.log(mel + LOG_EPS).transpose(1, 2)


def logmel_numpy(wave: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Reference implementation used for JS test vectors. -> [n_mels, T]"""
    fb = mel_filterbank(n_mels=n_mels)
    win = hann_window()
    T = num_frames(len(wave))
    out = np.zeros((n_mels, T), dtype=np.float32)
    for t in range(T):
        seg = wave[t * HOP: t * HOP + WIN].astype(np.float64) * win
        spec = np.fft.rfft(seg, n=N_FFT)
        power = spec.real ** 2 + spec.imag ** 2
        out[:, t] = np.log(fb.astype(np.float64) @ power + LOG_EPS)
    return out


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

def build_audio_model(n_mels=N_MELS, channels=80, dilations=(1, 2, 4, 8, 1, 2, 4, 8),
                      n_vis=len(VISEMES), n_ph=len(AUX_PHONES), dropout=0.1, voice_head=False):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.bn1 = nn.BatchNorm1d(c)
            self.pw = nn.Conv1d(c, c, 1)
            self.bn2 = nn.BatchNorm1d(c)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            y = torch.relu(self.bn1(self.conv(x)))
            y = self.bn2(self.pw(self.drop(y)))
            return torch.relu(x + y)

    class Audio2Viseme(nn.Module):
        """Non-causal TCN: ±31 frames (≈310 ms) of context, all-conv so it
        runs on any length and is trivial to re-implement in JavaScript."""

        def __init__(self):
            super().__init__()
            self.cfg = dict(n_mels=n_mels, channels=channels, dilations=list(dilations),
                            n_vis=n_vis, n_ph=n_ph)
            if voice_head:
                self.cfg["voice_head"] = True
            self.inp = nn.Conv1d(n_mels, channels, 3, padding=1)
            self.bn0 = nn.BatchNorm1d(channels)
            self.blocks = nn.ModuleList([Block(channels, d) for d in dilations])
            self.vis = nn.Conv1d(channels, n_vis, 1)
            self.ph = nn.Conv1d(channels, n_ph, 1)
            # singing model: "is a voice singing/speaking right now?" (vs. only music)
            self.voc = nn.Conv1d(channels, 1, 1) if voice_head else None

        def forward(self, x):            # x: [B, n_mels, T] (normalised)
            x = torch.relu(self.bn0(self.inp(x)))
            for b in self.blocks:
                x = b(x)
            if self.voc is not None:
                return self.vis(x), self.ph(x), self.voc(x)[:, 0]
            return self.vis(x), self.ph(x)

    return Audio2Viseme()


def build_g2p_model(n_chars, n_out=len(G2P_TOKENS) + 1, d=128, layers=4, heads=4,
                    ff=512, max_len=32, up=3, dropout=0.1):
    import torch
    import torch.nn as nn

    class G2P(nn.Module):
        """Non-autoregressive CTC G2P: every letter is repeated `up` times,
        a Transformer encoder labels each slot, CTC collapses to phones."""

        def __init__(self):
            super().__init__()
            self.cfg = dict(n_chars=n_chars, n_out=n_out, d=d, layers=layers, heads=heads,
                            ff=ff, max_len=max_len, up=up)
            self.char = nn.Embedding(n_chars, d, padding_idx=0)
            self.pos = nn.Embedding(max_len * up, d)
            self.sub = nn.Embedding(up, d)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
            self.out = nn.Linear(d, n_out)

        def forward(self, chars, lengths):   # chars [B, L] (0 = pad)
            B, L = chars.shape
            x = self.char(chars).repeat_interleave(up, dim=1)          # [B, L*up, d]
            pos = torch.arange(L * up, device=chars.device)
            x = x + self.pos(pos)[None] + self.sub(pos % up)[None]
            mask = pos[None, :] >= (lengths[:, None] * up)            # True = pad
            x = self.enc(x, src_key_padding_mask=mask)
            return self.out(x)                                         # [B, L*up, n_out]

    return G2P()


# ─────────────────────────────────────────────────────────────
# Emotion student
# ─────────────────────────────────────────────────────────────

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMO_DIM = 48
EMO_HIDDEN = 64
EMO_OOV_BUCKETS = 1024
EMO_BIGRAM_BUCKETS = 8192

import re as _re
_EMO_TOKEN = _re.compile(r"[a-z0-9']+|[!?]")


def emo_tokens(text: str):
    return _EMO_TOKEN.findall(text.lower().replace("’", "'"))


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def emo_feature_ids(tokens, vocab: dict):
    V = len(vocab)
    ids = []
    for t in tokens:
        ids.append(vocab[t] if t in vocab else V + fnv1a(t) % EMO_OOV_BUCKETS)
    base = V + EMO_OOV_BUCKETS
    for a, b in zip(tokens, tokens[1:]):
        ids.append(base + fnv1a(a + " " + b) % EMO_BIGRAM_BUCKETS)
    return ids


# ─────────────────────────────────────────────────────────────
# Export helpers: one JSON manifest + one little-endian .bin per model
# ─────────────────────────────────────────────────────────────

class BinWriter:
    def __init__(self):
        self.buf = bytearray()
        self.tensors = {}

    def _align(self):
        while len(self.buf) % 4:
            self.buf.append(0)

    def f32(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
        self._align()
        self.tensors[name] = {"dtype": "f32", "shape": list(a.shape), "offset": len(self.buf)}
        self.buf += a.tobytes()

    def i8_rows(self, name, arr):
        """Symmetric int8 quantisation with one scale per row."""
        a = np.asarray(arr, dtype=np.float32)
        scale = np.maximum(np.abs(a).max(axis=1), 1e-8) / 127.0
        q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8)
        self._align()
        off_s = len(self.buf)
        self.buf += scale.astype("<f4").tobytes()
        self._align()
        off_q = len(self.buf)
        self.buf += q.tobytes()
        self.tensors[name] = {"dtype": "i8rows", "shape": list(a.shape),
                              "scale_offset": off_s, "offset": off_q}

    def save(self, out_dir, stem, config: dict):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.bin").write_bytes(bytes(self.buf))
        manifest = {"format": "pearl-nn-1", "bin": f"{stem}.bin", "config": config,
                    "tensors": self.tensors}
        (out_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=1))
        return out_dir / f"{stem}.json"


def env_info():
    import platform
    info = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
