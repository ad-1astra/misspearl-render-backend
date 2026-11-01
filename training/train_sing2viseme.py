"""
Step 4 — train the singing lip-sync network ("sing2viseme").

Input: 80-band log-mel at 100 fps of a *song* (voice + music).
Outputs per 10 ms frame:
  • 15 viseme probabilities — the mouth shape of the singing voice;
  • voice activity — is a person singing/speaking now, or is it only the beat?
  • (aux) 40 phones, to make the trunk learn finer phonetic detail.

Training data (built on the fly on the GPU):
  synthetic songs  sung lines (Kokoro singing, exact labels) and speech
                   placed over real instrumental stems ("the beat"),
                   vocal reverb, random voice-to-music ratio −8…+12 dB,
                   gaps where only the music plays, MP3/phone band-limits.
  real songs       Free Music Archive mixtures; where the voice is comes from
                   the Demucs vocal stem, and the mouth shapes are distilled
                   from the speech model run on that isolated voice.
Validation: voices the model never heard × songs by artists it never heard.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (AUX_PHONES, FMAX, FMIN, HOP, LOG_EPS, N_FFT, SR, VISEMES, WIN,  # noqa: E402
                    BinWriter, build_audio_model, env_info, hann_window, logmel_numpy, logmel_torch,
                    mel_filterbank, num_frames, save_json)
from train_audio2viseme import fold, np_conv1d, spec_augment  # noqa: E402

N_MELS = 80
DILATIONS = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
SIL = 0


# ─────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────

def load_corpus(d):
    d = Path(d)
    idx = json.loads((d / "index.json").read_text())
    c = {"audio": np.load(d / "audio.npy", mmap_mode="r"), "vis": np.load(d / "vis.npy"),
         "ph": np.load(d / "ph.npy"), "utts": idx["utts"]}
    c["train"] = [u for u in c["utts"] if not u["val"]]
    c["val"] = [u for u in c["utts"] if u["val"]]
    for k in ("train", "val"):
        L = np.array([u["nf"] for u in c[k]], dtype=np.float64)
        c[k + "_p"] = L / L.sum() if len(L) else None
    return c


def load_music(d, test=False):
    d = Path(d)
    stem, name, key = ("test", "test_index.json", "songs") if test else ("fma", "index.json", "clips")
    if not (d / name).exists():
        return None
    idx = json.loads((d / name).read_text())
    return {k: np.load(d / f"{stem}_{k}.npy", mmap_mode="r") for k in ("mix", "acc", "voc")} | {"clips": idx[key]}


def frame_db(x):
    """int16/float signal → per-frame RMS in dBFS at 100 fps (same framing as the features)."""
    x = np.asarray(x, dtype=np.float32) / (32768.0 if np.asarray(x).dtype == np.int16 else 1.0)
    n = num_frames(len(x))
    if n <= 0:
        return np.zeros(0, np.float32)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    s = np.arange(n) * HOP
    e = (c[s + WIN] - c[s]) / WIN
    return (10 * np.log10(e + 1e-10)).astype(np.float32)


def _runs(mask):
    """(start, end, value) runs of a boolean array."""
    if len(mask) == 0:
        return []
    ch = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    st = np.concatenate([[0], ch])
    en = np.concatenate([ch, [len(mask)]])
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(st, en)]


def vad_from_stems(voc, mix):
    """Where is a voice singing? From the Demucs vocal stem vs. the full mix."""
    vdb, mdb = frame_db(voc), frame_db(mix)
    n = min(len(vdb), len(mdb))
    act = (vdb[:n] > -42) & (vdb[:n] > mdb[:n] - 16)
    for a, b, v in _runs(act):                       # fill short holes (consonants, breaths)
        if not v and b - a < 18 and a > 0 and b < n:
            act[a:b] = True
    for a, b, v in _runs(act):                       # drop blips (bleed, clicks)
        if v and b - a < 10:
            act[a:b] = False
    return act


class Teacher:
    """The trained speech model (models/audio2viseme.*), run in torch with its folded weights."""

    def __init__(self, manifest, device):
        import torch
        man = json.loads(Path(manifest).read_text())
        buf = (Path(manifest).parent / man["bin"]).read_bytes()
        W = {k: torch.tensor(np.frombuffer(buf, "<f4", int(np.prod(t["shape"])), t["offset"]).reshape(t["shape"]).copy(),
                             device=device) for k, t in man["tensors"].items()}
        self.W, self.cfg, self.torch = W, man["config"], torch
        self.window = torch.tensor(hann_window(), device=device)

    def logits(self, wave):                          # wave [B, N] float tensor
        torch, W = self.torch, self.W
        F = torch.nn.functional
        f = logmel_torch(wave, W["mel_fb"], self.window)
        f = (f - W["feat_mean"][None, :, None]) / W["feat_std"][None, :, None]
        x = torch.relu(F.conv1d(f, W["inp.w"], W["inp.b"], padding=1))
        for i, d in enumerate(self.cfg["dilations"]):
            y = torch.relu(F.conv1d(x, W[f"b{i}.conv.w"], W[f"b{i}.conv.b"], padding=d, dilation=d))
            y = F.conv1d(y, W[f"b{i}.pw.w"][:, :, None], W[f"b{i}.pw.b"])
            x = torch.relu(x + y)
        return F.conv1d(x, W["vis.w"][:, :, None], W["vis.b"])


def precompute_real(music, teacher, device, chunk_s=30):
    """Per clip: voice-activity labels (uint8) and teacher viseme probabilities on the vocal stem (fp16)."""
    import torch
    vad, soft = [], []
    for c in music["clips"]:
        a, n = c["a"], c["n"]
        v = vad_from_stems(music["voc"][a:a + n], music["mix"][a:a + n])
        x = torch.tensor(np.asarray(music["voc"][a:a + n], dtype=np.float32) / 32768.0, device=device)[None]
        with torch.no_grad():
            p = torch.softmax(teacher.logits(x), 1)[0].T.half().cpu().numpy()      # [T, 15]
        T = min(len(v), len(p))
        p = p[:T].copy()
        p[~v[:T]] = 0
        p[~v[:T], SIL] = 1
        vad.append(v[:T].astype(np.uint8))
        soft.append(p)
    return vad, soft


class Batcher:
    def __init__(self, sung, speech, music, real_vad, real_soft, crop, split, seed):
        self.sung, self.speech, self.music = sung, speech, music
        self.crop, self.split = crop, split
        self.rng = np.random.default_rng(seed)
        self.n_samp = (crop - 1) * HOP + WIN
        clips = music["clips"] if music else []
        self.mus_ids = [i for i, c in enumerate(clips) if (c["val"] if split == "val" else not c["val"])
                        and c["n"] > self.n_samp + 10]
        L = np.array([clips[i]["n"] for i in self.mus_ids], dtype=np.float64)
        self.mus_p = L / L.sum() if len(L) else None
        self.real_vad, self.real_soft = real_vad, real_soft

    def _utt(self, src):
        sp = self.split if src[self.split] else "train"
        k = self.rng.choice(len(src[sp]), p=src[sp + "_p"])
        return src[sp][k]

    def synth(self, p_voice=0.9, p_music=0.9, p_sung=0.65):
        """One synthetic song crop → (vocal, background, vis, ph, vad)."""
        rng, crop = self.rng, self.crop
        voc = np.zeros(self.n_samp, np.float32)
        vis = np.zeros(crop, np.int64)
        ph = np.zeros(crop, np.int64)
        if rng.random() < p_voice:
            t, k = int(rng.integers(-crop // 3, crop // 3)), 0
            while t < crop - 20 and k < 4:
                src = self.sung if (self.sung and (rng.random() < p_sung or not self.speech)) else self.speech
                u = self._utt(src)
                s0, f0 = max(0, -t), max(t, 0)
                m = min(u["nf"] - s0, crop - f0)
                if m > 10:
                    a0 = u["a"] + s0 * HOP
                    take = min((m - 1) * HOP + WIN, u["n"] - s0 * HOP, self.n_samp - f0 * HOP)
                    g = 10 ** (rng.uniform(-3, 3) / 20)
                    voc[f0 * HOP:f0 * HOP + take] += np.asarray(src["audio"][a0:a0 + take], np.float32) / 32768.0 * g
                    vis[f0:f0 + m] = src["vis"][u["f"] + s0:u["f"] + s0 + m]
                    ph[f0:f0 + m] = src["ph"][u["f"] + s0:u["f"] + s0 + m]
                t = f0 + max(m, 0) + int(rng.integers(12, 140))
                k += 1
        bg = np.zeros(self.n_samp, np.float32)
        if self.mus_ids and rng.random() < p_music:
            i = self.mus_ids[rng.choice(len(self.mus_ids), p=self.mus_p)]
            c = self.music["clips"][i]
            s = int(rng.integers(0, c["n"] - self.n_samp))
            bg = np.asarray(self.music["acc"][c["a"] + s:c["a"] + s + self.n_samp], np.float32) / 32768.0
        return voc, bg, vis, ph, (ph != 0).astype(np.float32)

    def real(self):
        """One real song crop → (mixture, vad, teacher probs [15, T])."""
        rng = self.rng
        i = self.mus_ids[rng.choice(len(self.mus_ids), p=self.mus_p)]
        c = self.music["clips"][i]
        T = len(self.real_vad[i])
        s = int(rng.integers(0, max(1, T - self.crop)))
        x = np.zeros(self.n_samp, np.float32)
        seg = np.asarray(self.music["mix"][c["a"] + s * HOP:c["a"] + s * HOP + self.n_samp], np.float32) / 32768.0
        x[:len(seg)] = seg
        vad = np.zeros(self.crop, np.float32)
        soft = np.zeros((self.crop, len(VISEMES)), np.float32)
        soft[:, SIL] = 1
        m = min(self.crop, T - s)
        vad[:m] = self.real_vad[i][s:s + m]
        soft[:m] = self.real_soft[i][s:s + m]
        return x, vad, soft.T


def reverb(torch, x, p, device, rt=(0.15, 1.1), wet=(0.15, 0.55)):
    B, N = x.shape
    on = torch.rand(B, device=device) < p
    if not on.any():
        return x
    L = int(0.6 * SR)
    rt60 = torch.empty(B, 1, device=device).uniform_(*rt)
    t = torch.arange(L, device=device)[None].float() / SR
    ir = torch.randn(B, L, device=device) * torch.exp(-6.9 * t / rt60)
    ir[:, :int(0.01 * SR)] *= 0.2                       # pre-delay
    ir[:, 0] = 1.0
    ir = ir / ir.norm(dim=1, keepdim=True)
    nfft = 1 << (N + L).bit_length()
    w = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(ir, nfft), nfft)[:, :N]
    w = w * (x.norm(dim=1, keepdim=True) / (w.norm(dim=1, keepdim=True) + 1e-8))
    mix = torch.empty(B, 1, device=device).uniform_(*wet)
    return torch.where(on[:, None], (1 - mix) * x + mix * w, x)


def finish_aug(torch, x, device, p_noise=0.25, p_lp=0.35):
    """Gain / clipping / noise / band-limit (MP3, phone speaker) on the final mix."""
    B, N = x.shape
    nz = torch.rand(B, device=device) < p_noise
    if nz.any():
        snr = torch.empty(B, 1, device=device).uniform_(20, 50)
        noise = torch.randn_like(x)
        sp = x.pow(2).mean(1, keepdim=True) + 1e-8
        x = torch.where(nz[:, None], x + noise * torch.sqrt(sp / 10 ** (snr / 10)), x)
    peak = x.abs().amax(1, keepdim=True) + 1e-6
    g = 10 ** (torch.empty(B, 1, device=device).uniform_(-20, 4) / 20) * (0.9 / peak)
    x = torch.clamp(x * g, -1, 1)
    nb = N_FFT // 2 + 1
    freqs = torch.linspace(0, SR / 2, nb, device=device)[None]
    lp = torch.rand(B, device=device) < p_lp
    cut = torch.empty(B, 1, device=device).uniform_(3500, 7800)
    mask = torch.where(lp[:, None], torch.sigmoid((cut - freqs) / 150.0), torch.ones_like(freqs))
    return x, mask


def mix_voice_music(torch, voc, bg, device, ratio_db):
    """Scale the voice to `ratio_db` above the music (RMS over the voiced samples)."""
    act = (voc.abs() > 1e-4).float()
    pv = (voc.pow(2) * act).sum(1, keepdim=True) / act.sum(1, keepdim=True).clamp(min=1)
    pb = bg.pow(2).mean(1, keepdim=True)
    has_v = act.sum(1, keepdim=True) > 100
    has_b = pb > 1e-7
    sv = torch.sqrt(pb * 10 ** (ratio_db / 10) / pv.clamp(min=1e-9))
    sv = torch.where(has_v & has_b, sv, torch.ones_like(sv))
    return voc * sv + bg


# ─────────────────────────────────────────────────────────────
# training
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", required=True, help="speech corpus dir (gen_audio_data.py)")
    ap.add_argument("--sung", required=True, help="sung corpus dir (gen_song_data.py sung)")
    ap.add_argument("--music", required=True, help="music stems dir (gen_song_data.py music)")
    ap.add_argument("--teacher", required=True, help="models/audio2viseme.json")
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=600)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Environment:", env_info(), flush=True)
    export, ckpt = Path(args.export), Path(args.ckpt)
    export.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    speech = load_corpus(args.speech) if (Path(args.speech) / "index.json").exists() else None
    sung = load_corpus(args.sung)
    music = load_music(args.music)
    test = load_music(args.music, test=True)
    if music is None:
        raise SystemExit("music stems missing — run gen_song_data.py music first")
    print(f"sung lines {len(sung['train']):,} train / {len(sung['val']):,} val | speech utts "
          f"{len(speech['train']) if speech else 0:,} | music clips {sum(not c['val'] for c in music['clips'])} train / "
          f"{sum(c['val'] for c in music['clips'])} val | test songs {len(test['clips']) if test else 0}", flush=True)

    teacher = Teacher(args.teacher, device)
    t0 = time.time()
    real_vad, real_soft = precompute_real(music, teacher, device)
    vad_rate = float(np.mean(np.concatenate(real_vad))) if real_vad else 0
    print(f"real-song labels ready in {time.time() - t0:.0f} s — voice present in {vad_rate:.0%} of song frames", flush=True)
    test_vad, test_soft = precompute_real(test, teacher, device) if test else ([], [])

    tr = Batcher(sung, speech, music, real_vad, real_soft, args.crop, "train", 1)
    fb = torch.tensor(mel_filterbank(n_mels=N_MELS), device=device)
    window = torch.tensor(hann_window(), device=device)

    def feats_raw(x, mask=None):
        return logmel_torch(x, fb, window, mask)

    # normalisation stats from typical training mixtures
    with torch.no_grad():
        items = [tr.synth() for _ in range(48)]
        voc = torch.tensor(np.stack([i[0] for i in items]), device=device)
        bg = torch.tensor(np.stack([i[1] for i in items]), device=device)
        x = mix_voice_music(torch, voc, bg, device, torch.full((48, 1), 4.0, device=device))
        x = x * (0.5 / (x.abs().amax(1, keepdim=True) + 1e-6))
        f = feats_raw(x)
        mean, std = f.mean(dim=(0, 2)), f.std(dim=(0, 2)) + 1e-3

    def features(x, mask=None):
        return (feats_raw(x, mask) - mean[None, :, None]) / std[None, :, None]

    vis_counts = np.bincount(sung["vis"], minlength=len(VISEMES)).astype(np.float64) + 1
    w_vis = (vis_counts.sum() / vis_counts) ** 0.3
    w_vis = torch.tensor(w_vis / w_vis.mean(), dtype=torch.float32, device=device)

    model = build_audio_model(n_mels=N_MELS, channels=args.channels, dilations=DILATIONS, voice_head=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sing2Viseme params: {n_params:,} | receptive field ±{1 + sum(DILATIONS)} frames", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # ── fixed validation sets ─────────────────────────────────
    va = Batcher(sung, speech, music, real_vad, real_soft, args.crop, "val", 99)
    n_val = 12 if args.smoke else 96
    val_items = [va.synth(p_voice=0.92, p_music=1.0, p_sung=0.8) for _ in range(n_val)]
    CONDS = {"a cappella": None, "voice +6 dB": 6.0, "voice 0 dB": 0.0, "voice -6 dB": -6.0}

    def val_mix(cond):
        voc = torch.tensor(np.stack([i[0] for i in val_items]), device=device)
        bg = torch.tensor(np.stack([i[1] for i in val_items]), device=device)
        if CONDS[cond] is None:
            x = voc
        else:
            x = mix_voice_music(torch, voc, bg, device, torch.full((len(val_items), 1), CONDS[cond], device=device))
        return x * (0.5 / (x.abs().amax(1, keepdim=True) + 1e-6))

    yv_val = np.stack([i[2] for i in val_items])
    vad_val = np.stack([i[4] for i in val_items]) > 0.5
    has_music = np.array([np.abs(i[1]).max() > 1e-4 for i in val_items])

    def scores(pred, pvoc, yv, vad, music_mask):
        """pred [B,T] viseme ids, pvoc [B,T] voice prob, yv labels, vad bool."""
        T = min(pred.shape[1], yv.shape[1])
        pred, yv, vad = pred[:, :T], yv[:, :T], vad[:, :T]
        pvoc = pvoc[:, :T] if pvoc is not None else None
        gated = np.where(pvoc >= 0.5, pred, SIL) if pvoc is not None else pred
        exact = float((gated == yv).mean())
        tol = np.zeros_like(yv, dtype=bool)
        for s in range(-2, 3):
            tol |= np.roll(gated, s, axis=1) == yv
        out = {"exact": round(exact, 4), "tol20ms": round(float(tol.mean()), 4),
               "voiced_exact": round(float((gated == yv)[vad].mean()), 4) if vad.any() else None}
        beat = (~vad) & music_mask[:, None]
        if beat.any():
            out["mouth_still_on_beat"] = round(float((gated == SIL)[beat].mean()), 4)
        if pvoc is not None:
            pv = pvoc >= 0.5
            tp, fp, fn = (pv & vad).sum(), (pv & ~vad).sum(), (~pv & vad).sum()
            out["voice_f1"] = round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4)
        return out

    @torch.no_grad()
    def run_model(x):
        model.eval()
        lv, _, lvo = model(features(x))
        model.train()
        return lv.argmax(1).cpu().numpy(), torch.sigmoid(lvo).cpu().numpy()

    @torch.no_grad()
    def evaluate_synth(with_teacher=False):
        res = {}
        for cond in CONDS:
            x = val_mix(cond)
            pred, pv = run_model(x)
            mm = has_music if CONDS[cond] is not None else np.zeros_like(has_music)
            res[cond] = scores(pred, pv, yv_val, vad_val, mm)
            if with_teacher:
                tp = teacher.logits(x).argmax(1).cpu().numpy()
                res[cond]["speech_model_before"] = scores(tp, None, yv_val, vad_val, mm)
        return res

    @torch.no_grad()
    def evaluate_real(mus, vads, softs, which, limit=None):
        """Real songs: voice detection vs. the Demucs vocal stem; mouth shapes vs. the teacher on that stem."""
        if not mus:
            return None
        ids = [i for i, c in enumerate(mus["clips"]) if which == "all" or c.get("val")][:limit]
        tp = fp = fn = 0
        agree = n_agree = 0
        onset = []
        still = n_still = 0
        for i in ids:
            c = mus["clips"][i]
            x = torch.tensor(np.asarray(mus["mix"][c["a"]:c["a"] + c["n"]], np.float32) / 32768.0, device=device)[None]
            pred, pv = run_model(x)
            pred, pv = pred[0], pv[0]
            v = vads[i].astype(bool)
            T = min(len(v), len(pv))
            pvb, v, pred = pv[:T] >= 0.5, v[:T], pred[:T]
            tp += int((pvb & v).sum()); fp += int((pvb & ~v).sum()); fn += int((~pvb & v).sum())
            tgt = softs[i][:T].argmax(1)
            gated = np.where(pvb, pred, SIL)
            agree += int((gated == tgt)[v].sum()); n_agree += int(v.sum())
            still += int((gated == SIL)[~v].sum()); n_still += int((~v).sum())
            # first sustained voice (≥0.3 s) — when does the singing start?
            def first_on(m):
                for a, b, val in _runs(m):
                    if val and b - a >= 30:
                        return a / 100.0
                return None
            ft, fp_ = first_on(v), first_on(pvb)
            if ft is not None and fp_ is not None:
                onset.append(abs(ft - fp_))
        return {"clips": len(ids), "voice_f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                "voice_precision": round(tp / max(tp + fp, 1), 4), "voice_recall": round(tp / max(tp + fn, 1), 4),
                "mouth_agrees_with_isolated_voice": round(agree / max(n_agree, 1), 4),
                "mouth_still_when_no_voice": round(still / max(n_still, 1), 4),
                "singing_start_error_s_median": round(float(np.median(onset)), 3) if onset else None}

    # ── loop ──────────────────────────────────────────────────
    total_s = (0.4 if args.smoke else args.minutes) * 60
    ck_path = ckpt / "sing2viseme_best.pt"
    best, history, step = -1.0, [], 0
    skip = False
    if ck_path.exists() and not args.smoke:
        prev = torch.load(ck_path, map_location="cpu", weights_only=False)
        if prev.get("done") and prev.get("cfg") == model.cfg:
            print(f"Finished checkpoint found (step {prev['step']}) — skipping training, exporting it.")
            skip = True
    B = 4 if args.smoke else args.batch
    B_real = max(1, B // 4)
    B_syn = B - B_real
    t_start = time.time()
    eval_every_s = 5 if args.smoke else 180          # evaluate every ~3 minutes of training
    last_eval = time.time()
    while not skip:
        frac = (time.time() - t_start) / total_s
        if frac >= 1:
            break
        lr = args.lr * (min(1.0, frac / 0.05) if frac < 0.05 else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95)))
        for g in opt.param_groups:
            g["lr"] = lr
        step += 1
        syn = [tr.synth() for _ in range(B_syn)]
        rea = [tr.real() for _ in range(B_real)]
        voc = torch.tensor(np.stack([i[0] for i in syn]), device=device)
        bg = torch.tensor(np.stack([i[1] for i in syn]), device=device)
        voc = reverb(torch, voc, 0.5, device)
        ratio = torch.where(torch.rand(B_syn, 1, device=device) < 0.15,
                            torch.empty(B_syn, 1, device=device).uniform_(-8, -3),
                            torch.empty(B_syn, 1, device=device).uniform_(-3, 12))
        xs = mix_voice_music(torch, voc, bg, device, ratio)
        xr = torch.tensor(np.stack([i[0] for i in rea]), device=device)
        x, mask = finish_aug(torch, torch.cat([xs, xr]), device)
        f = spec_augment(torch, features(x, mask))
        lv, lp, lvo = model(f)
        yv = torch.tensor(np.stack([i[2] for i in syn]), device=device)
        yp = torch.tensor(np.stack([i[3] for i in syn]), device=device)
        vad = torch.tensor(np.stack([i[4] for i in syn] + [i[1] for i in rea]), device=device)
        soft = torch.tensor(np.stack([i[2] for i in rea]), device=device)
        loss_v = F.cross_entropy(lv[:B_syn], yv, weight=w_vis, label_smoothing=0.05)
        loss_p = F.cross_entropy(lp[:B_syn], yp, label_smoothing=0.05)
        loss_voc = F.binary_cross_entropy_with_logits(lvo, vad)
        loss_kd = -(soft * F.log_softmax(lv[B_syn:], 1)).sum(1).mean()
        loss = loss_v + 0.3 * loss_p + 1.0 * loss_voc + 0.3 * loss_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if time.time() - last_eval >= eval_every_s or (time.time() - t_start) >= total_s:
            last_eval = time.time()
            ev = evaluate_synth()
            rv = evaluate_real(music, real_vad, real_soft, "val", limit=40)
            score = 0.5 * (ev["voice 0 dB"]["tol20ms"] + ev["voice +6 dB"]["tol20ms"]) + \
                0.5 * ev["voice 0 dB"]["voice_f1"] + 0.5 * (rv["voice_f1"] if rv else 0)
            history.append({"step": step, "min": round((time.time() - t_start) / 60, 1), "loss": round(float(loss), 4),
                            "lr": round(lr, 6), "score": round(score, 4),
                            **{c: ev[c]["tol20ms"] for c in ev}, "voice_f1_0db": ev["voice 0 dB"]["voice_f1"],
                            "real_voice_f1": rv["voice_f1"] if rv else None})
            print(f"step {step:5d} | {(time.time() - t_start) / 60:4.1f} min | loss {float(loss):.3f} "
                  f"(vis {float(loss_v):.2f} voice {float(loss_voc):.2f} kd {float(loss_kd):.2f}) | ±20ms: "
                  + " · ".join(f"{c} {ev[c]['tol20ms']:.3f}" for c in ev)
                  + f" | voice F1 0dB {ev['voice 0 dB']['voice_f1']:.3f}, real songs {rv['voice_f1'] if rv else 0:.3f}",
                  flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "cfg": model.cfg,
                            "step": step, "history": history, "done": False}, ck_path)
    if not skip:
        st = torch.load(ck_path, map_location="cpu", weights_only=False)
        st["done"] = True
        st["history"] = history
        torch.save(st, ck_path)

    state = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    mean, std = state["mean"].to(device), state["std"].to(device)
    history = state.get("history", history)
    final = evaluate_synth(with_teacher=True)
    real_val = evaluate_real(music, real_vad, real_soft, "val")
    test_songs = evaluate_real(test, test_vad, test_soft, "all") if test else None
    print("FINAL (best step %d)" % state["step"])
    for c, r in final.items():
        b = r.pop("speech_model_before")
        r["speech_model_before"] = b
        print(f"  {c:12s} new: exact {r['exact']:.3f} ±20ms {r['tol20ms']:.3f} voice-F1 {r.get('voice_f1', 0):.3f} "
              f"still-on-beat {r.get('mouth_still_on_beat', float('nan')):.3f} | speech model before: exact {b['exact']:.3f} "
              f"±20ms {b['tol20ms']:.3f} still-on-beat {b.get('mouth_still_on_beat', float('nan')):.3f}")
    print("  real songs (unseen artists):", real_val)
    print("  test songs:", test_songs)

    # ── export (BN folded) ────────────────────────────────────
    model = model.cpu().eval()
    W = {}
    W["inp.w"], W["inp.b"] = fold(model.inp, model.bn0)
    for i, blk in enumerate(model.blocks):
        W[f"b{i}.conv.w"], W[f"b{i}.conv.b"] = fold(blk.conv, blk.bn1)
        pw_w, pw_b = fold(blk.pw, blk.bn2)
        W[f"b{i}.pw.w"], W[f"b{i}.pw.b"] = pw_w[:, :, 0], pw_b
    for h in ("vis", "ph", "voc"):
        W[f"{h}.w"] = getattr(model, h).weight.detach().numpy()[:, :, 0]
        W[f"{h}.b"] = getattr(model, h).bias.detach().numpy()
    mean_np, std_np = state["mean"].cpu().numpy(), state["std"].cpu().numpy()
    bw = BinWriter()
    bw.f32("mel_fb", mel_filterbank(n_mels=N_MELS))
    bw.f32("feat_mean", mean_np)
    bw.f32("feat_std", std_np)
    for k, v in W.items():
        bw.f32(k, v)
    cfg = {"sr": SR, "win": WIN, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
           "log_eps": LOG_EPS, "window": "hann-periodic", "channels": model.cfg["channels"],
           "dilations": model.cfg["dilations"], "visemes": VISEMES, "aux_phones": AUX_PHONES, "voice_head": True,
           "params": n_params, "kind": "sing2viseme",
           "metrics": {"synthetic_songs_unseen_voices_and_artists": final, "real_songs_unseen_artists": real_val,
                       "test_songs": test_songs},
           "best_step": state["step"], "env": env_info()}
    bw.save(export, "sing2viseme", cfg)

    # verify the exported maths & write a JS test vector
    item = val_items[0]
    xm = mix_voice_music(torch, torch.tensor(item[0])[None], torch.tensor(item[1])[None], "cpu",
                         torch.full((1, 1), 3.0)).numpy()[0]
    xm = xm * (0.5 / (np.abs(xm).max() + 1e-6))
    clip = np.clip(np.round(xm[:SR * 3 // 2] * 32767), -32768, 32767).astype(np.int16)
    fe = logmel_numpy(clip.astype(np.float32) / 32768.0, n_mels=N_MELS)
    fn = (fe - mean_np[:, None]) / std_np[:, None]
    x = np.maximum(0, np_conv1d(fn, W["inp.w"], W["inp.b"], 1))
    for i, d in enumerate(DILATIONS):
        y = np.maximum(0, np_conv1d(x, W[f"b{i}.conv.w"], W[f"b{i}.conv.b"], d))
        y = W[f"b{i}.pw.w"].astype(np.float64) @ y + W[f"b{i}.pw.b"][:, None]
        x = np.maximum(0, x + y)
    ref_vis = W["vis.w"].astype(np.float64) @ x + W["vis.b"][:, None]
    ref_voc = (W["voc.w"].astype(np.float64) @ x + W["voc.b"][:, None])[0]
    with torch.no_grad():
        tv, _, tvo = model(torch.tensor(fn[None], dtype=torch.float32))
    diff = max(float(np.abs(ref_vis - tv[0].numpy()).max()), float(np.abs(ref_voc - tvo[0].numpy()).max()))
    print(f"export check: numpy(folded) vs torch max |Δlogit| = {diff:.2e}")
    assert diff < 1e-2, "BN folding / export mismatch"
    save_json(export / "tests" / "sing2viseme_test.json", {
        "pcm16": clip.astype(int).tolist(), "feats_norm_first5": fn[:, :5].round(5).tolist(),
        "vis_logits": ref_vis.round(4).tolist(), "voc_logits": ref_voc.round(4).tolist()})
    save_json(export / "metrics" / "sing2viseme.json", {
        "synthetic_songs_unseen_voices_and_artists": final, "real_songs_unseen_artists": real_val,
        "test_songs": test_songs, "history": history, "params": n_params,
        "sung_hours_train": round(sum(u["n"] for u in sung["train"]) / SR / 3600, 2),
        "speech_hours_train": round(sum(u["n"] for u in speech["train"]) / SR / 3600, 2) if speech else 0,
        "music_hours": round(sum(c["n"] for c in music["clips"]) / SR / 3600, 2),
        "real_voice_fraction": round(vad_rate, 3)})
    print("done →", export)


if __name__ == "__main__":
    main()
