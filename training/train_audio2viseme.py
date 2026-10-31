"""
Step 2 — train the audio → viseme network.

Input: 64-band log-mel at 100 fps.  Output: per-frame probabilities over
the 15 Meta/Oculus visemes (+ an auxiliary 40-way phone head that makes
the trunk learn finer phonetic detail).  Trained with heavy acoustic
augmentation so it works on TTS audio, MP3s and live microphones alike.
Validation uses voices the model never heard during training.
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
from common import (AUX_PHONES, FMAX, FMIN, HOP, LOG_EPS, N_FFT, N_MELS, SR, VISEMES,  # noqa: E402
                    WIN, BinWriter, build_audio_model, env_info, hann_window, logmel_numpy,
                    logmel_torch, mel_filterbank, save_json)


def load_data(data: Path):
    idx = json.loads((data / "index.json").read_text())
    audio = np.load(data / "audio.npy", mmap_mode="r")
    vis = np.load(data / "vis.npy")
    ph = np.load(data / "ph.npy")
    utts = idx["utts"]
    return audio, vis, ph, utts


def make_batch(rng, utts, audio, vis, ph, B, crop):
    n_samp = (crop - 1) * HOP + WIN
    xs = np.zeros((B, n_samp), dtype=np.float32)
    yv = np.zeros((B, crop), dtype=np.int64)          # pad = sil(0)
    yp = np.zeros((B, crop), dtype=np.int64)          # pad = SIL(0)
    lens = np.array([u["nf"] for u in utts], dtype=np.float64)
    p = lens / lens.sum()
    for b, k in enumerate(rng.choice(len(utts), size=B, p=p)):
        u = utts[k]
        nf = u["nf"]
        if nf > crop:
            s = int(rng.integers(0, nf - crop + 1))
            m = crop
        else:
            s, m = 0, nf
        a0 = u["a"] + s * HOP
        take = min((m - 1) * HOP + WIN, u["n"] - s * HOP)
        xs[b, :take] = audio[a0:a0 + take] / 32768.0
        yv[b, :m] = vis[u["f"] + s:u["f"] + s + m]
        yp[b, :m] = ph[u["f"] + s:u["f"] + s + m]
    return xs, yv, yp


def augment(torch, x, rng_t, device):
    """x [B, N] clean waveform → corrupted waveform + per-item low-pass masks."""
    B, N = x.shape
    # reverb (synthetic exponentially decaying impulse response)
    rv = torch.rand(B, device=device) < 0.25
    if rv.any():
        L = int(0.35 * SR)
        rt60 = torch.empty(B, 1, device=device).uniform_(0.12, 0.6)
        t = torch.arange(L, device=device)[None].float() / SR
        ir = torch.randn(B, L, device=device) * torch.exp(-6.9 * t / rt60)
        ir[:, 0] = 1.0
        ir = ir / ir.norm(dim=1, keepdim=True)
        nfft = 1 << (N + L).bit_length()
        wet = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(ir, nfft), nfft)[:, :N]
        mix = torch.empty(B, 1, device=device).uniform_(0.3, 0.9)
        x = torch.where(rv[:, None], (1 - mix) * x + mix * wet * (x.norm(dim=1, keepdim=True) /
                                                                  (wet.norm(dim=1, keepdim=True) + 1e-8)), x)
    # additive noise at random SNR (white or low-passed "room" noise)
    nz = torch.rand(B, device=device) < 0.6
    if nz.any():
        snr = torch.empty(B, 1, device=device).uniform_(5, 40)
        noise = torch.randn_like(x)
        brown = torch.rand(B, 1, device=device) < 0.4
        smooth = torch.cumsum(noise, dim=1)
        smooth = smooth - torch.nn.functional.avg_pool1d(smooth[:, None], 801, 1, 400,
                                                         count_include_pad=False)[:, 0]
        noise = torch.where(brown, smooth, noise)
        sp = x.pow(2).mean(1, keepdim=True) + 1e-8
        npow = noise.pow(2).mean(1, keepdim=True) + 1e-8
        noise = noise * torch.sqrt(sp / npow / (10 ** (snr / 10)))
        x = torch.where(nz[:, None], x + noise, x)
    # gain (−24…+6 dB) with occasional clipping
    g = 10 ** (torch.empty(B, 1, device=device).uniform_(-24, 6) / 20)
    x = torch.clamp(x * g, -1, 1)
    # band-limiting (telephone / MP3 / cheap mic)
    nb = N_FFT // 2 + 1
    freqs = torch.linspace(0, SR / 2, nb, device=device)[None]
    lp = torch.rand(B, device=device) < 0.3
    cut = torch.empty(B, 1, device=device).uniform_(3000, 7500)
    mask = torch.where(lp[:, None], torch.sigmoid((cut - freqs) / 150.0), torch.ones_like(freqs))
    return x, mask


def spec_augment(torch, f):
    B, M, T = f.shape
    for b in range(B):
        if np.random.rand() < 0.5:
            for _ in range(np.random.randint(1, 3)):
                w = np.random.randint(1, 6)
                s = np.random.randint(0, max(1, T - w))
                f[b, :, s:s + w] = 0
            w = np.random.randint(1, 9)
            s = np.random.randint(0, M - w)
            f[b, s:s + w, :] = 0
    return f


def fold(conv, bn):
    w = conv.weight.detach().cpu().double().numpy()
    b = conv.bias.detach().cpu().double().numpy()
    s = bn.weight.detach().cpu().double().numpy() / np.sqrt(bn.running_var.detach().cpu().double().numpy() + bn.eps)
    w2 = w * s[:, None, None]
    b2 = (b - bn.running_mean.detach().cpu().double().numpy()) * s + bn.bias.detach().cpu().double().numpy()
    return w2.astype(np.float32), b2.astype(np.float32)


def np_conv1d(x, w, b, d):
    """x [Cin, T], w [Cout, Cin, K] 'same' zero padding, dilation d."""
    K = w.shape[2]
    pad = d * (K - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    T = x.shape[1]
    y = np.zeros((w.shape[0], T), dtype=np.float64)
    for k in range(K):
        y += w[:, :, k].astype(np.float64) @ xp[:, k * d:k * d + T]
    return y + b[:, None]


def np_forward(W, feats, dilations):
    """Reference forward pass with BN folded — mirrors pearl/nn.js."""
    x = np.maximum(0, np_conv1d(feats, W["inp.w"], W["inp.b"], 1))
    for i, d in enumerate(dilations):
        y = np.maximum(0, np_conv1d(x, W[f"b{i}.conv.w"], W[f"b{i}.conv.b"], d))
        y = W[f"b{i}.pw.w"].astype(np.float64) @ y + W[f"b{i}.pw.b"][:, None]
        x = np.maximum(0, x + y)
    vis = W["vis.w"].astype(np.float64) @ x + W["vis.b"][:, None]
    ph = W["ph.w"].astype(np.float64) @ x + W["ph.b"][:, None]
    return vis, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--minutes", type=float, default=18.0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--channels", type=int, default=80)
    ap.add_argument("--smoke", action="store_true", help="random data, 30 steps, CPU ok")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Environment:", env_info())
    export, ckpt = Path(args.export), Path(args.ckpt)
    export.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        n_utt = 40
        nf = rng.integers(150, 500, size=n_utt)
        utts, a_off, f_off = [], 0, 0
        for i, n in enumerate(nf):
            ns = (n - 1) * HOP + WIN
            utts.append({"a": a_off, "n": int(ns), "f": f_off, "nf": int(n), "val": i < 6, "voice": "x"})
            a_off += ns
            f_off += n
        audio = (rng.standard_normal(a_off) * 3000).astype(np.int16)
        vis = rng.integers(0, len(VISEMES), f_off).astype(np.uint8)
        ph = rng.integers(0, len(AUX_PHONES), f_off).astype(np.uint8)
        args.steps, args.batch = 30, 8
    else:
        audio, vis, ph, utts = load_data(Path(args.data))
    train = [u for u in utts if not u["val"]]
    val = [u for u in utts if u["val"]]
    print(f"train utts {len(train):,} | val utts {len(val):,} (unseen voices)")

    fb = torch.tensor(mel_filterbank(), device=device)
    window = torch.tensor(hann_window(), device=device)

    # global normalisation stats from clean training audio
    sample = make_batch(rng, train, audio, vis, ph, 64, args.crop)[0]
    with torch.no_grad():
        f = logmel_torch(torch.tensor(sample, device=device), fb, window)
        mean = f.mean(dim=(0, 2))
        std = f.std(dim=(0, 2)) + 1e-3
    print("feature mean range", float(mean.min()), float(mean.max()))

    vis_counts = np.bincount(vis[: min(len(vis), 3_000_000)], minlength=len(VISEMES)).astype(np.float64) + 1
    w_vis = (vis_counts.sum() / vis_counts) ** 0.35
    w_vis = torch.tensor(w_vis / w_vis.mean(), dtype=torch.float32, device=device)

    model = build_audio_model(channels=args.channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Audio2Viseme params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.steps, pct_start=0.08)

    def features(x, mask=None):
        f = logmel_torch(x, fb, window, mask)
        return (f - mean[None, :, None]) / std[None, :, None]

    @torch.no_grad()
    def evaluate(noisy=False, limit=400):
        model.eval()
        conf = np.zeros((len(VISEMES), len(VISEMES)), dtype=np.int64)
        ph_ok = ph_n = tol_ok = 0
        g = torch.Generator(device=device).manual_seed(123)
        for u in val[:limit]:
            x = torch.tensor(audio[u["a"]:u["a"] + u["n"]] / 32768.0, dtype=torch.float32,
                             device=device)[None]
            if noisy:
                sp = x.pow(2).mean()
                x = x + torch.randn(x.shape, generator=g, device=device) * torch.sqrt(sp / 10 ** 1.5)
            lv, lp = model(features(x))
            pv = lv.argmax(1)[0].cpu().numpy()[: u["nf"]]
            pp = lp.argmax(1)[0].cpu().numpy()[: u["nf"]]
            yv_ = vis[u["f"]:u["f"] + u["nf"]][: len(pv)]
            yp_ = ph[u["f"]:u["f"] + u["nf"]][: len(pp)]
            np.add.at(conf, (yv_, pv), 1)
            ph_ok += int((pp == yp_).sum())
            ph_n += len(yp_)
            # ±2 frame (20 ms) boundary tolerance
            for t in range(len(pv)):
                lo, hi = max(0, t - 2), min(len(yv_), t + 3)
                tol_ok += int(pv[t] in yv_[lo:hi])
        model.train()
        acc = np.trace(conf) / max(conf.sum(), 1)
        prec = np.diag(conf) / np.maximum(conf.sum(0), 1)
        rec = np.diag(conf) / np.maximum(conf.sum(1), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        return {"viseme_acc": round(float(acc), 4), "viseme_macro_f1": round(float(f1.mean()), 4),
                "viseme_acc_tol20ms": round(tol_ok / max(conf.sum(), 1), 4),
                "phone_acc": round(ph_ok / max(ph_n, 1), 4),
                "per_viseme_f1": {VISEMES[i]: round(float(f1[i]), 3) for i in range(len(VISEMES))},
                "confusion": conf.tolist()}

    t0 = time.time()
    best, best_step, history = -1.0, 0, []
    model.train()
    for step in range(1, args.steps + 1):
        xs, yv, yp = make_batch(rng, train, audio, vis, ph, args.batch, args.crop)
        x = torch.tensor(xs, device=device)
        x, mask = augment(torch, x, None, device)
        f = spec_augment(torch, features(x, mask))
        lv, lp = model(f)
        yv_t = torch.tensor(yv, device=device)
        yp_t = torch.tensor(yp, device=device)
        loss = F.cross_entropy(lv, yv_t, weight=w_vis, label_smoothing=0.05) + \
            0.3 * F.cross_entropy(lp, yp_t, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        out_of_time = (time.time() - t0) > args.minutes * 60
        if step % 250 == 0 or step == args.steps or out_of_time:
            m = evaluate(limit=120 if not args.smoke else 6)
            history.append({"step": step, "loss": round(float(loss), 4), **{k: m[k] for k in
                            ("viseme_acc", "viseme_acc_tol20ms", "phone_acc", "viseme_macro_f1")}})
            print(f"step {step:5d} | loss {float(loss):.3f} | val viseme acc {m['viseme_acc']:.3f} "
                  f"(±20ms {m['viseme_acc_tol20ms']:.3f}) | phone acc {m['phone_acc']:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if m["viseme_acc"] > best:
                best, best_step = m["viseme_acc"], step
                torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                            "cfg": model.cfg, "step": step}, ckpt / "audio2viseme_best.pt")
        if out_of_time:
            print("time budget reached")
            break

    state = torch.load(ckpt / "audio2viseme_best.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    final = evaluate(limit=len(val))
    noisy = evaluate(noisy=True, limit=min(len(val), 300))
    print(f"FINAL (best step {best_step}) — unseen voices: viseme acc {final['viseme_acc']:.3f}, "
          f"±20ms {final['viseme_acc_tol20ms']:.3f}, macro-F1 {final['viseme_macro_f1']:.3f}, "
          f"phone acc {final['phone_acc']:.3f} | at 15 dB SNR: {noisy['viseme_acc']:.3f}")

    # ── export (BN folded) ────────────────────────────────────
    model = model.cpu().eval()
    W = {}
    W["inp.w"], W["inp.b"] = fold(model.inp, model.bn0)
    for i, blk in enumerate(model.blocks):
        W[f"b{i}.conv.w"], W[f"b{i}.conv.b"] = fold(blk.conv, blk.bn1)
        pw_w, pw_b = fold(blk.pw, blk.bn2)
        W[f"b{i}.pw.w"], W[f"b{i}.pw.b"] = pw_w[:, :, 0], pw_b
    W["vis.w"] = model.vis.weight.detach().numpy()[:, :, 0]
    W["vis.b"] = model.vis.bias.detach().numpy()
    W["ph.w"] = model.ph.weight.detach().numpy()[:, :, 0]
    W["ph.b"] = model.ph.bias.detach().numpy()

    bw = BinWriter()
    bw.f32("mel_fb", mel_filterbank())
    bw.f32("feat_mean", state["mean"].numpy())
    bw.f32("feat_std", state["std"].numpy())
    for k, v in W.items():
        bw.f32(k, v)
    cfg = {"sr": SR, "win": WIN, "hop": HOP, "n_fft": N_FFT, "n_mels": N_MELS, "fmin": FMIN,
           "fmax": FMAX, "log_eps": LOG_EPS, "window": "hann-periodic",
           "channels": model.cfg["channels"], "dilations": model.cfg["dilations"],
           "visemes": VISEMES, "aux_phones": AUX_PHONES, "params": n_params,
           "metrics": {k: v for k, v in final.items() if k != "confusion"},
           "metrics_noisy_15db": {k: noisy[k] for k in ("viseme_acc", "viseme_acc_tol20ms")},
           "best_step": best_step, "env": env_info()}
    bw.save(export, "audio2viseme", cfg)

    # ── verify the exported maths & write JS test vectors ─────
    u = val[0]
    clip = np.asarray(audio[u["a"]:u["a"] + min(u["n"], SR * 3 // 2)])
    feats = logmel_numpy(clip.astype(np.float32) / 32768.0)
    feats_n = (feats - state["mean"].numpy()[:, None]) / state["std"].numpy()[:, None]
    ref_vis, _ = np_forward(W, feats_n, model.cfg["dilations"])
    with torch.no_grad():
        t_vis, _ = model(torch.tensor(feats_n[None], dtype=torch.float32))
    diff = float(np.abs(ref_vis - t_vis[0].numpy()).max())
    print(f"export check: numpy(folded) vs torch max |Δlogit| = {diff:.2e}")
    assert diff < 1e-2, "BN folding / export mismatch"
    save_json(export / "tests" / "audio2viseme_test.json", {
        "pcm16": clip.astype(int).tolist(), "feats_norm_first5": feats_n[:, :5].round(5).tolist(),
        "vis_logits": ref_vis.round(4).tolist(), "argmax": ref_vis.argmax(0).tolist()})

    save_json(export / "metrics" / "audio2viseme.json", {
        "final_unseen_voices": final, "noisy_15db": {k: v for k, v in noisy.items() if k != "confusion"},
        "history": history, "train_utts": len(train), "val_utts": len(val),
        "train_hours": round(sum(u["n"] for u in train) / SR / 3600, 2), "params": n_params})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf = np.array(final["confusion"], dtype=np.float64)
        conf = conf / np.maximum(conf.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(conf, cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(VISEMES)), VISEMES, rotation=60)
        ax.set_yticks(range(len(VISEMES)), VISEMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Audio→viseme on unseen voices (acc {final['viseme_acc']:.2f})")
        for i in range(len(VISEMES)):
            for j in range(len(VISEMES)):
                if conf[i, j] >= 0.1:
                    ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.tight_layout()
        fig.savefig(export / "metrics" / "audio2viseme_confusion.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    print("done →", export)


if __name__ == "__main__":
    main()
