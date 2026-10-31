"""
Step 3 — grapheme→phoneme model + pronunciation lexicon.

The browser looks words up in the CMU Pronouncing Dictionary first
(≈126k words, exported as a compact lexicon).  Anything not in the
dictionary — names, slang, typos, invented words — goes through this
neural G2P so Miss Pearl can pronounce (and therefore lip-sync) ANY word.

Model: letters are repeated ×3, a 4-layer Transformer encoder labels
each slot, CTC collapses the slots into an ARPAbet phone sequence with
lexical stress.  Non-autoregressive → one forward pass per word.
"""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (G2P_CODE_ALPHABET, G2P_TOKENS, BinWriter, build_g2p_model,  # noqa: E402
                    env_info, save_json)

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
CHARS = ["<pad>"] + list("'abcdefghijklmnopqrstuvwxyz")
CHAR_ID = {c: i for i, c in enumerate(CHARS)}
TOK_ID = {t: i + 1 for i, t in enumerate(G2P_TOKENS)}          # 0 = CTC blank
MAX_LEN = 32
WORD_RE = re.compile(r"^[a-z']+$")


def load_cmudict(cache: Path):
    if not cache.exists():
        print("downloading CMUdict …")
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CMUDICT_URL, cache)
    lex = {}
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        word = re.sub(r"\(\d+\)$", "", parts[0].lower())
        phones = parts[1:]
        if word in lex or not WORD_RE.match(word) or len(word) > MAX_LEN - 2:
            continue
        if all(p in TOK_ID for p in phones) and phones:
            lex[word] = phones
    return lex


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def ctc_greedy(ids):
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(G2P_TOKENS[i - 1])
        prev = i
    return out


def np_layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def np_gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def np_forward(W, cfg, word):
    """Reference forward pass mirroring pearl/nn.js (single word, no padding)."""
    up, d, H = cfg["up"], cfg["d"], cfg["heads"]
    ids = [CHAR_ID[c] for c in word]
    x = np.repeat(W["char"][ids], up, axis=0).astype(np.float64)
    T = x.shape[0]
    pos = np.arange(T)
    x = x + W["pos"][pos] + W["sub"][pos % up]
    hd = d // H
    for l in range(cfg["layers"]):
        p = f"l{l}."
        h = np_layernorm(x, W[p + "ln1.w"], W[p + "ln1.b"])
        qkv = h @ W[p + "in.w"].T + W[p + "in.b"]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        att = np.zeros_like(h)
        for i in range(H):
            sl = slice(i * hd, (i + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / math.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att[:, sl] = s @ v[:, sl]
        x = x + att @ W[p + "out.w"].T + W[p + "out.b"]
        h = np_layernorm(x, W[p + "ln2.w"], W[p + "ln2.b"])
        h = np_gelu(h @ W[p + "ff1.w"].T + W[p + "ff1.b"])
        x = x + h @ W[p + "ff2.w"].T + W[p + "ff2.b"]
    x = np_layernorm(x, W["lnf.w"], W["lnf.b"])
    return x @ W["head.w"].T + W["head.b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="/content/cache")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Environment:", env_info())
    export, ckpt = Path(args.export), Path(args.ckpt)
    export.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    lex = load_cmudict(Path(args.cache) / "cmudict.dict")
    print(f"CMUdict entries: {len(lex):,}")

    # ── compact lexicon for the browser ───────────────────────
    code = {t: G2P_CODE_ALPHABET[i] for i, t in enumerate(G2P_TOKENS)}
    lines = [" ".join(G2P_TOKENS)]
    lines += [f"{w} {''.join(code[p] for p in ph)}" for w, ph in sorted(lex.items())]
    (export / "lexicon.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"lexicon.txt: {(export / 'lexicon.txt').stat().st_size / 1e6:.2f} MB")

    words = sorted(lex)
    random.shuffle(words)
    if args.smoke:
        words = words[:3000]
        args.epochs, args.minutes = 1, 2
    n_val = max(200, int(len(words) * 0.03))
    val_w, train_w = words[:n_val], words[n_val:]

    def encode(batch):
        L = max(len(w) for w in batch)
        chars = torch.zeros(len(batch), L, dtype=torch.long)
        for i, w in enumerate(batch):
            chars[i, :len(w)] = torch.tensor([CHAR_ID[c] for c in w])
        lens = torch.tensor([len(w) for w in batch])
        tgt = [torch.tensor([TOK_ID[p] for p in lex[w]]) for w in batch]
        return chars, lens, tgt

    model = build_g2p_model(len(CHARS), max_len=MAX_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"G2P params: {n_params:,} | train {len(train_w):,} words | val {len(val_w):,} words")
    steps_per_epoch = math.ceil(len(train_w) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=total, pct_start=0.05)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    @torch.no_grad()
    def evaluate(ws):
        model.eval()
        errs = errs_ns = n_ph = wrong = 0
        for i in range(0, len(ws), 512):
            batch = ws[i:i + 512]
            chars, lens, _ = encode(batch)
            out = model(chars.to(device), lens.to(device)).argmax(-1).cpu().numpy()
            for j, w in enumerate(batch):
                hyp = ctc_greedy(out[j][: lens[j] * 3])
                ref = lex[w]
                e = edit_distance(hyp, ref)
                errs += e
                errs_ns += edit_distance([re.sub(r"\d", "", p) for p in hyp],
                                         [re.sub(r"\d", "", p) for p in ref])
                n_ph += len(ref)
                wrong += e > 0
        model.train()
        return {"per": round(errs / n_ph, 4), "per_no_stress": round(errs_ns / n_ph, 4),
                "word_acc": round(1 - wrong / len(ws), 4)}

    t0, step, best = time.time(), 0, 9.0
    done = False
    for ep in range(args.epochs):
        random.shuffle(train_w)
        for i in range(0, len(train_w), args.batch):
            chars, lens, tgt = encode(train_w[i:i + args.batch])
            logits = model(chars.to(device), lens.to(device))
            logp = F.log_softmax(logits, -1).transpose(0, 1)
            loss = ctc(logp, torch.cat(tgt).to(device), (lens * 3).to(device),
                       torch.tensor([len(t) for t in tgt]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total - 1:
                sched.step()
            step += 1
            if (time.time() - t0) > args.minutes * 60:
                done = True
                break
        m = evaluate(val_w[:2000])
        print(f"epoch {ep + 1:3d} | step {step:6d} | loss {float(loss):.3f} | val PER {m['per']:.3f} "
              f"(no-stress {m['per_no_stress']:.3f}) | word acc {m['word_acc']:.3f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if m["per"] < best:
            best = m["per"]
            torch.save({"model": model.state_dict(), "cfg": model.cfg}, ckpt / "g2p_best.pt")
        if done:
            print("time budget reached")
            break

    model.load_state_dict(torch.load(ckpt / "g2p_best.pt", map_location=device)["model"])
    final = evaluate(val_w)
    print(f"FINAL G2P on {len(val_w):,} held-out words: PER {final['per']:.3f}, "
          f"PER(no stress) {final['per_no_stress']:.3f}, word accuracy {final['word_acc']:.3f}")

    # ── export ────────────────────────────────────────────────
    model = model.cpu().eval()
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    W = {"char": sd["char.weight"], "pos": sd["pos.weight"], "sub": sd["sub.weight"],
         "lnf.w": sd["enc.norm.weight"], "lnf.b": sd["enc.norm.bias"],
         "head.w": sd["out.weight"], "head.b": sd["out.bias"]}
    for l in range(model.cfg["layers"]):
        p = f"enc.layers.{l}."
        W.update({f"l{l}.in.w": sd[p + "self_attn.in_proj_weight"], f"l{l}.in.b": sd[p + "self_attn.in_proj_bias"],
                  f"l{l}.out.w": sd[p + "self_attn.out_proj.weight"], f"l{l}.out.b": sd[p + "self_attn.out_proj.bias"],
                  f"l{l}.ln1.w": sd[p + "norm1.weight"], f"l{l}.ln1.b": sd[p + "norm1.bias"],
                  f"l{l}.ln2.w": sd[p + "norm2.weight"], f"l{l}.ln2.b": sd[p + "norm2.bias"],
                  f"l{l}.ff1.w": sd[p + "linear1.weight"], f"l{l}.ff1.b": sd[p + "linear1.bias"],
                  f"l{l}.ff2.w": sd[p + "linear2.weight"], f"l{l}.ff2.b": sd[p + "linear2.bias"]})
    bw = BinWriter()
    for k, v in W.items():
        bw.f32(k, v)
    cfg = dict(model.cfg, chars=CHARS, tokens=G2P_TOKENS, blank=0, metrics=final,
               params=n_params, env=env_info())
    bw.save(export, "g2p", cfg)

    tests = {}
    for w in ["pearl", "kampala", "tickety", "genius", "uganda", "zorblax", "claude",
              "photosynthesis", "musoke", "nakawa"]:
        logits = np_forward(W, model.cfg, w)
        with torch.no_grad():
            tl = model(torch.tensor([[CHAR_ID[c] for c in w]]), torch.tensor([len(w)]))[0].numpy()
        diff = float(np.abs(logits - tl).max())
        assert diff < 1e-3, f"G2P export mismatch {diff}"
        tests[w] = {"phones": ctc_greedy(logits.argmax(-1)), "in_lexicon": w in lex,
                    "logits_first_row": logits[0].round(4).tolist()}
        print(f"  {w:16s} → {' '.join(tests[w]['phones'])}{'' if w not in lex else '   (lexicon: ' + ' '.join(lex[w]) + ')'}")
    save_json(export / "tests" / "g2p_test.json", tests)
    save_json(export / "metrics" / "g2p.json", {"final_heldout": final, "params": n_params,
                                                "train_words": len(train_w), "val_words": len(val_w)})
    print("done →", export)


if __name__ == "__main__":
    main()
