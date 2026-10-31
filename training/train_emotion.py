"""
Step 4 — sentence emotion model (drives brows, lids, cheeks, smile).

Teacher: j-hartmann/emotion-english-distilroberta-base (7 Ekman classes).
Student: a tiny bag-of-ngrams network (≈1 MB, int8) that runs instantly in
the browser.  The teacher labels ~100k sentences — GoEmotions, the
dair-ai emotion set, assistant replies from OpenAssistant and Wikipedia —
and the student learns to match its probability distribution.  When the
LLM backend is online, its inline [emotion] tags take priority; this
model covers everything else (typed text, offline mode, untagged replies).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (EMO_BIGRAM_BUCKETS, EMO_DIM, EMO_HIDDEN, EMO_OOV_BUCKETS, EMOTIONS,  # noqa: E402
                    BinWriter, emo_feature_ids, emo_tokens, env_info, save_json)

TEACHER = "j-hartmann/emotion-english-distilroberta-base"
DAIR_MAP = {0: "sadness", 1: "joy", 2: "joy", 3: "anger", 4: "fear", 5: "surprise"}  # love → joy

EXTRA = [
    "Well hello there, my genius friend!", "Tickety-tock! Pearl AI Labs is so proud of your curiosity!",
    "Oh no, I'm so sorry to hear that.", "That is absolutely disgusting.", "Wow, I can't believe it!",
    "I'm terrified of what might happen.", "Stop that right now, I'm furious!", "The meeting is at 3 pm.",
    "Hmm, let me think about that.", "Congratulations, you did it!", "I miss you so much.",
    "What a lovely surprise!", "That's gross, please don't.", "Water boils at 100 degrees Celsius.",
]


def sentences(text):
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if 2 <= len(s.split()) <= 40:
            yield s


def gather(seed, smoke=False):
    from datasets import load_dataset
    rng = random.Random(seed)
    texts, gold = [], []

    def add(t, g=-1):
        texts.append(t)
        gold.append(g)

    def safe(name, fn):
        try:
            n0 = len(texts)
            fn()
            print(f"  {name}: {len(texts) - n0:,}")
        except Exception as e:
            print(f"  ! {name} unavailable: {e}")

    def goemo():
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        for split in ("train", "validation"):
            for r in ds[split]:
                add(r["text"])

    def dair():
        ds = load_dataset("dair-ai/emotion", "split")
        for r in ds["train"]:
            add(r["text"], EMOTIONS.index(DAIR_MAP[r["label"]]))

    def oasst():
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        pool = []
        for r in ds:
            if r["role"] == "assistant" and r["lang"] == "en":
                pool.extend(sentences(r["text"]))
        rng.shuffle(pool)
        for s in pool[:40000]:
            add(s)

    def wiki():
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        pool = [s for r in ds for s in sentences(r["text"]) if not s.startswith("=")]
        rng.shuffle(pool)
        for s in pool[:10000]:
            add(s)

    safe("go_emotions", goemo)
    safe("dair-ai/emotion", dair)
    safe("oasst1 assistant", oasst)
    safe("wikitext", wiki)
    for t in EXTRA * 5:
        add(t)
    if smoke:
        idx = rng.sample(range(len(texts)), min(2000, len(texts)))
        texts = [texts[i] for i in idx]
        gold = [gold[i] for i in idx]
    return texts, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Environment:", env_info())
    export, ckpt = Path(args.export), Path(args.ckpt)
    export.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    cache = ckpt / ("emotion_teacher_smoke.npz" if args.smoke else "emotion_teacher.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        texts, gold, soft = list(z["texts"]), list(z["gold"]), z["soft"]
        print(f"loaded {len(texts):,} teacher-labelled sentences from cache")
    else:
        print("gathering text …")
        texts, gold = gather(0, args.smoke)
        print(f"labelling {len(texts):,} sentences with teacher {TEACHER} …")
        from transformers import pipeline
        clf = pipeline("text-classification", model=TEACHER, top_k=None, truncation=True,
                       max_length=128, device=0 if device == "cuda" else -1)
        t0 = time.time()
        soft = np.zeros((len(texts), len(EMOTIONS)), dtype=np.float32)
        bs = 256 if device == "cuda" else 32
        for i in range(0, len(texts), bs):
            outs = clf(texts[i:i + bs], batch_size=bs)
            for j, o in enumerate(outs):
                for d in o:
                    soft[i + j, EMOTIONS.index(d["label"])] = d["score"]
            if (i // bs) % 50 == 0:
                print(f"  {i + bs:,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(cache, texts=np.array(texts, dtype=object), gold=np.array(gold), soft=soft)
    gold = np.asarray(gold)

    # ── vocabulary ────────────────────────────────────────────
    toks = [emo_tokens(t) for t in texts]
    cnt = Counter(t for ts in toks for t in ts)
    vocab_list = [w for w, c in cnt.most_common(args.vocab) if c >= 3]
    vocab = {w: i for i, w in enumerate(vocab_list)}
    n_rows = len(vocab) + EMO_OOV_BUCKETS + EMO_BIGRAM_BUCKETS
    feats = [emo_feature_ids(ts, vocab) for ts in toks]
    print(f"vocab {len(vocab):,} | embedding rows {n_rows:,}")

    idx = np.arange(len(texts))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    n_val = max(200, len(idx) // 20)
    val_i, tr_i = idx[:n_val], idx[n_val:]

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.EmbeddingBag(n_rows, EMO_DIM, mode="mean")
            self.l1 = nn.Linear(EMO_DIM, EMO_HIDDEN)
            self.l2 = nn.Linear(EMO_HIDDEN, len(EMOTIONS))
            self.drop = nn.Dropout(0.1)

        def forward(self, flat, offsets):
            h = self.emb(flat, offsets)
            return self.l2(self.drop(torch.relu(self.l1(h))))

    def batchify(ii):
        flat, offs = [], []
        for i in ii:
            offs.append(len(flat))
            f = feats[i] or [0]
            flat.extend(f)
        return torch.tensor(flat, device=device), torch.tensor(offs, device=device)

    model = Student().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    soft_t = torch.tensor(soft, device=device)
    gold_t = torch.tensor(gold, device=device)

    @torch.no_grad()
    def evaluate(ii):
        model.eval()
        agree, n, g_ok, g_n = 0, 0, 0, 0
        for s in range(0, len(ii), 2048):
            b = ii[s:s + 2048]
            p = model(*batchify(b)).argmax(-1)
            agree += int((p == soft_t[b].argmax(-1)).sum())
            n += len(b)
            m = gold_t[b] >= 0
            g_ok += int((p[m] == gold_t[b][m]).sum())
            g_n += int(m.sum())
        model.train()
        return {"teacher_agreement": round(agree / n, 4),
                "gold_acc_dair": round(g_ok / max(g_n, 1), 4) if g_n else None}

    best = -1
    for ep in range(args.epochs):
        rng.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            b = tr_i[s:s + 256]
            logits = model(*batchify(b))
            logp = F.log_softmax(logits, -1)
            loss = F.kl_div(logp, soft_t[b], reduction="batchmean")
            m = gold_t[b] >= 0
            if m.any():
                loss = loss + 0.5 * F.nll_loss(logp[m], gold_t[b][m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        r = evaluate(val_i)
        print(f"epoch {ep + 1:2d} | loss {float(loss):.3f} | val teacher agreement "
              f"{r['teacher_agreement']:.3f} | dair gold acc {r['gold_acc_dair']}", flush=True)
        if r["teacher_agreement"] > best:
            best = r["teacher_agreement"]
            torch.save(model.state_dict(), ckpt / "emotion_best.pt")
    model.load_state_dict(torch.load(ckpt / "emotion_best.pt", map_location=device))
    final = evaluate(val_i)
    teacher_gold = None
    m = gold[val_i] >= 0
    if m.any():
        teacher_gold = round(float((soft[val_i][m].argmax(-1) == gold[val_i][m]).mean()), 4)
    print(f"FINAL emotion student: teacher agreement {final['teacher_agreement']:.3f}, "
          f"gold acc {final['gold_acc_dair']} (teacher itself: {teacher_gold})")

    # ── export ────────────────────────────────────────────────
    model = model.cpu().eval()
    bw = BinWriter()
    bw.i8_rows("emb", model.emb.weight.detach().numpy())
    bw.f32("l1.w", model.l1.weight.detach().numpy())
    bw.f32("l1.b", model.l1.bias.detach().numpy())
    bw.f32("l2.w", model.l2.weight.detach().numpy())
    bw.f32("l2.b", model.l2.bias.detach().numpy())
    cfg = {"labels": EMOTIONS, "vocab": vocab_list, "oov_buckets": EMO_OOV_BUCKETS,
           "bigram_buckets": EMO_BIGRAM_BUCKETS, "dim": EMO_DIM, "hidden": EMO_HIDDEN,
           "token_regex": "[a-z0-9']+|[!?]", "hash": "fnv1a32",
           "teacher": TEACHER, "metrics": dict(final, teacher_gold_acc=teacher_gold),
           "env": env_info()}
    bw.save(export, "emotion", cfg)

    # quantised reference outputs for the JS test
    emb_q = model.emb.weight.detach().numpy()
    scale = np.maximum(np.abs(emb_q).max(1), 1e-8) / 127
    emb_q = np.clip(np.round(emb_q / scale[:, None]), -127, 127) * scale[:, None]
    tests = []
    for t in ["I am so happy to see you!", "That is terrible news, I feel awful.",
              "What?! No way!", "The capital of Uganda is Kampala.", "I hate this so much.",
              "I'm scared of the dark.", "Ew, that smells rotten.", "Well hello there, my genius friend!"]:
        ids = emo_feature_ids(emo_tokens(t), vocab) or [0]
        h = emb_q[ids].mean(0)
        h = np.maximum(0, model.l1.weight.detach().numpy() @ h + model.l1.bias.detach().numpy())
        z = model.l2.weight.detach().numpy() @ h + model.l2.bias.detach().numpy()
        p = np.exp(z - z.max())
        p /= p.sum()
        tests.append({"text": t, "probs": p.round(4).tolist(), "top": EMOTIONS[int(p.argmax())]})
        print(f"  {EMOTIONS[int(p.argmax())]:9s} {p.max():.2f}  {t}")
    save_json(export / "tests" / "emotion_test.json", tests)
    save_json(export / "metrics" / "emotion.json", {"final": final, "teacher_gold_acc": teacher_gold,
                                                    "sentences": len(texts), "vocab": len(vocab)})
    print("done →", export)


if __name__ == "__main__":
    main()
