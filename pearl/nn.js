// Miss Pearl — tiny neural-network runtime (no dependencies).
// Loads the models trained in training/*.py (manifest .json + little-endian .bin)
// and runs them on the CPU in plain JavaScript. Each forward pass mirrors the
// numpy reference implementation that the trainer verifies against PyTorch.

export async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model ${url}: HTTP ${res.status}`);
  const man = await res.json();
  const binUrl = new URL(man.bin, new URL(url, location.href)).href;
  const bres = await fetch(binUrl);
  if (!bres.ok) throw new Error(`weights ${binUrl}: HTTP ${bres.status}`);
  return parseModel(man, await bres.arrayBuffer());
}

/** manifest JSON + weights ArrayBuffer → { config, T } (also used by the Node tests). */
export function parseModel(man, buf) {
  const T = {};
  for (const [name, t] of Object.entries(man.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "f32") {
      T[name] = { shape: t.shape, data: new Float32Array(buf, t.offset, n) };
    } else if (t.dtype === "i8rows") {
      T[name] = {
        shape: t.shape,
        q: new Int8Array(buf, t.offset, n),
        scale: new Float32Array(buf, t.scale_offset, t.shape[0]),
      };
    } else {
      throw new Error(`unknown dtype ${t.dtype}`);
    }
  }
  return { config: man.config, T };
}

// ─────────────────────────────────────────────────────────────
// Audio → viseme TCN (BatchNorm folded at export)
// ─────────────────────────────────────────────────────────────

// y[co][t] = b[co] + Σ_ci Σ_k w[co][ci][k] · x[ci][t + (k-1)·d]   (zero padded)
// Four output channels per pass share each input load (same summation order → identical results).
function conv1d(x, Cin, T, w, b, Cout, K, d, relu) {
  const y = new Float32Array(Cout * T);
  const pad = (d * (K - 1)) >> 1;
  const CK = Cin * K;
  let co = 0;
  for (; co + 3 < Cout; co += 4) {
    const y0 = co * T, y1 = y0 + T, y2 = y1 + T, y3 = y2 + T;
    y.fill(b[co], y0, y1); y.fill(b[co + 1], y1, y2); y.fill(b[co + 2], y2, y3); y.fill(b[co + 3], y3, y3 + T);
    for (let ci = 0; ci < Cin; ci++) {
      const xo = ci * T, w0 = co * CK + ci * K;
      for (let k = 0; k < K; k++) {
        const a0 = w[w0 + k], a1 = w[w0 + CK + k], a2 = w[w0 + 2 * CK + k], a3 = w[w0 + 3 * CK + k];
        const sh = k * d - pad;
        const t0 = Math.max(0, -sh), t1 = Math.min(T, T - sh);
        for (let t = t0; t < t1; t++) {
          const xv = x[xo + t + sh];
          y[y0 + t] += a0 * xv; y[y1 + t] += a1 * xv; y[y2 + t] += a2 * xv; y[y3 + t] += a3 * xv;
        }
      }
    }
  }
  for (; co < Cout; co++) {
    const yo = co * T;
    y.fill(b[co], yo, yo + T);
    for (let ci = 0; ci < Cin; ci++) {
      const xo = ci * T, wo = co * CK + ci * K;
      for (let k = 0; k < K; k++) {
        const wv = w[wo + k];
        const sh = k * d - pad;
        const t0 = Math.max(0, -sh), t1 = Math.min(T, T - sh);
        for (let t = t0; t < t1; t++) y[yo + t] += wv * x[xo + t + sh];
      }
    }
  }
  if (relu) for (let i = 0; i < y.length; i++) if (y[i] < 0) y[i] = 0;
  return y;
}

// 1×1 conv = matmul: y[co][t] = b[co] + Σ_ci w[co][ci] x[ci][t]
function pointwise(x, Cin, T, w, b, Cout) {
  const y = new Float32Array(Cout * T);
  let co = 0;
  for (; co + 3 < Cout; co += 4) {
    const y0 = co * T, y1 = y0 + T, y2 = y1 + T, y3 = y2 + T;
    y.fill(b[co], y0, y1); y.fill(b[co + 1], y1, y2); y.fill(b[co + 2], y2, y3); y.fill(b[co + 3], y3, y3 + T);
    for (let ci = 0; ci < Cin; ci++) {
      const a0 = w[co * Cin + ci], a1 = w[(co + 1) * Cin + ci], a2 = w[(co + 2) * Cin + ci], a3 = w[(co + 3) * Cin + ci];
      const xo = ci * T;
      for (let t = 0; t < T; t++) {
        const xv = x[xo + t];
        y[y0 + t] += a0 * xv; y[y1 + t] += a1 * xv; y[y2 + t] += a2 * xv; y[y3 + t] += a3 * xv;
      }
    }
  }
  for (; co < Cout; co++) {
    const yo = co * T;
    y.fill(b[co], yo, yo + T);
    for (let ci = 0; ci < Cin; ci++) {
      const wv = w[co * Cin + ci];
      const xo = ci * T;
      for (let t = 0; t < T; t++) y[yo + t] += wv * x[xo + t];
    }
  }
  return y;
}

export class Audio2Viseme {
  constructor(model) {
    this.cfg = model.config;
    this.T = model.T;
    this.C = this.cfg.channels;
    this.visemes = this.cfg.visemes;
    this.auxPhones = this.cfg.aux_phones;
  }
  static async load(url) { return new Audio2Viseme(await loadManifest(url)); }

  get featureConfig() {
    const T = this.T;
    return { ...this.cfg, fb: T.mel_fb.data, mean: T.feat_mean.data, std: T.feat_std.data };
  }

  /** True for the song model: it also says whether a voice is singing in each frame. */
  get hasVoice() { return !!this.T["voc.w"]; }

  /** feats: Float32Array [n_mels][T] (already normalised). Returns {vis, ph, voc?} logits, channel-major. */
  forward(feats, nFrames, { phones = true } = {}) {
    const { C, T: W } = this;
    const M = this.cfg.n_mels;
    let x = conv1d(feats, M, nFrames, W["inp.w"].data, W["inp.b"].data, C, 3, 1, true);
    this.cfg.dilations.forEach((d, i) => {
      let y = conv1d(x, C, nFrames, W[`b${i}.conv.w`].data, W[`b${i}.conv.b`].data, C, 3, d, true);
      y = pointwise(y, C, nFrames, W[`b${i}.pw.w`].data, W[`b${i}.pw.b`].data, C);
      for (let j = 0; j < y.length; j++) { const v = x[j] + y[j]; x[j] = v > 0 ? v : 0; }
    });
    const nv = this.visemes.length, np = this.auxPhones.length;
    const vis = pointwise(x, C, nFrames, W["vis.w"].data, W["vis.b"].data, nv);
    const ph = phones ? pointwise(x, C, nFrames, W["ph.w"].data, W["ph.b"].data, np) : null;
    const voc = this.hasVoice ? pointwise(x, C, nFrames, W["voc.w"].data, W["voc.b"].data, 1) : null;
    return { vis, ph, voc, nFrames };
  }

  /** Voice-activity probability per frame (song model only). */
  voice(out) {
    if (!out.voc) return null;
    return Float32Array.from(out.voc, (z) => 1 / (1 + Math.exp(-z)));
  }

  /** Softmax over the viseme axis → array of Float32Array(15) per frame. */
  probs(out, temperature = 1) {
    const { vis, nFrames } = out;
    const nv = this.visemes.length;
    const frames = new Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      const p = new Float32Array(nv);
      let mx = -1e9;
      for (let k = 0; k < nv; k++) mx = Math.max(mx, vis[k * nFrames + t]);
      let s = 0;
      for (let k = 0; k < nv; k++) { p[k] = Math.exp((vis[k * nFrames + t] - mx) / temperature); s += p[k]; }
      for (let k = 0; k < nv; k++) p[k] /= s;
      frames[t] = p;
    }
    return frames;
  }

  /** Receptive field half-width in frames (outputs this far from a window edge are exact). */
  get context() { return 1 + this.cfg.dilations.reduce((a, d) => a + d, 0); }
}

// ─────────────────────────────────────────────────────────────
// G2P Transformer (CTC)
// ─────────────────────────────────────────────────────────────

function layerNorm(x, T, d, w, b) {
  const y = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    let m = 0, v = 0;
    for (let i = 0; i < d; i++) m += x[t * d + i];
    m /= d;
    for (let i = 0; i < d; i++) { const z = x[t * d + i] - m; v += z * z; }
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[t * d + i] = (x[t * d + i] - m) * inv * w[i] + b[i];
  }
  return y;
}

// y[t][o] = b[o] + Σ_i x[t][i] · W[o][i]   (torch Linear layout)
function linear(x, T, din, W, b, dout) {
  const y = new Float32Array(T * dout);
  for (let t = 0; t < T; t++) {
    const xo = t * din;
    for (let o = 0; o < dout; o++) {
      let s = b[o];
      const wo = o * din;
      for (let i = 0; i < din; i++) s += x[xo + i] * W[wo + i];
      y[t * dout + o] = s;
    }
  }
  return y;
}

function erf(x) { // Abramowitz–Stegun 7.1.26, |err| < 1.5e-7
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

export class G2P {
  constructor(model) {
    this.cfg = model.config;
    this.T = model.T;
    this.charId = new Map(this.cfg.chars.map((c, i) => [c, i]));
    this.cache = new Map();
  }
  static async load(url) { return new G2P(await loadManifest(url)); }

  logits(word) {
    const { d, heads: H, up, layers } = this.cfg;
    const W = this.T;
    const ids = [...word].map((c) => this.charId.get(c) ?? 0);
    const T = ids.length * up;
    let x = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const c = ids[Math.floor(t / up)];
      for (let i = 0; i < d; i++) {
        x[t * d + i] = W.char.data[c * d + i] + W.pos.data[t * d + i] + W.sub.data[(t % up) * d + i];
      }
    }
    const hd = d / H, sc = 1 / Math.sqrt(hd);
    for (let l = 0; l < layers; l++) {
      const p = `l${l}.`;
      let h = layerNorm(x, T, d, W[p + "ln1.w"].data, W[p + "ln1.b"].data);
      const qkv = linear(h, T, d, W[p + "in.w"].data, W[p + "in.b"].data, 3 * d);
      const att = new Float32Array(T * d);
      const s = new Float32Array(T);
      for (let hh = 0; hh < H; hh++) {
        const qo = hh * hd, ko = d + hh * hd, vo = 2 * d + hh * hd;
        for (let i = 0; i < T; i++) {
          let mx = -1e9;
          for (let j = 0; j < T; j++) {
            let dot = 0;
            for (let e = 0; e < hd; e++) dot += qkv[i * 3 * d + qo + e] * qkv[j * 3 * d + ko + e];
            s[j] = dot * sc;
            if (s[j] > mx) mx = s[j];
          }
          let sum = 0;
          for (let j = 0; j < T; j++) { s[j] = Math.exp(s[j] - mx); sum += s[j]; }
          for (let e = 0; e < hd; e++) {
            let acc = 0;
            for (let j = 0; j < T; j++) acc += s[j] * qkv[j * 3 * d + vo + e];
            att[i * d + hh * hd + e] = acc / sum;
          }
        }
      }
      const o = linear(att, T, d, W[p + "out.w"].data, W[p + "out.b"].data, d);
      for (let i = 0; i < x.length; i++) x[i] += o[i];
      h = layerNorm(x, T, d, W[p + "ln2.w"].data, W[p + "ln2.b"].data);
      const ffDim = W[p + "ff1.b"].data.length;
      const f = linear(h, T, d, W[p + "ff1.w"].data, W[p + "ff1.b"].data, ffDim);
      for (let i = 0; i < f.length; i++) f[i] = 0.5 * f[i] * (1 + erf(f[i] / Math.SQRT2));
      const f2 = linear(f, T, ffDim, W[p + "ff2.w"].data, W[p + "ff2.b"].data, d);
      for (let i = 0; i < x.length; i++) x[i] += f2[i];
    }
    x = layerNorm(x, T, d, W["lnf.w"].data, W["lnf.b"].data);
    return { logits: linear(x, T, d, W["head.w"].data, W["head.b"].data, this.cfg.n_out), T };
  }

  /** word (lowercase a–z and ') → array of ARPAbet tokens with stress, e.g. ["P","ER1","L"]. */
  phonemize(word) {
    word = word.toLowerCase().replace(/[^a-z']/g, "").slice(0, this.cfg.max_len - 2);
    if (!word) return [];
    if (this.cache.has(word)) return this.cache.get(word);
    const { logits, T } = this.logits(word);
    const n = this.cfg.n_out, out = [];
    let prev = 0;
    for (let t = 0; t < T; t++) {
      let best = 0, bv = -1e9;
      for (let k = 0; k < n; k++) if (logits[t * n + k] > bv) { bv = logits[t * n + k]; best = k; }
      if (best !== prev && best !== 0) out.push(this.cfg.tokens[best - 1]);
      prev = best;
    }
    this.cache.set(word, out);
    return out;
  }
}

// ─────────────────────────────────────────────────────────────
// Emotion student (bag of n-grams, int8 embeddings)
// ─────────────────────────────────────────────────────────────

function fnv1a(s) {
  let h = 0x811c9dc5;
  const bytes = new TextEncoder().encode(s);
  for (const b of bytes) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

export class EmotionModel {
  constructor(model) {
    this.cfg = model.config;
    this.T = model.T;
    this.labels = this.cfg.labels;
    this.vocab = new Map(this.cfg.vocab.map((w, i) => [w, i]));
    this.re = new RegExp(this.cfg.token_regex, "g");
  }
  static async load(url) { return new EmotionModel(await loadManifest(url)); }

  tokens(text) { return (text.toLowerCase().replace(/’/g, "'").match(this.re) || []); }

  featureIds(toks) {
    const V = this.vocab.size, ids = [];
    for (const t of toks) ids.push(this.vocab.has(t) ? this.vocab.get(t) : V + (fnv1a(t) % this.cfg.oov_buckets));
    const base = V + this.cfg.oov_buckets;
    for (let i = 0; i + 1 < toks.length; i++) ids.push(base + (fnv1a(toks[i] + " " + toks[i + 1]) % this.cfg.bigram_buckets));
    return ids.length ? ids : [0];
  }

  /** → { label: prob } over anger, disgust, fear, joy, neutral, sadness, surprise */
  predict(text) {
    const { dim, hidden } = this.cfg;
    const E = this.T.emb, ids = this.featureIds(this.tokens(text));
    const h = new Float32Array(dim);
    for (const r of ids) {
      const s = E.scale[r];
      for (let i = 0; i < dim; i++) h[i] += E.q[r * dim + i] * s;
    }
    for (let i = 0; i < dim; i++) h[i] /= ids.length;
    const z1 = linear(h, 1, dim, this.T["l1.w"].data, this.T["l1.b"].data, hidden).map((v) => (v > 0 ? v : 0));
    const z = linear(z1, 1, hidden, this.T["l2.w"].data, this.T["l2.b"].data, this.labels.length);
    const mx = Math.max(...z);
    const e = Array.from(z, (v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    const out = {};
    this.labels.forEach((l, i) => (out[l] = e[i] / s));
    return out;
  }
}
