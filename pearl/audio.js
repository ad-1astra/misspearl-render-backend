// Miss Pearl — audio front-end: 16 kHz resampling, log-mel features identical
// to training/common.py (hann-periodic 400 / hop 160 / FFT 512 / 64 mels),
// loudness envelopes and whole-clip viseme analysis with the trained model.

export class FeatureExtractor {
  constructor(cfg) {
    this.cfg = cfg;
    const { win, n_fft } = cfg;
    this.window = new Float64Array(win);
    for (let k = 0; k < win; k++) this.window[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / win);
    // radix-2 FFT tables
    this.N = n_fft;
    this.rev = new Uint32Array(n_fft);
    const bits = Math.log2(n_fft);
    for (let i = 0; i < n_fft; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n_fft / 2);
    this.sin = new Float64Array(n_fft / 2);
    for (let i = 0; i < n_fft / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n_fft);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n_fft);
    }
    this.re = new Float64Array(n_fft);
    this.im = new Float64Array(n_fft);
    this.power = new Float64Array(n_fft / 2 + 1);
    this.nBins = n_fft / 2 + 1;
  }

  numFrames(n) { return n < this.cfg.win ? 0 : 1 + Math.floor((n - this.cfg.win) / this.cfg.hop); }

  _fft() {
    const { N, re, im, rev, cos, sin } = this;
    for (let i = 0; i < N; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1, step = N / size;
      for (let s = 0; s < N; s += size) {
        for (let k = 0; k < half; k++) {
          const wr = cos[k * step], wi = sin[k * step];
          const a = s + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  /** samples: Float32Array @16 kHz in [-1,1] → { data: Float32Array[n_mels*T] (normalised, channel-major), T } */
  compute(samples, startFrame = 0, nFrames = null) {
    const { win, hop, n_mels, fb, mean, std, log_eps } = this.cfg;
    const total = this.numFrames(samples.length);
    const T = nFrames == null ? total - startFrame : Math.min(nFrames, total - startFrame);
    const out = new Float32Array(n_mels * Math.max(T, 0));
    const nb = this.nBins;
    for (let t = 0; t < T; t++) {
      const off = (startFrame + t) * hop;
      this.re.fill(0); this.im.fill(0);
      for (let k = 0; k < win; k++) this.re[k] = samples[off + k] * this.window[k];
      this._fft();
      for (let b = 0; b < nb; b++) this.power[b] = this.re[b] * this.re[b] + this.im[b] * this.im[b];
      for (let m = 0; m < n_mels; m++) {
        let s = 0;
        const fo = m * nb;
        for (let b = 0; b < nb; b++) { const w = fb[fo + b]; if (w) s += w * this.power[b]; }
        out[m * T + t] = (Math.log(s + log_eps) - mean[m]) / std[m];
      }
    }
    return { data: out, T };
  }
}

/** Decode → mono 16 kHz Float32Array using the browser's resampler. */
export async function to16kMono(audioBuffer) {
  const sr = 16000;
  const len = Math.max(1, Math.ceil(audioBuffer.duration * sr));
  const ctx = new OfflineAudioContext(1, len, sr);
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  return out.getChannelData(0).slice();
}

/** RMS loudness per 10 ms frame, normalised to ~[0,1] by the clip's 95th percentile. */
export function loudness(samples, hop = 160, win = 400) {
  const n = samples.length < win ? 0 : 1 + Math.floor((samples.length - win) / hop);
  const e = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    let s = 0;
    const o = t * hop;
    for (let k = 0; k < win; k++) s += samples[o + k] * samples[o + k];
    e[t] = Math.sqrt(s / win);
  }
  const sorted = Array.from(e).sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 1e-4;
  for (let t = 0; t < n; t++) e[t] = Math.min(1, e[t] / Math.max(p95, 1e-4));
  return e;
}

/**
 * Run the trained audio→viseme network over a whole clip in overlapping chunks
 * (yielding to the UI between chunks). Returns per-frame viseme probabilities.
 */
export async function analyzeClip(model, fx, samples16k, onProgress) {
  const total = fx.numFrames(samples16k.length);
  const ctx = model.context;            // frames of context needed for exact outputs
  const CH = 600;
  const frames = new Array(total);
  const voice = model.hasVoice ? new Float32Array(total) : null;
  const feats = { n: fx.cfg.n_mels, data: new Float32Array(fx.cfg.n_mels * total) };   // kept for beat tracking
  for (let s = 0; s < total; s += CH) {
    const a = Math.max(0, s - ctx), b = Math.min(total, s + CH + ctx);
    const f = fx.compute(samples16k, a, b - a);
    const out = model.forward(f.data, f.T, { phones: false });
    const pr = model.probs(out);
    const vp = voice ? model.voice(out) : null;
    for (let t = s; t < Math.min(total, s + CH); t++) {
      frames[t] = pr[t - a];
      if (vp) voice[t] = vp[t - a];
      if (feats) for (let m = 0; m < feats.n; m++) feats.data[m * total + t] = f.data[m * f.T + t - a];
    }
    onProgress && onProgress(Math.min(1, (s + CH) / total));
    await new Promise((r) => setTimeout(r, 0));
  }
  return { fps: 100, frames, voice, feats, energy: loudness(samples16k), duration: samples16k.length / 16000 };
}

/**
 * Live microphone → visemes. Keeps a rolling buffer and re-runs the network on a
 * short window, emitting exact outputs `context` frames behind real time.
 */
export class StreamingVisemes {
  constructor(model, fx, onFrame) {
    this.model = model; this.fx = fx; this.onFrame = onFrame;
    this.ctxFrames = model.context;
    this.buf = new Float32Array(16000 * 4);
    this.len = 0;            // samples currently in buf
    this.emitted = 0;        // absolute frame index of next frame to emit
    this.base = 0;           // absolute frame index of buf[0]
  }
  push(chunk) {
    const { hop, win } = this.fx.cfg;
    if (this.len + chunk.length > this.buf.length) {
      // drop old audio but keep enough left context
      const keepFrames = Math.max(0, this.emitted - this.base - this.ctxFrames - 2);
      const drop = keepFrames * hop;
      this.buf.copyWithin(0, drop, this.len);
      this.len -= drop;
      this.base += keepFrames;
    }
    this.buf.set(chunk, this.len);
    this.len += chunk.length;
    const avail = this.fx.numFrames(this.len);              // frames in buffer
    const ready = this.base + avail - this.ctxFrames;       // last exact frame (exclusive)
    if (ready <= this.emitted) return;
    const a = Math.max(0, this.emitted - this.base - this.ctxFrames);
    const b = avail;
    const f = this.fx.compute(this.buf.subarray(0, this.len), a, b - a);
    const out = this.model.forward(f.data, f.T);
    const pr = this.model.probs(out);
    for (let abs = this.emitted; abs < ready; abs++) {
      const i = abs - this.base - a;
      const o = (abs - this.base) * hop;
      let e = 0;
      for (let k = 0; k < win && o + k < this.len; k++) e += this.buf[o + k] ** 2;
      this.onFrame(pr[i], Math.sqrt(e / win), abs);
    }
    this.emitted = ready;
  }
}
