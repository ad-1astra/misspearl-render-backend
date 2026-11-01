// Miss Pearl — songs.
//
//   voice vs. beat  the trained song model (sing2viseme) says, every 10 ms, whether a
//                   voice is singing; the mouth only moves then. Between vocal phrases
//                   the music plays on and she grooves to the beat with a closed mouth.
//   beat            onset envelope → tempo (autocorrelation) → beat times (dynamic
//                   programming, Ellis 2007). Her head bobs on the beat.
//   lyrics          optional: Whisper (in the browser, transformers.js) hears the words with
//                   word timestamps. Each line is turned into its true mouth-shape sequence
//                   and force-aligned to the song model's per-frame visemes, and the
//                   emotion model reads the line so her face acts out what it means.

import { phonesToVisemes, planUtterance, VISEMES } from "./text.js?v=20261101h";

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const VOWEL_V = new Set(["aa", "E", "ih", "oh", "ou", "RR"]);

// ── voice activity → sung sections ──────────────────────────
export function voiceSections(voice, fps = 100, { on = 0.55, off = 0.4, minOn = 0.12, minGap = 0.25 } = {}) {
  const T = voice.length, sm = new Float32Array(T);
  for (let t = 0; t < T; t++) {            // light smoothing (±30 ms)
    let s = 0, n = 0;
    for (let k = -3; k <= 3; k++) { const j = t + k; if (j >= 0 && j < T) { s += voice[j]; n++; } }
    sm[t] = s / n;
  }
  const secs = [];
  let inside = false, a = 0;
  for (let t = 0; t < T; t++) {
    if (!inside && sm[t] >= on) { inside = true; a = t; }
    else if (inside && sm[t] < off) { inside = false; secs.push([a, t]); }
  }
  if (inside) secs.push([a, T]);
  const merged = [];
  for (const s of secs) {
    const last = merged[merged.length - 1];
    if (last && (s[0] - last[1]) / fps < minGap) last[1] = s[1];
    else merged.push([...s]);
  }
  return { smooth: sm, sections: merged.filter(([a, b]) => (b - a) / fps >= minOn).map(([a, b]) => ({ t0: a / fps, t1: b / fps })) };
}

// ── beat tracking ────────────────────────────────────────────
/** feats: normalised log-mel [n_mels × T] (channel-major). → { tempo, beats: [{t, s}] } */
export function trackBeats(feats, nMels, T, fps = 100) {
  if (T < fps * 4) return { tempo: 0, beats: [] };
  const o = new Float32Array(T);
  const lowBands = Math.round(nMels * 0.25);
  for (let t = 1; t < T; t++) {
    let s = 0;
    for (let m = 0; m < nMels; m++) {
      const d = feats[m * T + t] - feats[m * T + t - 1];
      if (d > 0) s += m < lowBands ? 1.6 * d : d;        // kick & bass count more
    }
    o[t] = s;
  }
  // remove the local mean, half-wave rectify, normalise
  const W = Math.round(0.4 * fps), env = new Float32Array(T);
  let acc = 0;
  for (let t = 0; t < T; t++) {
    acc += o[t] - (t - W >= 0 ? o[t - W] : 0);
    env[t] = Math.max(0, o[t] - acc / Math.min(t + 1, W));
  }
  let mu = 0; for (const v of env) mu += v; mu /= T;
  let sd = 0; for (const v of env) sd += (v - mu) ** 2; sd = Math.sqrt(sd / T) || 1;
  for (let t = 0; t < T; t++) env[t] /= sd;
  // tempo: autocorrelation with a preference around 120 BPM
  const minLag = Math.round((60 / 200) * fps), maxLag = Math.round((60 / 60) * fps);
  let bestLag = 50, bestScore = -1;
  for (let L = minLag; L <= maxLag; L++) {
    let s = 0;
    for (let t = L; t < T; t++) s += env[t] * env[t - L];
    s /= T - L;
    const bpm = (60 * fps) / L;
    const w = Math.exp(-0.5 * (Math.log2(bpm / 120) / 0.9) ** 2);
    if (s * w > bestScore) { bestScore = s * w; bestLag = L; }
  }
  // dynamic-programming beat tracker
  const P = bestLag, lambda = 6;
  const C = new Float32Array(T), back = new Int32Array(T).fill(-1);
  for (let t = 0; t < T; t++) {
    let best = 0, bi = -1;
    for (let tau = Math.round(P / 2); tau <= 2 * P; tau++) {
      const j = t - tau;
      if (j < 0) break;
      const sc = C[j] - lambda * Math.log(tau / P) ** 2;
      if (sc > best || bi < 0) { best = sc; bi = j; }
    }
    C[t] = env[t] + (bi >= 0 ? Math.max(0, best) : 0);
    back[t] = best > 0 ? bi : -1;
  }
  let t = T - 1, bestEnd = T - 1;
  for (let k = Math.max(0, T - P); k < T; k++) if (C[k] > C[bestEnd]) bestEnd = k;
  const beats = [];
  for (t = bestEnd; t >= 0; t = back[t]) {
    let pk = 0;
    for (let k = Math.max(0, t - 3); k <= Math.min(T - 1, t + 3); k++) pk = Math.max(pk, env[k]);
    beats.push({ t: t / fps, s: clamp(pk / 3, 0.15, 1) });
    if (back[t] < 0) break;
  }
  beats.reverse();
  return { tempo: Math.round((60 * fps) / P), beats };
}

// ── lyrics: Whisper in the browser ──────────────────────────
const TJS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
export const LYRICS_MODEL = "Xenova/whisper-base.en";
let asrPromise = null;

export function loadLyricsReader(onStatus) {
  if (!asrPromise) {
    asrPromise = (async () => {
      const { pipeline, env } = await import(TJS_URL);
      env.allowLocalModels = false;
      const files = new Map();
      // 8-bit encoder + decoder on the CPU (WebAssembly): the smallest download (77 MB) and the
      // most reliable word timestamps. (fp16/4-bit variants read sung words badly; 8-bit ops are slow on WebGPU.)
      return pipeline("automatic-speech-recognition", LYRICS_MODEL, {
        device: "wasm",
        dtype: { encoder_model: "q8", decoder_model_merged: "q8" },
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            files.set(p.file, [p.loaded, p.total]);
            let l = 0, tot = 0; for (const [a, b] of files.values()) { l += a; tot += b; }
            onStatus && onStatus(`Downloading the lyrics reader… ${Math.round(l / 1e6)} / ${Math.round(tot / 1e6)} MB (only the first time)`);
          }
        },
      });
    })();
    asrPromise.catch(() => { asrPromise = null; });
  }
  return asrPromise;
}

/**
 * Hear the words of a song. Only the sung sections are passed to Whisper (the rest is
 * silenced) so instrumental parts can't produce invented words.
 * → [{ w: "word", t0, t1 }]
 */
export async function readLyrics(samples16k, sections, onStatus) {
  const asr = await loadLyricsReader(onStatus);
  const x = new Float32Array(samples16k.length);
  for (const s of sections) {
    const a = Math.max(0, Math.floor((s.t0 - 0.15) * 16000)), b = Math.min(x.length, Math.ceil((s.t1 + 0.15) * 16000));
    x.set(samples16k.subarray(a, b), a);
  }
  onStatus && onStatus("Listening to the lyrics…");
  const out = await asr(x, { return_timestamps: "word", chunk_length_s: 30, stride_length_s: 5 });
  return cleanWords(out.chunks || [], sections);
}

/**
 * Whisper word stamps on songs need tidying: at the 30 s chunk seams it can emit words
 * out of order or with zero length, and over instrumentals it may invent words.
 */
export function cleanWords(chunks, sections) {
  const raw = [];
  for (const c of chunks) {
    const w = (c.text || "").trim();
    const [t0, t1] = c.timestamp || [];
    if (!w || t0 == null || !isFinite(t0)) continue;
    if (/^\[.*\]$|^\(.*\)$|♪|^[-–—.,!?…]+$/.test(w)) continue;   // [Music], (instrumental), ♪, bare punctuation
    raw.push({ w, t0, t1: t1 != null && isFinite(t1) ? t1 : t0 + 0.3 });
  }
  const words = [];
  for (const wd of raw) {
    const prev = words[words.length - 1];
    if (wd.t1 - wd.t0 < 0.03 && Math.abs(((wd.t0 + 0.02) % 30) - 0.02) < 0.06) continue;   // phantom at a chunk seam
    if (prev && wd.t0 < prev.t0 - 0.05) continue;                                          // out of order
    const t0 = prev ? Math.max(wd.t0, prev.t1 - 0.02) : wd.t0;
    words.push({ w: wd.w, t0, t1: Math.max(wd.t1, t0 + 0.08) });
  }
  // Whisper stretches a word back across the silence before it (e.g. the first word of a
  // song "starting" at 0:00). Snap every word into the sung section it belongs to.
  const secOf = (t) => sections.findIndex((s) => t >= s.t0 - 0.3 && t <= s.t1 + 0.3);
  for (const wd of words) {
    let k = secOf(wd.t1);
    if (k >= 0) {
      const s = sections[k];
      if (wd.t0 < s.t0 - 0.15) wd.t0 = Math.max(s.t0 - 0.05, wd.t1 - 3);
    } else if ((k = secOf(wd.t0)) >= 0) {
      wd.t1 = Math.max(wd.t0 + 0.08, Math.min(wd.t1, sections[k].t1 + 0.05));
    }
    if (wd.t1 - wd.t0 > 3) wd.t0 = wd.t1 - 3;          // no sung word lasts longer than ~3 s
    wd.sec = k;
  }
  // keep only words inside (or right next to) a sung section
  return words.filter((wd) => sections.some((s) => wd.t1 > s.t0 - 0.3 && wd.t0 < s.t1 + 0.3));
}

/**
 * Words → lines: break where the voice pauses (a gap between sung sections), on long gaps,
 * line-ending punctuation, after ~8 s, or every ~10 words.
 */
export function groupLines(words, sections = []) {
  const lines = [];
  let cur = null;
  words.forEach((wd, i) => {
    const prev = words[i - 1];
    const gap = prev ? wd.t0 - prev.t1 : Infinity;
    const punctBreak = prev && /[.!?;]$/.test(prev.w) && gap > 0.15;
    const voicePause = prev && wd.sec >= 0 && prev.sec >= 0 && wd.sec !== prev.sec &&
      sections[wd.sec] && sections[prev.sec] && sections[wd.sec].t0 - sections[prev.sec].t1 > 0.5;
    if (!cur || gap > 0.6 || punctBreak || voicePause || (/,$/.test(prev?.w || "") && gap > 0.35) ||
        cur.words.length >= 10 || wd.t1 - cur.t0 > 8) {
      cur = { words: [], t0: wd.t0, t1: wd.t1 };
      lines.push(cur);
    }
    cur.words.push(wd);
    cur.t1 = wd.t1;
  });
  for (const l of lines) l.text = l.words.map((w) => w.w).join(" ");
  return lines;
}

/**
 * One lyric line → viseme events at absolute song times.
 * Words keep Whisper's timing; inside a word the consonants keep spoken durations and the
 * vowels take the rest (sung vowels are long). Then a forced alignment against the song
 * model's per-frame visemes moves every mouth shape onto the actual sung sound.
 */
export function alignLine(line, pron, durations, frames, refine, fps = 100) {
  const plan = planUtterance(line.text, pron, durations);
  if (!plan.words.length) return null;
  // map plan words → Whisper words (same order; plan may split numbers etc.)
  const ww = line.words;
  const W = plan.words.length;
  plan.words.forEach((wd, i) => {
    const k = Math.min(ww.length - 1, Math.floor((i * ww.length) / W));
    const span = ww[k];
    const sub = plan.words.filter((_, j) => Math.min(ww.length - 1, Math.floor((j * ww.length) / W)) === k);
    const idx = sub.indexOf(wd), n = sub.length;
    wd.src = k;                                   // which heard (Whisper) word this spoken word belongs to
    wd.start = span.t0 + ((span.t1 - span.t0) * idx) / n;
    wd.end = span.t0 + ((span.t1 - span.t0) * (idx + 1)) / n;
  });
  const byWord = new Map();
  plan.phones.forEach((p) => { if (!byWord.has(p.word)) byWord.set(p.word, []); byWord.get(p.word).push(p); });
  plan.words.forEach((wd, wi) => {
    const ps = byWord.get(wi) || [];
    const span = Math.max(0.06, wd.end - wd.start);
    const isV = (p) => /^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)$/.test(p.p);
    let cons = 0, nv = 0;
    ps.forEach((p) => { if (isV(p)) nv++; else cons += Math.min(p.t1 - p.t0, 0.09); });
    const consScale = cons > span * 0.6 ? (span * 0.6) / cons : 1;
    const vowelTime = Math.max(0, span - cons * consScale);
    let t = wd.start;
    ps.forEach((p) => {
      const d = isV(p) ? (nv ? vowelTime / nv : 0) : Math.min(p.t1 - p.t0, 0.09) * consScale + (nv ? 0 : (span - cons * consScale) / ps.length);
      p.t0 = t; p.t1 = t + d; t += d;
    });
  });
  // rebuild events in a local window, refine on the audio, shift back to song time
  const ws = Math.max(0, line.t0 - 0.5), we = line.t1 + 0.5;
  const local = plan.phones.map((p) => ({ ...p, t0: p.t0 - ws, t1: p.t1 - ws }));
  const lplan = { phones: local, words: plan.words, visemes: phonesToVisemes(local, plan.words, we - ws) };
  const f0 = Math.floor(ws * fps), f1 = Math.min(frames.length, Math.ceil(we * fps));
  let refined = false;
  if (refine && f1 - f0 > lplan.visemes.length) refined = refine(lplan, frames.slice(f0, f1), fps, 0.6, 0.04);
  const ev = lplan.visemes.map((e) => ({ ...e, t0: e.t0 + ws, t1: e.t1 + ws }));
  return { events: ev, words: plan.words, refined, text: line.text, t0: line.t0, t1: line.t1 };
}

/** Merge per-line events into one timeline, keeping "sil" where no line is sung. */
export function mergeLines(aligned, duration) {
  const ev = [];
  let cursor = 0;
  for (const L of aligned) {
    const inner = L.events.filter((e) => e.v !== "sil" || (e.t0 > L.t0 && e.t1 < L.t1));
    if (!inner.length) continue;
    const a = Math.max(cursor, inner[0].t0);
    if (a - cursor > 0.02) ev.push({ v: "sil", t0: cursor, t1: a, stress: -1, word: -1 });
    for (const e of inner) {
      if (e.t1 <= cursor) continue;
      ev.push({ ...e, t0: Math.max(e.t0, cursor), line: L });
      cursor = e.t1;
    }
  }
  ev.push({ v: "sil", t0: cursor, t1: Math.max(duration, cursor + 0.1), stress: -1, word: -1 });
  return ev;
}

/** Is this viseme a held vowel (singers' open, sustained notes)? */
export const isVowelViseme = (v) => VOWEL_V.has(v);
export { VISEMES };
