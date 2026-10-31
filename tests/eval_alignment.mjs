// Lip-sync timing benchmark with exact ground truth (see training/make_align_eval.py).
//
// For every utterance we know the true viseme of each 10 ms frame. The browser is
// given what a timestamped TTS (ElevenLabs /with-timestamps) gives it — the text,
// the audio and per-character times — and we score the mouth timeline it builds:
//
//   A. timestamps only          (retimeWithAlignment)
//   B. timestamps + audio net   (retimeWithAlignment → refineWithAudio)
//   C. audio net alone          (what the Audio tab does without text)
//
//   node tests/eval_alignment.mjs [tests/align_eval.json]
import { readFileSync, writeFileSync } from "node:fs";
import { Audio2Viseme, G2P, parseModel } from "../pearl/nn.js";
import { FeatureExtractor, analyzeClip } from "../pearl/audio.js";
import { Durations, Lexicon, Pronouncer, VISEMES, planUtterance, refineWithAudio, retimeWithAlignment } from "../pearl/text.js";

const M = "models";
const load = (stem) => {
  const man = JSON.parse(readFileSync(`${M}/${stem}.json`, "utf8"));
  const b = readFileSync(`${M}/${man.bin}`);
  return parseModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const a2v = new Audio2Viseme(load("audio2viseme"));
const fx = new FeatureExtractor(a2v.featureConfig);
const pron = new Pronouncer(new Lexicon(readFileSync(`${M}/lexicon.txt`, "utf8")), new G2P(load("g2p")));
const durations = new Durations(JSON.parse(readFileSync(`${M}/durations.json`, "utf8")));
const data = JSON.parse(readFileSync(process.argv[2] || "tests/align_eval.json", "utf8"));

const norm = (w) => w.toLowerCase().replace(/[^a-z']/g, "");

/** Build an ElevenLabs-style character alignment for plan.spoken from word times. */
function charAlignment(plan, words) {
  const n = plan.spoken.length;
  const st = new Array(n).fill(null), en = new Array(n).fill(null);
  let j = 0;
  for (const wd of plan.words) {
    // match plan words to the TTS's words in order
    let k = j;
    while (k < words.length && norm(words[k].text) !== norm(wd.w)) k++;
    if (k >= words.length) return null;
    j = k + 1;
    const { start, end } = words[k], L = wd.charEnd - wd.charStart;
    for (let c = 0; c < L; c++) {
      st[wd.charStart + c] = start + ((end - start) * c) / L;
      en[wd.charStart + c] = start + ((end - start) * (c + 1)) / L;
    }
  }
  // spaces / punctuation take the gap between the words around them
  for (let c = 0; c < n; c++) if (st[c] === null) {
    let a = c - 1; while (a >= 0 && en[a] === null) a--;
    let b = c + 1; while (b < n && st[b] === null) b++;
    st[c] = a >= 0 ? en[a] : 0; en[c] = b < n ? st[b] : st[c];
  }
  return { characters: [...plan.spoken], character_start_times_seconds: st, character_end_times_seconds: en };
}

const frameLabels = (visemes, T, fps) => Array.from({ length: T }, (_, t) => {
  const tc = (t * 160 + 200) / 16000;     // frame centre, as in training labels
  const e = visemes.find((e) => tc >= e.t0 && tc < e.t1);
  return e ? VISEMES.indexOf(e.v) : 0;
});

function score(pred, gold) {
  let n = 0, exact = 0, tol = 0;
  for (let t = 0; t < gold.length; t++) {
    if (gold[t] === 0 && pred[t] === 0) continue;        // both silent: not informative
    n++;
    if (pred[t] === gold[t]) exact++;
    // ±20 ms: the predicted viseme appears in gold within 2 frames
    for (let d = -2; d <= 2; d++) if (gold[t + d] === pred[t]) { tol++; break; }
  }
  // bilabial closures (P/B/M): gold PP runs, and how far the predicted run's centre is
  const runs = (seq) => { const r = []; for (let t = 0; t < seq.length; t++) if (seq[t] === 1 && seq[t - 1] !== 1) { let e = t; while (seq[e + 1] === 1) e++; r.push((t + e) / 2); } return r; };
  const g = runs(gold), p = runs(pred);
  const off = g.map((c) => Math.min(...p.map((q) => Math.abs(q - c)), 99));
  return { n, exact, tol, pp: g.length, ppHit: off.filter((o) => o <= 3).length, ppOff: off.filter((o) => o <= 8) };
}

// deterministic gaussian noise for the "imperfect timestamps" condition
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const JITTER = 0.04;

const methods = { "timestamps only": [], "timestamps + audio net": [],
  "noisy timestamps only": [], "noisy timestamps + audio": [], "audio net alone": [] };
let used = 0;
for (const u of data.utts) {
  const pcm = Float32Array.from(new Int16Array(Buffer.from(u.pcm16_b64, "base64").buffer.slice(0)), (v) => v / 32768);
  const plan0 = planUtterance(u.text, pron, durations);
  const al = charAlignment(plan0, u.words);
  if (!al) { console.log(`skip (word mismatch): ${u.text}`); continue; }
  used++;
  const an = await analyzeClip(a2v, fx, pcm);
  const T = Math.min(u.vis.length, an.frames.length);
  const gold = u.vis.slice(0, T);
  const run = (words, nameA, nameB) => {
    const plan = planUtterance(u.text, pron, durations);
    retimeWithAlignment(plan, charAlignment(plan, words));
    methods[nameA].push(score(frameLabels(plan.visemes, T), gold));
    refineWithAudio(plan, an.frames);
    methods[nameB].push(score(frameLabels(plan.visemes, T), gold));
  };
  run(u.words, "timestamps only", "timestamps + audio net");
  // imperfect timestamps: every word boundary moved by N(0, 40 ms), order kept
  const noisy = u.words.map((w) => ({ ...w }));
  const b = [noisy[0].start, ...noisy.map((w) => w.end)].map((t) => t + JITTER * gauss());
  for (let i = 1; i < b.length; i++) b[i] = Math.max(b[i], b[i - 1] + 0.03);
  noisy.forEach((w, i) => { w.start = Math.max(0, b[i]); w.end = b[i + 1]; });
  run(noisy, "noisy timestamps only", "noisy timestamps + audio");
  methods["audio net alone"].push(score(an.frames.slice(0, T).map((p) => p.indexOf(Math.max(...p))), gold));
}

const summary = {};
console.log(`\n${used} utterances from voices never used in training (${[...new Set(data.utts.map((u) => u.voice))].join(", ")})\n`);
console.log("method                      exact   ±20ms   P/B/M closures within 30 ms   median closure error");
for (const [name, rs] of Object.entries(methods)) {
  const s = rs.reduce((a, r) => ({ n: a.n + r.n, exact: a.exact + r.exact, tol: a.tol + r.tol, pp: a.pp + r.pp, ppHit: a.ppHit + r.ppHit, off: a.off.concat(r.ppOff) }), { n: 0, exact: 0, tol: 0, pp: 0, ppHit: 0, off: [] });
  const med = s.off.sort((a, b) => a - b)[Math.floor(s.off.length / 2)] ?? NaN;
  summary[name] = { exact: +(s.exact / s.n).toFixed(4), tol20ms: +(s.tol / s.n).toFixed(4), closures_within_30ms: +(s.ppHit / s.pp).toFixed(4), median_closure_error_ms: med * 10 };
  console.log(`${name.padEnd(26)}  ${(100 * s.exact / s.n).toFixed(1).padStart(5)}%  ${(100 * s.tol / s.n).toFixed(1).padStart(5)}%   ${String(s.ppHit).padStart(3)}/${s.pp} (${(100 * s.ppHit / s.pp).toFixed(0)}%)`.padEnd(80) + `${(med * 10).toFixed(0)} ms`);
}
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify({ utterances: used, ...summary }, null, 2));
