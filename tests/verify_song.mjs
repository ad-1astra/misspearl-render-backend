// Checks the song logic in pearl/song.js on signals with a known answer:
//   beat tracking (synthetic drum patterns at known tempos), sung-section detection,
//   and the clean-up of Whisper word timestamps.
//   node tests/verify_song.mjs
import { cleanWords, groupLines, trackBeats, voiceSections } from "../pearl/song.js";

let failures = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`); if (!ok) failures++; };

// deterministic noise
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

// ── beat tracking ──
function drumFeats(bpm, seconds, { offset = 0.37, swing = 0, nMels = 80 } = {}) {
  const T = seconds * 100, f = new Float32Array(nMels * T);
  for (let m = 0; m < nMels; m++) for (let t = 0; t < T; t++) f[m * T + t] = 0.3 * (rnd() - 0.5);
  const truth = [];
  const P = 60 / bpm;
  for (let k = 0, tb = offset; tb < seconds - 0.1; k++, tb = offset + k * P + (k % 2 ? swing : 0)) {
    truth.push(tb);
    const t0 = Math.round(tb * 100);
    const kick = k % 2 === 0;
    for (let d = 0; d < 12 && t0 + d < T; d++) {
      const e = Math.exp(-d / 3);
      for (let m = 0; m < nMels; m++) {
        const low = m < 20;
        if (kick ? low : m > 30) f[m * T + t0 + d] += (kick ? 3 : 2) * e;
      }
    }
    // off-beat hi-hat
    const th = Math.round((tb + P / 2) * 100);
    for (let d = 0; d < 4 && th + d < T; d++) for (let m = 55; m < nMels; m++) f[m * T + th + d] += 0.8 * Math.exp(-d / 1.5);
  }
  return { f, T, truth, nMels };
}
for (const bpm of [92, 120, 140]) {
  const { f, T, truth, nMels } = drumFeats(bpm, 30);
  const r = trackBeats(f, nMels, T);
  const hits = truth.filter((tb) => r.beats.some((b) => Math.abs(b.t - tb) <= 0.05)).length;
  const tempoOk = [1, 2, 0.5].some((m) => Math.abs(r.tempo - bpm * m) / (bpm * m) < 0.04);
  check(`beat tracking ${bpm} BPM`, tempoOk && hits / truth.length > 0.9,
    `tempo ${r.tempo} BPM, ${hits}/${truth.length} beats found within 50 ms, ${r.beats.length} beats placed`);
}

// ── sung sections from voice probabilities ──
{
  const T = 4000, v = new Float32Array(T);
  const truth = [[3.2, 9.8], [12.0, 20.5], [27.0, 36.0]];
  for (let t = 0; t < T; t++) {
    const s = t / 100;
    const inside = truth.some(([a, b]) => s >= a && s < b);
    v[t] = inside ? 0.85 + 0.1 * rnd() : 0.08 + 0.1 * rnd();
    if (inside && rnd() < 0.03) v[t] = 0.3;                 // brief dips (consonants, breaths)
  }
  const { sections } = voiceSections(v);
  const ok = sections.length === truth.length &&
    sections.every((s, i) => Math.abs(s.t0 - truth[i][0]) < 0.06 && Math.abs(s.t1 - truth[i][1]) < 0.06);
  check("sung sections", ok, sections.map((s) => `${s.t0.toFixed(2)}–${s.t1.toFixed(2)} s`).join(", "));
}

// ── Whisper word clean-up ──
{
  const chunks = [
    { text: " Hello", timestamp: [10.0, 10.4] }, { text: " world", timestamp: [10.4, 11.0] },
    { text: " [Music]", timestamp: [12, 20] },
    { text: " again", timestamp: [29.98, 29.98] },             // phantom at the 30 s seam
    { text: " late", timestamp: [33.0, 33.5] }, { text: " early", timestamp: [31.0, 31.4] },   // out of order
    { text: " ghost", timestamp: [50.0, 50.6] },              // outside every sung section
  ];
  const w = cleanWords(chunks, [{ t0: 9.5, t1: 40 }]);
  const got = w.map((x) => x.w).join(" ");
  check("lyric word clean-up", got === "Hello world late" && w.every((x, i) => x.t1 > x.t0 && (!i || x.t0 >= w[i - 1].t0)), `kept: "${got}"`);
  // a first word stretched back across a silent intro snaps to where the singing starts;
  // lines break where the voice pauses
  const secs = [{ t0: 12.4, t1: 17.5 }, { t0: 20.6, t1: 23 }];
  const w2 = cleanWords([{ text: " I", timestamp: [0, 12.7] }, { text: " sing", timestamp: [12.7, 13.4] },
    { text: " today", timestamp: [13.4, 20.9] }, { text: " again", timestamp: [20.9, 21.6] }], secs);
  const lines = groupLines(w2, secs);
  check("lyric words snap to the singing", Math.abs(w2[0].t0 - 12.35) < 0.01 && w2[2].t1 - w2[2].t0 <= 3 && lines.length === 2,
    `first word starts ${w2[0].t0.toFixed(2)} s (singing starts 12.40 s); ${lines.length} lines: ${lines.map((l) => `${l.t0.toFixed(1)}–${l.t1.toFixed(1)}`).join(", ")}`);
}

if (failures) { console.log(`\n${failures} check(s) failed`); process.exit(1); }
console.log("\nall song checks passed");
