// Verifies that the browser runtime (pearl/*.js) reproduces the trained models
// exactly, using the reference vectors the Python trainers export.
//   node tests/verify_models.mjs [models_dir] [vectors_dir]
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Audio2Viseme, EmotionModel, G2P, parseModel } from "../pearl/nn.js";
import { FeatureExtractor } from "../pearl/audio.js";
import { Lexicon, Pronouncer, Durations, planUtterance } from "../pearl/text.js";

const M = process.argv[2] || "models";
const V = process.argv[3] || "tests/vectors";
const load = (stem) => {
  const man = JSON.parse(readFileSync(join(M, `${stem}.json`), "utf8"));
  const b = readFileSync(join(M, man.bin));
  return parseModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
let failures = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`); if (!ok) failures++; };

// ── audio → viseme ──
{
  const a2v = new Audio2Viseme(load("audio2viseme"));
  const tv = JSON.parse(readFileSync(join(V, "audio2viseme_test.json"), "utf8"));
  const pcm = Float32Array.from(tv.pcm16, (v) => v / 32768);
  const fx = new FeatureExtractor(a2v.featureConfig);
  const f = fx.compute(pcm);
  let fd = 0;
  tv.feats_norm_first5.forEach((row, m) => row.forEach((v, t) => { fd = Math.max(fd, Math.abs(f.data[m * f.T + t] - v)); }));
  check("log-mel features", fd < 2e-3, `max |Δ| = ${fd.toExponential(2)} over ${f.T} frames`);
  const t0 = performance.now();
  const out = a2v.forward(f.data, f.T);
  const ms = performance.now() - t0;
  let ld = 0, agree = 0;
  const T = f.T, nv = a2v.visemes.length;
  for (let t = 0; t < T; t++) {
    let best = 0;
    for (let k = 0; k < nv; k++) {
      ld = Math.max(ld, Math.abs(out.vis[k * T + t] - tv.vis_logits[k][t]));
      if (out.vis[k * T + t] > out.vis[best * T + t]) best = k;
    }
    if (best === tv.argmax[t]) agree++;
  }
  check("audio→viseme logits", ld < 5e-2 && agree / T > 0.99, `max |Δlogit| = ${ld.toExponential(2)}, argmax agreement ${(100 * agree / T).toFixed(1)}%, ${T} frames in ${ms.toFixed(0)} ms (${(T / 100 / (ms / 1000)).toFixed(0)}× realtime)`);
}

// ── G2P ──
{
  const g2p = new G2P(load("g2p"));
  const tv = JSON.parse(readFileSync(join(V, "g2p_test.json"), "utf8"));
  let ok = 0, n = 0;
  for (const [w, ref] of Object.entries(tv)) {
    const got = g2p.phonemize(w);
    const same = got.join(" ") === ref.phones.join(" ");
    ok += same; n++;
    console.log(`      ${w.padEnd(16)} ${got.join(" ")}${same ? "" : `   ≠ python: ${ref.phones.join(" ")}`}`);
  }
  check("G2P phones", ok === n, `${ok}/${n} words identical to PyTorch`);
}

// ── emotion ──
{
  const emo = new EmotionModel(load("emotion"));
  const tv = JSON.parse(readFileSync(join(V, "emotion_test.json"), "utf8"));
  let md = 0, top = 0;
  for (const c of tv) {
    const p = emo.predict(c.text);
    const arr = emo.labels.map((l) => p[l]);
    c.probs.forEach((v, i) => (md = Math.max(md, Math.abs(arr[i] - v))));
    const best = emo.labels[arr.indexOf(Math.max(...arr))];
    top += best === c.top;
    console.log(`      ${best.padEnd(9)} ${(Math.max(...arr) * 100).toFixed(0).padStart(3)}%  ${c.text}`);
  }
  check("emotion probabilities", md < 2e-3 && top === tv.length, `max |Δp| = ${md.toExponential(2)}, top-1 ${top}/${tv.length}`);
}

// ── text pipeline end-to-end ──
{
  const lex = new Lexicon(readFileSync(join(M, "lexicon.txt"), "utf8"));
  const g2p = new G2P(load("g2p"));
  const pron = new Pronouncer(lex, g2p);
  const dur = new Durations(JSON.parse(readFileSync(join(M, "durations.json"), "utf8")));
  const plan = planUtterance("Hello Dr. Musoke, it's 2026 and Zorblax costs $45.99!", pron, dur);
  console.log(`      spoken: "${plan.spoken}"`);
  console.log(`      visemes: ${plan.visemes.map((v) => v.v).join(" ")}`);
  console.log(`      duration ${plan.duration.toFixed(2)} s, ${plan.words.length} words (${pron.stats.lexicon} lexicon, ${pron.stats.neural} neural)`);
  check("text → viseme timeline", plan.words.length > 10 && plan.visemes.length > 30 && pron.stats.neural >= 1, `${lex.size} lexicon entries`);
}

process.exit(failures ? 1 : 0);
