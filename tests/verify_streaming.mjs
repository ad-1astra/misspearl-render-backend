// The live-mic path (StreamingVisemes) must give the same per-frame visemes as
// analysing the whole clip at once. Feeds the reference clip in mic-sized chunks.
import { readFileSync } from "node:fs";
import { Audio2Viseme, parseModel } from "../pearl/nn.js";
import { FeatureExtractor, StreamingVisemes } from "../pearl/audio.js";
const man = JSON.parse(readFileSync("models/audio2viseme.json", "utf8"));
const b = readFileSync("models/audio2viseme.bin");
const a2v = new Audio2Viseme(parseModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
const fx = new FeatureExtractor(a2v.featureConfig);
const tv = JSON.parse(readFileSync("tests/vectors/audio2viseme_test.json", "utf8"));
let pcm = Float32Array.from(tv.pcm16, (v) => v / 32768);
// repeat the clip to ~6 s so the rolling buffer has to drop old audio
const rep = new Float32Array(pcm.length * 4); for (let i = 0; i < 4; i++) rep.set(pcm, i * pcm.length); pcm = rep;
const full = fx.compute(pcm); const ref = a2v.probs(a2v.forward(full.data, full.T));
const got = [];
const sv = new StreamingVisemes(a2v, fx, (p, rms, abs) => { got[abs] = p; });
for (let i = 0; i < pcm.length; i += 2048) sv.push(pcm.subarray(i, Math.min(pcm.length, i + 2048)));
let n = 0, maxd = 0, agree = 0;
for (let t = 0; t < got.length; t++) {
  if (!got[t]) continue;
  n++;
  let a = 0, b2 = 0;
  for (let k = 0; k < ref[t].length; k++) { maxd = Math.max(maxd, Math.abs(got[t][k] - ref[t][k])); if (got[t][k] > got[t][a]) a = k; if (ref[t][k] > ref[t][b2]) b2 = k; }
  agree += a === b2;
}
const ok = n > full.T - a2v.context - 5 && maxd < 1e-4 && agree === n;
console.log(`${ok ? "PASS" : "FAIL"}  streaming == offline: ${n}/${full.T} frames emitted (${a2v.context} held back for look-ahead), max |Δp| ${maxd.toExponential(2)}, argmax agreement ${agree}/${n}`);
process.exit(ok ? 0 : 1);
