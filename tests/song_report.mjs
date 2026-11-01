// What does Miss Pearl hear in a song? Runs the speech model and the song model on an audio
// file and prints where each thinks a voice is singing, plus the beat.
//   node tests/song_report.mjs path/to/song.mp3 [models_dir]      (needs ffmpeg on PATH)
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Audio2Viseme, parseModel } from "../pearl/nn.js";
import { FeatureExtractor, analyzeClip } from "../pearl/audio.js";
import { trackBeats, voiceSections } from "../pearl/song.js";

const file = process.argv[2];
const M = process.argv[3] || "models";
if (!file) { console.log("usage: node tests/song_report.mjs song.mp3 [models_dir]"); process.exit(1); }
const load = (stem) => {
  const man = JSON.parse(readFileSync(join(M, `${stem}.json`), "utf8"));
  const b = readFileSync(join(M, man.bin));
  return new Audio2Viseme(parseModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
};
const raw = execFileSync("ffmpeg", ["-v", "error", "-i", file, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"], { maxBuffer: 1 << 30 });
const s16 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
console.log(`${file.split(/[\\/]/).pop()}: ${fmt(s16.length / 16000)}`);

for (const stem of ["audio2viseme", "sing2viseme"]) {
  let model;
  try { model = load(stem); } catch { console.log(`\n${stem}: not found in ${M}`); continue; }
  const fx = new FeatureExtractor(model.featureConfig);
  const t0 = performance.now();
  const an = await analyzeClip(model, fx, s16);
  const ms = performance.now() - t0;
  const voice = an.voice || Float32Array.from(an.frames, (p) => 1 - p[0]);
  const { sections } = voiceSections(voice);
  const sung = sections.reduce((s, x) => s + x.t1 - x.t0, 0);
  const closed = an.frames.filter((p) => p.indexOf(Math.max(...p)) === 0).length / an.frames.length;
  console.log(`\n${stem} (${an.voice ? "voice head" : "no voice head: 'not silent' used as voice"}) — ${(ms / 1000).toFixed(1)} s, ` +
    `${(an.frames.length / 100 / (ms / 1000)).toFixed(0)}× realtime`);
  console.log(`  voice in ${sections.length} part(s), ${fmt(sung)} of singing; mouth-closed frames ${(100 * closed).toFixed(0)}%`);
  console.log("  " + sections.slice(0, 14).map((s) => `${fmt(s.t0)}–${fmt(s.t1)}`).join("  ") + (sections.length > 14 ? " …" : ""));
  if (stem === "sing2viseme" || !an.voice) {
    const bt = trackBeats(an.feats.data, an.feats.n, an.frames.length);
    if (stem === "sing2viseme") console.log(`  beat: ${bt.tempo} BPM, ${bt.beats.length} beats`);
  }
}
