// Miss Pearl — speech performer: turns (tagged) text or audio into a timed
// performance of mouth shapes, emotions and gestures.
//
// Engines
//   browser  : SpeechSynthesis. Text-driven viseme timeline, re-anchored at
//              every word-boundary event and rate-calibrated per voice.
//   backend  : ElevenLabs via app.py /tts/timed. Character-level alignment
//              gives each word its true start/end; the audio→viseme network
//              then places every mouth shape on the real audio (forced alignment).
//   audio    : any audio (TTS without timings, files, mic). The trained
//              audio→viseme network reads the mouth shapes from the sound.

import { parseTags, planUtterance, refineWithAudio, retimeWithAlignment, splitSentences, wordAtChar } from "./text.js?v=20261031b";
import { blendVisemeProbs, sampleTimeline, VISEME_SHAPES } from "./face.js?v=20261031b";
import { analyzeClip, loudness, to16kMono } from "./audio.js?v=20261031b";

const now = () => performance.now() / 1000;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const EMO_KEYS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"];

export class Performer {
  constructor({ face, pron, durations, emotionModel, a2v, fx, hooks = {} }) {
    Object.assign(this, { face, pron, durations, emotionModel, a2v, fx, hooks });
    this.lead = 0.045;           // lips lead the sound slightly (perceptually tighter)
    this.active = null;          // current driver
    this.queueId = 0;
    this.voiceSpeed = (() => { try { return JSON.parse(localStorage.getItem("pearl.voiceSpeed") || "{}"); } catch { return {}; } })();
    this.audioCtx = null;
    this.analyser = null;
  }

  ctx() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.connect(this.audioCtx.destination);
    }
    if (this.audioCtx.state === "suspended") this.audioCtx.resume();
    return this.audioCtx;
  }

  // ── emotion understanding ─────────────────────────────────
  readEmotion(sentence, marks = []) {
    const tagEmo = marks.find((m) => !["laugh", "wink", "nod", "blush", "smirk"].includes(m.tag));
    let probs = null, source = "model";
    if (this.emotionModel) probs = this.emotionModel.predict(sentence);
    const target = {};
    // a gesture tag with no emotion tag still says how the sentence feels
    const GESTURE_MOOD = { laugh: ["joy", 0.9], wink: ["joy", 0.5], blush: ["love", 0.6] };
    const gestureMood = !tagEmo && marks.map((m) => GESTURE_MOOD[m.tag]).find(Boolean);
    if (tagEmo) {
      source = "llm-tag";
      target[tagEmo.tag] = 0.9;
    } else if (gestureMood) {
      source = "llm-tag";
      target[gestureMood[0]] = gestureMood[1];
    } else if (probs) {
      const emotional = 1 - probs.neutral;
      for (const k of EMO_KEYS) {
        const p = probs[k];
        if (p > 0.12) target[k === "sadness" ? "sadness" : k] = clamp(p * 1.25 * (0.6 + emotional * 0.6), 0, 1);
      }
      if (Object.keys(target).length === 0) target.neutral = 1;
    } else {
      target.neutral = 1;
      source = "none";
    }
    const t = sentence.trim();
    if (/!\s*$/.test(t)) {
      if (target.joy) { target.excited = Math.max(target.excited || 0, target.joy * 0.7); }
      else if (!target.anger && !target.sadness) target.excited = 0.4;
    }
    if (/\?\s*$/.test(t) && !target.anger && !target.sadness) target.curious = Math.max(target.curious || 0, 0.45);
    if (marks.some((m) => m.tag === "smirk")) target.thinking = 0.35;
    return { target, probs, source, gestures: marks.filter((m) => ["laugh", "wink", "nod", "blush"].includes(m.tag)) };
  }

  /** Text (optionally with [emotion] tags) → sentences with plans, emotions and gestures. */
  prepare(text) {
    const { clean, marks } = parseTags(text);
    const sents = splitSentences(clean);
    if (!sents.length && clean) sents.push({ text: clean, start: 0 });
    return {
      clean, marks,
      sentences: sents.map((s, i) => {
        const end = i + 1 < sents.length ? sents[i + 1].start : Infinity;
        const sm = marks.filter((m) => m.pos >= s.start && m.pos < end).map((m) => ({ ...m, rel: m.pos - s.start }));
        // a tag right at the very end of the previous sentence belongs to this one
        return { ...s, marks: sm, emo: this.readEmotion(s.text, sm) };
      }),
    };
  }

  stop() {
    this.queueId++;
    if (window.speechSynthesis) speechSynthesis.cancel();
    if (this.active && this.active.src) { try { this.active.src.stop(); } catch {} }
    this.active = null;
    this.face.setMouth({ ...VISEME_SHAPES.sil });
  }

  // ── per-frame: compute the mouth for the current driver ───
  tick() {
    const d = this.active;
    if (!d) return null;
    let info = null;
    if (d.kind === "timeline") {
      const t = d.clock() + this.lead;
      const { shape, active } = sampleTimeline(d.plan.visemes, t, d.hint || 0);
      if (active) {
        d.hint = Math.max(0, d.plan.visemes.indexOf(active) - 2);
        if (active !== d.lastEvent) {
          d.lastEvent = active;
          const wd = d.plan.words[active.word];
          if (active.stress === 1 && wd && !wd.func && !active.glide && (wd.emph || Math.random() < 0.35)) this.face.emphasis(wd.emph ? 1 : 0.45);
        }
      }
      let gain = 1;
      if (d.energy) {
        const fi = clamp(Math.floor((t - (d.energyOffset || 0)) * 100), 0, d.energy.length - 1);
        gain = 0.55 + 0.6 * d.energy[fi];
      }
      this.face.setMouth(shape, gain);
      const wi = active ? active.word : -1;
      if (wi !== d.lastWord && wi >= 0) { d.lastWord = wi; d.onWord && d.onWord(wi); }
      for (const g of d.events || []) {
        if (!g.done && t >= g.t) { g.done = true; g.fire(); }
      }
      info = { viseme: active ? active.v : "sil", word: wi >= 0 ? d.plan.words[wi].w : "", phone: active ? active.v : "",
               source: d.source, sync: d.syncLabel ? d.syncLabel() : "" };
    } else if (d.kind === "frames") {
      const t = this.ctx().currentTime - d.start + this.lead;
      const f = t * d.fps;
      const i0 = clamp(Math.floor(f), 0, d.frames.length - 1), i1 = Math.min(i0 + 1, d.frames.length - 1), a = clamp(f - i0, 0, 1);
      const p = d.frames[i0].map((v, k) => v * (1 - a) + d.frames[i1][k] * a);
      const e = d.energy[i0] * (1 - a) + d.energy[i1] * a;
      this.face.setMouth(blendVisemeProbs(p), 0.45 + 0.75 * e);
      let best = 0; p.forEach((v, k) => { if (v > p[best]) best = k; });
      info = { viseme: this.a2v.visemes[best], probs: p, source: "neural audio→viseme", sync: `frame ${i0}` };
      if (t > d.duration + 0.2) this._finish(d);
    } else if (d.kind === "stream") {
      const fr = d.latest;
      if (fr) {
        this.face.setMouth(blendVisemeProbs(fr.p), fr.gain);
        let best = 0; fr.p.forEach((v, k) => { if (v > fr.p[best]) best = k; });
        info = { viseme: this.a2v.visemes[best], probs: fr.p, source: "neural audio→viseme (live)", sync: `${Math.round(d.latency * 1000)} ms latency` };
      }
    }
    return info;
  }

  _finish(d) {
    if (this.active === d) {
      this.active = null;
      this.face.setMouth({ ...VISEME_SHAPES.sil });
    }
    d.resolve && d.resolve();
  }

  _applySentenceMood(s) {
    console.debug(`[pearl] sentence "${s.text.slice(0, 48)}" → ${JSON.stringify(s.emo.target)} (${s.emo.source})`);
    this.face.setEmotion(s.emo.target);
    this.hooks.onEmotion && this.hooks.onEmotion(s.emo);
  }

  // ── Engine: browser SpeechSynthesis ───────────────────────
  async speakBrowser(text, opts = {}) {
    const id = ++this.queueId;
    const perf = this.prepare(text);
    this.hooks.onPerformance && this.hooks.onPerformance(perf);
    for (let si = 0; si < perf.sentences.length; si++) {
      if (id !== this.queueId) return;
      const s = perf.sentences[si];
      const plan = planUtterance(s.text, this.pron, this.durations);
      if (!plan.words.length) continue;
      this._applySentenceMood(s);
      await this._utter(plan, s, si, opts, id);
    }
    if (id === this.queueId) this.hooks.onDone && this.hooks.onDone();
  }

  _gestureEvents(plan, s) {
    return s.marks.map((m) => {
      // word whose token starts at/after the tag position
      let wi = plan.words.findIndex((w) => plan.tokens[w.tok].start >= m.rel);
      if (wi < 0) wi = plan.words.length - 1;
      const t = Math.max(0, plan.words[wi].start - 0.15);
      return { t, fire: () => {
        if (["laugh", "wink", "nod", "blush"].includes(m.tag)) this.face.gesture(m.tag);
        else this.face.setEmotion({ [m.tag]: 0.9 });
      } };
    });
  }

  _utter(plan, s, si, opts, id) {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis;
      if (!synth) { resolve(); return; }
      const u = new SpeechSynthesisUtterance(plan.spoken);
      const voice = opts.voice || null;
      if (voice) u.voice = voice;
      u.rate = opts.rate || 1; u.pitch = opts.pitch || 1.1; u.lang = voice ? voice.lang : "en-US";
      const vkey = (voice && voice.name) || "default";
      const clock = { real0: null, plan0: 0, speed: (this.voiceSpeed[vkey] || 1) * u.rate, boundaries: 0, last: null, next: Infinity };
      const d = {
        kind: "timeline", plan, source: "browser voice + text timeline", hint: 0, lastWord: -1,
        events: this._gestureEvents(plan, s),
        clock: () => {
          if (clock.real0 == null) return -1;
          let t = clock.plan0 + (now() - clock.real0) * clock.speed;
          if (clock.boundaries > 0) t = Math.min(t, clock.next + 0.05);   // wait for the next word event
          return t;
        },
        syncLabel: () => clock.boundaries ? `word-synced (${clock.boundaries} anchors, ×${clock.speed.toFixed(2)})` : `rate-calibrated ×${clock.speed.toFixed(2)}`,
        onWord: (wi) => this.hooks.onWord && this.hooks.onWord(si, wi, plan),
        resolve,
      };
      const started = now();
      u.onstart = () => { clock.real0 = now(); this.face.setState("speaking"); };
      u.onboundary = (e) => {
        if (e.name && e.name !== "word") return;
        const wi = wordAtChar(plan, e.charIndex);
        if (wi < 0) return;
        const t = now();
        if (clock.real0 == null) clock.real0 = t;
        if (clock.last && wi > clock.last.wi) {
          const dr = t - clock.last.t, dp = plan.words[wi].start - plan.words[clock.last.wi].start;
          if (dr > 0.05 && dp > 0.05) clock.speed = 0.6 * clock.speed + 0.4 * clamp(dp / dr, 0.4, 2.5);
        }
        clock.plan0 = plan.words[wi].start; clock.real0 = t;
        clock.last = { t, wi }; clock.boundaries++;
        clock.next = wi + 1 < plan.words.length ? plan.words[wi + 1].start : plan.duration;
      };
      const end = (ev) => {
        if (d.ended) return;
        d.ended = true;
        console.debug(`[pearl] utterance done: ${plan.words.length} words, planned ${plan.duration.toFixed(2)} s, real ${(now() - (clock.real0 ?? started)).toFixed(2)} s, ${clock.boundaries} word anchors, speed ×${clock.speed.toFixed(2)}${ev && ev.error ? ", error " + ev.error : ""}`);
        if (clock.real0 != null && clock.boundaries === 0) {
          const real = now() - clock.real0;
          if (real > 0.4) {
            const f = clamp(plan.duration / real / u.rate, 0.5, 2);
            this.voiceSpeed[vkey] = 0.5 * (this.voiceSpeed[vkey] || 1) + 0.5 * f;
            try { localStorage.setItem("pearl.voiceSpeed", JSON.stringify(this.voiceSpeed)); } catch {}
          }
        }
        this._finish(d);
      };
      u.onend = end; u.onerror = end;
      this.active = d;
      synth.speak(u);
      // some voices never fire onstart: start the clock anyway
      setTimeout(() => { if (clock.real0 == null && this.active === d) clock.real0 = now(); }, 350);
      // safety: never hang if the engine drops the utterance
      setTimeout(() => { if (this.active === d && now() - started > plan.duration * 3 + 5) end(); }, (plan.duration * 3 + 6) * 1000);
    });
  }

  // ── Engine: backend ElevenLabs with timestamps ────────────
  async speakBackend(text, api, voiceId = "") {
    const id = ++this.queueId;
    const perf = this.prepare(text);
    this.hooks.onPerformance && this.hooks.onPerformance(perf);
    const plan = planUtterance(perf.clean, this.pron, this.durations);
    const res = await fetch(`${api}/tts/timed`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: plan.spoken, voice_id: voiceId || "" }) });
    if (!res.ok) {
      let why = `tts/timed ${res.status}`;
      try { why = (await res.json()).error || why; } catch {}
      throw new Error(why);
    }
    const data = await res.json();
    if (id !== this.queueId) return;
    const bytes = Uint8Array.from(atob(data.audio_base64), (c) => c.charCodeAt(0));
    const ctx = this.ctx();
    const buf = await ctx.decodeAudioData(bytes.buffer);
    if (data.alignment) retimeWithAlignment(plan, data.alignment);
    // let the trained audio→viseme network place every mouth shape on the real audio
    if (this.a2v && this.fx) {
      try {
        const t0 = performance.now();
        const an = await analyzeClip(this.a2v, this.fx, await to16kMono(buf));
        if (id !== this.queueId) return;
        const ok = refineWithAudio(plan, an.frames, an.fps);
        console.debug(`[pearl] audio refinement ${ok ? "applied" : "skipped"}: ${an.frames.length} frames in ${Math.round(performance.now() - t0)} ms`);
      } catch (e) { console.warn("[pearl] audio refinement failed", e); }
    }
    const mono = buf.getChannelData(0);
    const energy = loudness(mono, Math.round(buf.sampleRate / 100), Math.round(buf.sampleRate / 40));
    // sentence moods at their real start times
    const events = [];
    let wcursor = 0;
    perf.sentences.forEach((s, si) => {
      const sp = planUtterance(s.text, this.pron, this.durations);
      const w0 = wcursor; wcursor += sp.words.length;
      const t0 = plan.words[w0] ? plan.words[w0].start - 0.2 : 0;
      events.push({ t: Math.max(0, t0), fire: () => this._applySentenceMood(s) });
      for (const ge of this._gestureEvents(sp, s)) {
        const wi = Math.min(w0 + Math.max(0, sp.words.findIndex((w) => w.start >= ge.t)), plan.words.length - 1);
        events.push({ t: Math.max(0, (plan.words[wi] ? plan.words[wi].start : 0) - 0.15), fire: ge.fire });
      }
    });
    this.lastVoice = data.voice_name || "";
    return this._playTimeline(plan, buf, energy, events, `ElevenLabs${data.voice_name ? ` (${data.voice_name})` : ""} + character alignment`, id);
  }

  _playTimeline(plan, buf, energy, events, source, id) {
    return new Promise((resolve) => {
      const ctx = this.ctx();
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.analyser);
      const start = ctx.currentTime + 0.05;
      const d = { kind: "timeline", plan, src, energy, energyOffset: 0, events, source, hint: 0, lastWord: -1,
        clock: () => ctx.currentTime - start,
        syncLabel: () => plan.refined ? "TTS timestamps + audio network" : plan.aligned ? "TTS timestamps" : "audio clock",
        onWord: (wi) => this.hooks.onWord && this.hooks.onWord(-1, wi, plan), resolve };
      src.onended = () => { this._finish(d); if (id === this.queueId) this.hooks.onDone && this.hooks.onDone(); };
      this.active = d;
      this.face.setState("speaking");
      src.start(start);
    });
  }

  // ── Engine: any audio through the trained network ─────────
  async playAudio(arrayBuffer, { text = "", onProgress } = {}) {
    if (!this.a2v) throw new Error("audio model not loaded");
    const id = ++this.queueId;
    const ctx = this.ctx();
    const buf = await ctx.decodeAudioData(arrayBuffer);
    const s16 = await to16kMono(buf);
    const t0 = performance.now();
    const an = await analyzeClip(this.a2v, this.fx, s16, onProgress);
    const ms = performance.now() - t0;
    if (id !== this.queueId) return;
    if (text) { const p = this.prepare(text); if (p.sentences[0]) this._applySentenceMood(p.sentences[0]); }
    return new Promise((resolve) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.analyser);
      const start = ctx.currentTime + 0.05;
      const d = { kind: "frames", src, start, fps: an.fps, frames: an.frames, energy: an.energy, duration: an.duration, resolve, analysisMs: ms };
      src.onended = () => this._finish(d);
      this.active = d;
      this.face.setState("speaking");
      this.hooks.onAnalysis && this.hooks.onAnalysis({ frames: an.frames.length, ms, duration: an.duration });
      src.start(start);
    });
  }

  /** Live microphone: frames arrive from StreamingVisemes. */
  startStream() {
    const d = { kind: "stream", latest: null, latency: 0.35, peak: 0.02 };
    this.active = d;
    this.face.setState("speaking");
    return {
      push: (p, rms) => {
        d.peak = Math.max(d.peak * 0.995, rms);
        const g = clamp(rms / (d.peak * 0.6 + 1e-4), 0, 1.3);
        d.latest = { p, gain: rms < 0.004 ? 0 : 0.35 + 0.7 * g };
      },
      stop: () => this._finish(d),
      driver: d,
    };
  }
}
