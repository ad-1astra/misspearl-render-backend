// Miss Pearl — expressive SVG face rig.
//
// Layers (all blended every frame, then smoothed with critically-damped springs):
//   1. Mouth: viseme shapes (Meta/Oculus 15) blended by a coarticulation model
//   2. Emotion: 7 trained emotions + warm/curious/thinking/concern/proud mixes
//   3. Behaviour: blinks, micro-saccades, gaze, breathing, speech nods & brow flashes
//   4. Gestures: laugh, wink, nod, smirk, blush

import { VISEMES } from "./text.js";

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const f1 = (x) => Math.round(x * 100) / 100;

// ── Viseme mouth targets ──────────────────────────────────────
// open: lip gap px · w: half width px · round: 0..1 · tT/tB: teeth top/bottom ·
// tUp: tongue tip up · tOut: tongue between teeth · tuck: lower lip under teeth · press: lips pressed
export const VISEME_SHAPES = {
  sil: { open: 0, w: 19, round: 0, tT: 0, tB: 0, tUp: 0, tOut: 0, tuck: 0, press: 0 },
  PP: { open: 0, w: 17.5, round: 0.15, tT: 0, tB: 0, tUp: 0, tOut: 0, tuck: 0, press: 1 },
  FF: { open: 3.4, w: 18.5, round: 0, tT: 1, tB: 0, tUp: 0, tOut: 0, tuck: 1, press: 0 },
  TH: { open: 5, w: 18, round: 0, tT: 0.85, tB: 0.6, tUp: 0, tOut: 1, tuck: 0, press: 0 },
  DD: { open: 6, w: 18.5, round: 0, tT: 0.8, tB: 0.6, tUp: 0.85, tOut: 0, tuck: 0, press: 0 },
  kk: { open: 8, w: 18, round: 0.05, tT: 0.6, tB: 0.45, tUp: 0.15, tOut: 0, tuck: 0, press: 0 },
  CH: { open: 5, w: 13.5, round: 0.6, tT: 1, tB: 1, tUp: 0.2, tOut: 0, tuck: 0, press: 0 },
  SS: { open: 3.4, w: 20.5, round: 0, tT: 1, tB: 1, tUp: 0.1, tOut: 0, tuck: 0, press: 0 },
  nn: { open: 5.5, w: 18.5, round: 0, tT: 0.7, tB: 0.45, tUp: 1, tOut: 0, tuck: 0, press: 0 },
  RR: { open: 6, w: 14.5, round: 0.6, tT: 0.5, tB: 0.2, tUp: 0.35, tOut: 0, tuck: 0, press: 0 },
  aa: { open: 17, w: 19.5, round: 0.1, tT: 0.6, tB: 0.25, tUp: 0, tOut: 0, tuck: 0, press: 0 },
  E: { open: 11, w: 21, round: 0, tT: 0.85, tB: 0.45, tUp: 0.1, tOut: 0, tuck: 0, press: 0 },
  ih: { open: 7, w: 22, round: 0, tT: 0.9, tB: 0.6, tUp: 0.15, tOut: 0, tuck: 0, press: 0 },
  oh: { open: 13, w: 13, round: 0.85, tT: 0.3, tB: 0.1, tUp: 0, tOut: 0, tuck: 0, press: 0 },
  ou: { open: 6, w: 10, round: 1, tT: 0.1, tB: 0, tUp: 0, tOut: 0, tuck: 0, press: 0 },
};
const MOUTH_KEYS = Object.keys(VISEME_SHAPES.sil);
// How strongly each viseme imposes its shape on neighbours (coarticulation dominance)
const DOMINANCE = { sil: 0.8, PP: 3.2, FF: 2.6, TH: 1.6, DD: 0.55, kk: 0.5, CH: 1.5, SS: 1.0, nn: 0.55, RR: 1.1,
  aa: 1.2, E: 1.1, ih: 1.0, oh: 1.6, ou: 2.0 };

export function blendVisemeProbs(probs) {
  const out = Object.fromEntries(MOUTH_KEYS.map((k) => [k, 0]));
  VISEMES.forEach((v, i) => {
    const s = VISEME_SHAPES[v], p = probs[i];
    for (const k of MOUTH_KEYS) out[k] += p * s[k];
  });
  return out;
}

/** Sample a viseme timeline at time t with a dominance-based coarticulation model. */
export function sampleTimeline(events, t, startHint = 0) {
  const out = Object.fromEntries(MOUTH_KEYS.map((k) => [k, 0]));
  let wsum = 0, i = startHint, active = null;
  while (i > 0 && events[i].t0 > t - 0.3) i--;
  for (; i < events.length; i++) {
    const e = events[i];
    if (e.t0 > t + 0.3) break;
    if (e.t1 < t - 0.3) continue;
    const c = (e.t0 + e.t1) / 2, h = Math.max((e.t1 - e.t0) / 2, 0.02);
    const inside = t >= e.t0 && t < e.t1;
    if (inside) active = e;
    const d = Math.abs(t - c) / (h + 0.035);
    let w = DOMINANCE[e.v] * Math.exp(-1.6 * d * d);
    if (inside && (e.v === "PP" || e.v === "FF" || e.v === "ou")) w *= 3; // must reach closure/rounding
    if (e.v === "sil" && e.t1 - e.t0 < 0.06) w *= 0.3;                     // micro-gaps don't close the mouth
    const s = VISEME_SHAPES[e.v];
    let gain = 1;
    if (e.stress === 1) gain = 1.12; else if (e.stress === 0) gain = 0.8;
    if (e.emph) gain *= 1.12;
    for (const k of MOUTH_KEYS) out[k] += w * (k === "open" ? s[k] * gain : s[k]);
    wsum += w;
  }
  if (wsum < 1e-4) return { shape: { ...VISEME_SHAPES.sil }, active: null };
  for (const k of MOUTH_KEYS) out[k] /= wsum;
  return { shape: out, active };
}

// ── Emotion → face parameter deltas ───────────────────────────
const EXPR_KEYS = ["browY", "browIn", "browOut", "browSq", "browAsym", "lidU", "lidL", "tiltIn", "tiltOut", "pupil",
  "cheek", "blush", "corner", "widthAdd", "openAdd", "roundAdd", "teeth", "nose", "sneer", "asym", "press",
  "headTilt", "headNod", "gazeX", "gazeY", "pout"];
const EXPR = {
  neutral: {},
  joy: { browY: 3, lidU: -0.08, lidL: 0.42, cheek: 6, blush: 0.45, corner: -10, widthAdd: 4, openAdd: 2.5, teeth: 0.7, headTilt: 4 },
  love: { browY: 1.5, browIn: 2.5, lidU: -0.2, lidL: 0.5, cheek: 5, blush: 0.85, corner: -7, widthAdd: 2, headTilt: 8, pupil: 1.2 },
  excited: { browY: 7, lidU: 0.18, lidL: 0.2, cheek: 6, blush: 0.5, corner: -10, widthAdd: 4, openAdd: 8, teeth: 0.8, pupil: 1, headNod: -3 },
  proud: { browY: 2, lidU: -0.2, lidL: 0.2, corner: -7, widthAdd: 2, headNod: -4, headTilt: -3, cheek: 3 },
  sadness: { browIn: 11, browOut: -4, lidU: -0.38, tiltOut: 0.35, corner: 9, widthAdd: -4, gazeY: 0.55, headNod: 5, blush: -0.15, pout: 2.5, headTilt: 5 },
  concern: { browIn: 7, browOut: -2, lidU: -0.12, corner: 4, widthAdd: -2, headTilt: 5 },
  anger: { browIn: -10, browSq: 5, browOut: 3, lidU: -0.32, tiltIn: 0.55, lidL: 0.32, corner: 5, widthAdd: -4, press: 0.55, nose: 1.5, blush: 0.4, headNod: 3, gazeY: -0.1 },
  fear: { browIn: 11, browOut: 3, browY: 5, lidU: 0.38, lidL: -0.1, pupil: -2, corner: 4, widthAdd: 5, openAdd: 5, headNod: -3, gazeX: 0.2 },
  surprise: { browY: 11, browIn: 2, lidU: 0.42, lidL: -0.1, pupil: -1.4, openAdd: 14, roundAdd: 0.6, widthAdd: -5, headNod: -4 },
  disgust: { browIn: -6, browSq: 3, lidL: 0.4, lidU: -0.15, nose: 3.5, sneer: 6, corner: 6, widthAdd: -2, headTilt: -6, headNod: -2 },
  thinking: { browAsym: 7, browY: 1.5, gazeX: -0.55, gazeY: -0.7, asym: 1, press: 0.4, corner: 1, headTilt: -6, lidU: -0.08 },
  curious: { browAsym: 5, browY: 2.5, headTilt: 9, corner: -2.5, lidU: 0.08 },
};
export const EMOTIONS = Object.keys(EXPR);

// Critically damped spring per parameter
class Spring {
  constructor(v, w) { this.v = v; this.t = v; this.vel = 0; this.w = w; }
  step(dt) {
    const w = this.w, x = this.v - this.t;
    const exp = Math.exp(-w * dt);
    const nx = (x + (this.vel + w * x) * dt) * exp;
    this.vel = (this.vel - w * (this.vel + w * x) * dt) * exp;
    this.v = this.t + nx;
    return this.v;
  }
}

export class Face {
  static _n = 0;
  constructor(svg) {
    this.svg = svg;
    this._build();
    this.time = 0;
    this.state = "idle";
    // springs
    const S = (v, w) => new Spring(v, w);
    this.m = Object.fromEntries(MOUTH_KEYS.map((k) => [k, S(VISEME_SHAPES.sil[k], 38)]));
    this.mouthOpenSpring = S(0, 45);
    this.x = Object.fromEntries(EXPR_KEYS.map((k) => [k, S(0, 7)]));
    this.emotion = Object.fromEntries(EMOTIONS.map((e) => [e, 0]));
    this.emoTarget = { neutral: 1 };
    this.baseMood = { joy: 0.18 };
    this.gaze = { x: S(0, 22), y: S(0, 22) };
    this.gazeTarget = { x: 0, y: 0 };
    this.mouse = null;
    this.head = { tilt: S(0, 6), nod: S(0, 8), x: S(0, 5) };
    this.blink = { t: -1, next: 1.5, dur: 0.16, double: false };
    this.wink = -1;
    this.laughT = -1;
    this.nodT = -1;
    this.blushT = -1;
    this.flash = 0;        // brow flash impulse (emphasis)
    this.nodImpulse = 0;   // head nod impulse (stressed syllables)
    this.saccadeNext = 0.5;
    this.speechEnergy = 0;
    this.mouthTarget = { ...VISEME_SHAPES.sil };
    this.mouthGain = 1;
    this.listeningLevel = 0;
    this.debug = { viseme: "sil" };
  }

  // ── construction ────────────────────────────────────────────
  _build() {
    const svg = this.svg;
    svg.setAttribute("viewBox", "-10 -10 240 240");
    svg.innerHTML = "";
    const P = `pf${++Face._n}-`;
    const defs = el("defs", {}, svg);
    const fg = el("radialGradient", { id: P + "face", cx: "42%", cy: "36%", r: "70%" }, defs);
    el("stop", { offset: "0%", "stop-color": "#fffaf3" }, fg);
    el("stop", { offset: "78%", "stop-color": "#f7efe4" }, fg);
    el("stop", { offset: "100%", "stop-color": "#efe2d2" }, fg);
    const bg = el("radialGradient", { id: P + "blush" }, defs);
    el("stop", { offset: "0%", "stop-color": "#f38b8b", "stop-opacity": "0.9" }, bg);
    el("stop", { offset: "100%", "stop-color": "#f38b8b", "stop-opacity": "0" }, bg);
    const ig = el("radialGradient", { id: P + "iris", cx: "50%", cy: "45%", r: "55%" }, defs);
    el("stop", { offset: "0%", "stop-color": "#7a4a2a" }, ig);
    el("stop", { offset: "65%", "stop-color": "#4a2814" }, ig);
    el("stop", { offset: "100%", "stop-color": "#24120a" }, ig);
    this.eyeClip = ["L", "R"].map((s) => el("path", {}, el("clipPath", { id: `${P}eye${s}` }, defs)));
    this.mouthClip = el("path", {}, el("clipPath", { id: P + "mouth" }, defs));

    this.headG = el("g", {}, svg);
    el("circle", { cx: 110, cy: 110, r: 100, fill: `url(#${P}face)`, stroke: "#0a0a0a", "stroke-width": 3 }, this.headG);
    this.cheeks = [58, 162].map((cx) => el("ellipse", { cx, cy: 124, rx: 19, ry: 12, fill: `url(#${P}blush)`, opacity: 0.35 }, this.headG));

    this.eyes = [{ cx: 80, cy: 94, side: -1 }, { cx: 140, cy: 94, side: 1 }].map((e, i) => {
      const g = el("g", {}, this.headG);
      e.white = el("path", { fill: "#fff" }, g);
      const ig2 = el("g", { "clip-path": `url(#${P}eye${i ? "R" : "L"})` }, g);
      e.iris = el("circle", { r: 10.5, fill: `url(#${P}iris)` }, ig2);
      e.pupil = el("circle", { r: 5, fill: "#0a0a0a" }, ig2);
      e.shine1 = el("circle", { r: 3.1, fill: "#fff" }, ig2);
      e.shine2 = el("circle", { r: 1.3, fill: "#fff", opacity: 0.85 }, ig2);
      e.lidShadow = el("path", { fill: "none", stroke: "#e6d6c4", "stroke-width": 3, opacity: 0.6 }, ig2);
      e.lower = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 1.5, "stroke-linecap": "round", opacity: 0.75 }, g);
      e.upper = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 2.8, "stroke-linecap": "round" }, g);
      e.lashes = [0, 1, 2].map(() => el("line", { stroke: "#0a0a0a", "stroke-width": 2.3, "stroke-linecap": "round" }, g));
      e.clip = this.eyeClip[i];
      e.brow = el("path", { fill: "none", stroke: "#2a1a10", "stroke-width": 4.6, "stroke-linecap": "round" }, this.headG);
      return e;
    });
    this.nose = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 2.1, "stroke-linecap": "round", opacity: 0.55 }, this.headG);
    this.nostrils = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 1.4, "stroke-linecap": "round", opacity: 0 }, this.headG);

    this.mouthG = el("g", {}, this.headG);
    this.mInterior = el("path", { fill: "#4a1018" }, this.mouthG);
    const mc = el("g", { "clip-path": `url(#${P}mouth)` }, this.mouthG);
    this.mTongue = el("ellipse", { fill: "#e06478" }, mc);
    this.mTeethT = el("rect", { fill: "#fff" }, mc);
    this.mTeethB = el("rect", { fill: "#f4f1ec" }, mc);
    this.mTongueTip = el("ellipse", { fill: "#ea7488" }, mc);
    this.mUpper = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 2.1, "stroke-linecap": "round", "stroke-linejoin": "round" }, this.mouthG);
    this.mLower = el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 1.8, "stroke-linecap": "round" }, this.mouthG);
    this.mLipLine = el("path", { fill: "none", stroke: "#b2475a", "stroke-width": 1.8, "stroke-linecap": "round", opacity: 0.35 }, this.mouthG);
    this.dimples = [0, 1].map(() => el("path", { fill: "none", stroke: "#0a0a0a", "stroke-width": 1.3, "stroke-linecap": "round", opacity: 0 }, this.mouthG));
  }

  // ── public API ──────────────────────────────────────────────
  setState(s) {
    this.state = s;
    if (s === "thinking") this.setEmotion({ thinking: 0.9 });
    if (s === "listening") this.setEmotion({ curious: 0.45, joy: 0.25 });
    if (s === "idle") this.setEmotion({ neutral: 1 });
  }
  /** emo: { joy: 0.8, … } — target mix; missing emotions decay to 0. */
  setEmotion(emo) { this.emoTarget = { ...emo }; }
  setMouth(shape, gain = 1) { this.mouthTarget = shape; this.mouthGain = gain; }
  lookAtScreen(nx, ny) { this.mouse = { x: nx, y: ny, t: this.time }; }
  emphasis(k = 1) { this.flash = Math.min(1.2, this.flash + 0.7 * k); this.nodImpulse = Math.min(1.5, this.nodImpulse + 0.6 * k); }
  gesture(name) {
    if (name === "wink") this.wink = 0;
    else if (name === "laugh") this.laughT = 0;
    else if (name === "nod") this.nodT = 0;
    else if (name === "blush") this.blushT = 0;
    else if (name === "blink") this.blink.t = 0;
  }

  // ── per-frame update ────────────────────────────────────────
  update(dt) {
    dt = Math.min(dt, 1 / 20);
    this.time += dt;
    const T = this.time;

    // emotion mix (with Miss Pearl's warm base mood)
    const target = { ...this.baseMood };
    let tsum = 0;
    for (const [k, v] of Object.entries(this.emoTarget)) { target[k] = Math.max(target[k] || 0, v); tsum += v; }
    if (this.emoTarget.neutral) target.joy = 0.12;
    const neg = ["sadness", "anger", "fear", "disgust", "concern"].reduce((s, k) => s + (target[k] || 0), 0);
    if (neg > 0.2) target.joy = this.emoTarget.joy || 0;
    if (this.laughT >= 0) target.joy = 1;
    for (const e of EMOTIONS) this.emotion[e] += ((target[e] || 0) - this.emotion[e]) * (1 - Math.exp(-dt * 5));

    const ex = Object.fromEntries(EXPR_KEYS.map((k) => [k, 0]));
    for (const e of EMOTIONS) {
      const w = this.emotion[e];
      if (w < 0.005) continue;
      for (const [k, v] of Object.entries(EXPR[e])) ex[k] += v * w;
    }
    // gestures
    if (this.laughT >= 0) {
      this.laughT += dt;
      const p = this.laughT / 1.4;
      if (p >= 1) this.laughT = -1;
      else {
        const env = Math.sin(Math.PI * Math.min(1, p * 1.3));
        ex.lidU -= 0.55 * env; ex.lidL += 0.55 * env; ex.openAdd += 7 * env * (0.6 + 0.4 * Math.abs(Math.sin(T * 18)));
        ex.teeth += 0.6 * env; ex.headNod += 2.5 * env * Math.sin(T * 18); ex.headTilt += 4 * env; ex.cheek += 2 * env;
      }
    }
    if (this.wink >= 0) { this.wink += dt; if (this.wink > 0.45) this.wink = -1; else ex.corner -= 2.5; }
    if (this.nodT >= 0) { this.nodT += dt; if (this.nodT > 0.9) this.nodT = -1; else ex.headNod += 5 * Math.sin(this.nodT * Math.PI * 2 / 0.45); }
    if (this.blushT >= 0) { this.blushT += dt; if (this.blushT > 2.5) this.blushT = -1; else ex.blush += 0.8 * Math.sin(Math.PI * this.blushT / 2.5); }
    // speech-driven brow flash & nods
    this.flash *= Math.exp(-dt * 6);
    this.nodImpulse *= Math.exp(-dt * 7);
    ex.browY += 3.2 * this.flash;
    ex.headNod += 2.2 * this.nodImpulse;
    if (this.state === "listening") ex.headNod += 1.5 * this.listeningLevel;
    for (const k of EXPR_KEYS) { this.x[k].t = ex[k]; this.x[k].step(dt); }
    const X = Object.fromEntries(EXPR_KEYS.map((k) => [k, this.x[k].v]));

    // mouth: viseme target + emotion modifiers, then springs
    const mt = this.mouthTarget, speaking = this.state === "speaking";
    for (const k of MOUTH_KEYS) {
      let v = mt[k];
      if (k === "open") v = v * this.mouthGain;
      this.m[k].t = v; this.m[k].step(dt);
    }
    const M = Object.fromEntries(MOUTH_KEYS.map((k) => [k, this.m[k].v]));

    // gaze: mouse > emotion bias > micro-saccades
    this.saccadeNext -= dt;
    if (this.saccadeNext <= 0) {
      const big = Math.random() < (speaking ? 0.12 : 0.25);
      const amp = big ? 0.55 : 0.12;
      this.gazeTarget = { x: (Math.random() * 2 - 1) * amp, y: (Math.random() * 2 - 1) * amp * 0.7 };
      if (big && Math.random() < 0.5 && this.blink.t < 0) this.blink.t = 0;
      this.saccadeNext = speaking ? 0.5 + Math.random() * 1.5 : 0.8 + Math.random() * 2.6;
    }
    let gx = this.gazeTarget.x, gy = this.gazeTarget.y;
    if (this.mouse && T - this.mouse.t < 2.5 && this.state !== "thinking") {
      gx = this.mouse.x * 0.9 + gx * 0.2; gy = this.mouse.y * 0.8 + gy * 0.2;
    } else if (speaking) { gx *= 0.4; gy *= 0.4; }
    gx += X.gazeX; gy += X.gazeY;
    this.gaze.x.t = clamp(gx, -1, 1); this.gaze.y.t = clamp(gy, -1, 1);
    this.gaze.x.step(dt); this.gaze.y.step(dt);

    // blinks (more often while speaking, at random, with occasional doubles)
    let blinkClose = 0;
    if (this.blink.t >= 0) {
      this.blink.t += dt;
      const p = this.blink.t / this.blink.dur;
      blinkClose = p < 0.4 ? p / 0.4 : p < 1 ? 1 - (p - 0.4) / 0.6 : 0;
      if (p >= 1) {
        this.blink.t = this.blink.double ? -0.0001 : -1;
        if (this.blink.double) { this.blink.double = false; this.blink.t = 0; }
      }
    } else {
      this.blink.next -= dt;
      if (this.blink.next <= 0) {
        this.blink.t = 0; this.blink.double = Math.random() < 0.15;
        this.blink.next = speaking ? 1.6 + Math.random() * 2.5 : 2.2 + Math.random() * 4;
      }
    }

    // head: slow idle drift + speech movement + emotion pose
    const drift = speaking ? 1.6 : 1;
    this.head.tilt.t = X.headTilt + drift * (1.2 * Math.sin(T * 0.53) + 0.7 * Math.sin(T * 1.31 + 1));
    this.head.nod.t = X.headNod + drift * (0.8 * Math.sin(T * 0.71 + 2)) + (speaking ? 0.6 * Math.sin(T * 2.3) : 0);
    this.head.x.t = drift * 1.2 * Math.sin(T * 0.37 + 0.5);
    this.head.tilt.step(dt); this.head.nod.step(dt); this.head.x.step(dt);

    this._render(X, M, blinkClose);
  }

  // ── drawing ─────────────────────────────────────────────────
  _render(X, M, blinkClose) {
    const breath = Math.sin(this.time * 1.6) * 1.2;
    this.headG.setAttribute("transform",
      `translate(${f1(this.head.x.v)} ${f1(this.head.nod.v * 0.6 + breath)}) rotate(${f1(this.head.tilt.v)} 110 205)`);

    // cheeks
    const cheekY = 124 - X.cheek;
    const blush = clamp(0.3 + X.blush * 0.6, 0.05, 0.95);
    this.cheeks.forEach((c) => { c.setAttribute("cy", f1(cheekY)); c.setAttribute("opacity", f1(blush)); });

    // eyes
    const gx = this.gaze.x.v, gy = this.gaze.y.v;
    this.eyes.forEach((e, i) => {
      const winkClose = i === 1 && this.wink >= 0 ? Math.sin(Math.PI * clamp(this.wink / 0.45, 0, 1)) : 0;
      const close = Math.max(blinkClose, winkClose);
      const openU = clamp((1 + X.lidU) * (1 - close), 0, 1.35);
      const lowR = clamp(X.lidL + close * 0.15 + winkClose * 0.3, -0.2, 1.1);
      const w = 20, hu = 23.5, hl = 17;
      const outer = e.side < 0 ? e.cx - w : e.cx + w, inner = e.side < 0 ? e.cx + w : e.cx - w;
      const cy = e.cy;
      const lowerH = hl * (1 - lowR * 1.35);               // can go negative → "^" happy eye
      const lowCp = cy + (4 / 3) * lowerH;
      const upIn = cy - (4 / 3) * hu * (1 - X.tiltIn) , upOut = cy - (4 / 3) * hu * (1 - X.tiltOut);
      const cpIn = lerp(lowCp, upIn, openU), cpOut = lerp(lowCp, upOut, openU);
      const upperD = `M${f1(outer)},${f1(cy)} C${f1(outer)},${f1(cpOut)} ${f1(inner)},${f1(cpIn)} ${f1(inner)},${f1(cy)}`;
      const lowerD = `C${f1(inner)},${f1(lowCp)} ${f1(outer)},${f1(lowCp)} ${f1(outer)},${f1(cy)}`;
      const shape = `${upperD} ${lowerD} Z`;
      e.white.setAttribute("d", shape);
      e.clip.setAttribute("d", shape);
      e.upper.setAttribute("d", upperD);
      e.lower.setAttribute("d", `M${f1(inner)},${f1(cy)} ${lowerD}`);
      e.lower.setAttribute("opacity", f1(0.75 * clamp(openU * 1.5, 0.15, 1)));
      e.lidShadow.setAttribute("d", upperD);
      // iris / pupil
      const ix = e.cx + gx * 7.5 + (e.side < 0 ? 0.6 : -0.6), iy = cy + 1 + gy * 5.5;
      const pr = clamp(5 + X.pupil * 0.9, 3.2, 7);
      e.iris.setAttribute("cx", f1(ix)); e.iris.setAttribute("cy", f1(iy));
      e.pupil.setAttribute("cx", f1(ix)); e.pupil.setAttribute("cy", f1(iy)); e.pupil.setAttribute("r", f1(pr));
      e.shine1.setAttribute("cx", f1(ix + 3.6)); e.shine1.setAttribute("cy", f1(iy - 4));
      e.shine2.setAttribute("cx", f1(ix - 3.4)); e.shine2.setAttribute("cy", f1(iy + 3.4));
      // lashes on the outer half of the upper lid
      const bez = (t) => {
        const a = [outer, cy], b = [outer, cpOut], c = [inner, cpIn], d = [inner, cy], u = 1 - t;
        const p = [0, 1].map((k) => u * u * u * a[k] + 3 * u * u * t * b[k] + 3 * u * t * t * c[k] + t * t * t * d[k]);
        const dv = [0, 1].map((k) => 3 * u * u * (b[k] - a[k]) + 6 * u * t * (c[k] - b[k]) + 3 * t * t * (d[k] - c[k]));
        return { p, dv };
      };
      [0.04, 0.16, 0.3].forEach((t, k) => {
        const { p, dv } = bez(t);
        const len = Math.hypot(dv[0], dv[1]) || 1;
        let nx = -dv[1] / len, ny = dv[0] / len;
        if (ny > 0) { nx = -nx; ny = -ny; }
        const L = [7.5, 7, 5.5][k];
        const ox = e.side * 0.55 * (1 - k * 0.3);
        e.lashes[k].setAttribute("x1", f1(p[0])); e.lashes[k].setAttribute("y1", f1(p[1]));
        e.lashes[k].setAttribute("x2", f1(p[0] + (nx + ox) * L)); e.lashes[k].setAttribute("y2", f1(p[1] + ny * L));
      });
      // brows
      const by = cy - 27 - X.browY - this.flash * 0.6;
      const asym = e.side < 0 ? X.browAsym : -X.browAsym * 0.35;
      const sq = X.browSq * (e.side < 0 ? 1 : -1);
      const bOut = [e.side < 0 ? e.cx - 18 : e.cx + 18, by + 4 - X.browOut - asym];
      const bIn = [(e.side < 0 ? e.cx + 16 : e.cx - 16) + sq, by + 4 - X.browIn - asym * 0.6];
      const bMid = [e.cx + e.side * -1, by - 4.5 - asym - (X.browIn + X.browOut) * 0.25];
      const ctrl = [2 * bMid[0] - (bOut[0] + bIn[0]) / 2, 2 * bMid[1] - (bOut[1] + bIn[1]) / 2];
      e.brow.setAttribute("d", `M${f1(bOut[0])},${f1(bOut[1])} Q${f1(ctrl[0])},${f1(ctrl[1])} ${f1(bIn[0])},${f1(bIn[1])}`);
    });
    // nose
    const ny = 111 - X.nose;
    this.nose.setAttribute("d", `M${f1(110.5)},${f1(ny)} Q${f1(104.5 - X.nose * 0.3)},${f1(ny + 10)} ${f1(109)},${f1(ny + 13)} Q${f1(112.5)},${f1(ny + 14.3)} ${f1(115)},${f1(ny + 12)}`);
    this.nostrils.setAttribute("opacity", f1(clamp(X.nose * 0.25, 0, 0.5)));
    this.nostrils.setAttribute("d", `M104,${f1(ny + 12)} q-2,-2 -1,-4 M117,${f1(ny + 11.5)} q2,-2 1,-4`);

    // mouth
    const speakingOpen = M.open;
    let open = Math.max(0, speakingOpen + X.openAdd * (1 - M.press));
    const round = clamp(M.round + X.roundAdd, 0, 1);
    let W = clamp(M.w + X.widthAdd * (1 - round * 0.6), 7, 26);
    const press = clamp(M.press + X.press, 0, 1);
    open *= 1 - press * 0.9;
    const cornerSmile = X.corner * (1 - round * 0.6) * (open > 8 ? 0.8 : 1);
    const asym = X.asym;
    const cxL = -W + asym * 2, cxR = W + asym * 2;
    const cyL = cornerSmile - asym * 2.5, cyR = cornerSmile + asym * 1.2 - X.sneer * 0.4;
    const kx = 0.55 + 0.45 * round;
    // lip centres: jaw drops the lower lip more than the upper lip rises
    const tuck = clamp(M.tuck, 0, 1);
    const upC = -open * 0.32 - X.sneer * 0.35 - round * open * 0.12;
    let loC = open * 0.68 + round * open * 0.1 + X.pout * 0.4;
    loC = lerp(loC, upC + 3.2, tuck * 0.85);
    const smileBow = -cornerSmile * 0.45 + 0.8;                 // centre sags for smiles, lifts for frowns
    const cpU = (8 * (upC + smileBow * (open < 2 ? 1 : 0.4)) - (cyL + cyR)) / 6;
    const cpL = (8 * (loC + smileBow * (open < 2 ? 1 : 0.4)) - (cyL + cyR)) / 6;
    const upper = `M${f1(cxL)},${f1(cyL)} C${f1(cxL * kx)},${f1(cpU)} ${f1(cxR * kx)},${f1(cpU)} ${f1(cxR)},${f1(cyR)}`;
    const lower = `C${f1(cxR * kx)},${f1(cpL)} ${f1(cxL * kx)},${f1(cpL)} ${f1(cxL)},${f1(cyL)}`;
    const interior = `${upper} ${lower} Z`;
    this.mouthG.setAttribute("transform", `translate(110 149) scale(1.3)`);
    this.mInterior.setAttribute("d", interior);
    this.mouthClip.setAttribute("d", interior);
    const gap = loC - upC;
    this.mInterior.setAttribute("opacity", gap > 0.8 ? 1 : 0);
    this.mUpper.setAttribute("d", upper);
    this.mUpper.setAttribute("stroke-width", f1(2.1 + press * 0.6));
    this.mLower.setAttribute("d", `M${f1(cxR)},${f1(cyR)} ${lower}`);
    this.mLower.setAttribute("opacity", gap > 1.2 ? 1 : 0);
    // teeth & tongue
    const teethT = clamp(Math.max(M.tT, X.teeth * clamp(gap / 6, 0, 1)), 0, 1);
    const th = 2 + 4.2 * teethT;
    this.mTeethT.setAttribute("x", f1(-W)); this.mTeethT.setAttribute("width", f1(2 * W));
    this.mTeethT.setAttribute("y", f1(upC - 3)); this.mTeethT.setAttribute("height", f1(teethT > 0.05 ? th + 3 : 0));
    const tb = 1.5 + 3.5 * M.tB;
    this.mTeethB.setAttribute("x", f1(-W)); this.mTeethB.setAttribute("width", f1(2 * W));
    this.mTeethB.setAttribute("y", f1(loC - tb + 1.5)); this.mTeethB.setAttribute("height", f1(M.tB > 0.05 && tuck < 0.5 ? tb + 2 : 0));
    const tongueY = lerp(loC - 2.5, upC + 5, clamp(M.tUp, 0, 1) * clamp(gap / 7, 0, 1));
    this.mTongue.setAttribute("cx", 0); this.mTongue.setAttribute("cy", f1(tongueY));
    this.mTongue.setAttribute("rx", f1(W * 0.6)); this.mTongue.setAttribute("ry", f1(clamp(gap * 0.28, 1.5, 5.5)));
    this.mTongueTip.setAttribute("cx", 0); this.mTongueTip.setAttribute("cy", f1((upC + loC) / 2 + 0.8));
    this.mTongueTip.setAttribute("rx", f1(W * 0.42)); this.mTongueTip.setAttribute("ry", f1(2.6 * clamp(M.tOut, 0, 1)));
    // lip fullness & dimples
    this.mLipLine.setAttribute("d", `M${f1(-W * 0.42)},${f1(loC + 4.2 + X.pout * 0.4)} Q0,${f1(loC + 6.6 + X.pout)} ${f1(W * 0.42)},${f1(loC + 4.2 + X.pout * 0.4)}`);
    const dim = clamp(-cornerSmile / 6 - 0.15, 0, 1);
    this.dimples[0].setAttribute("d", `M${f1(cxL - 2.5)},${f1(cyL - 3)} q-2.2,2.6 0.3,5`);
    this.dimples[1].setAttribute("d", `M${f1(cxR + 2.5)},${f1(cyR - 3)} q2.2,2.6 -0.3,5`);
    this.dimples.forEach((d) => d.setAttribute("opacity", f1(dim * 0.7)));
  }
}
