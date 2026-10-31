// Miss Pearl — app wiring: models, face loop, UI, backend.
import { Audio2Viseme, EmotionModel, G2P } from "./nn.js";
import { Durations, Lexicon, Pronouncer, VISEMES } from "./text.js";
import { Face } from "./face.js";
import { FeatureExtractor, StreamingVisemes } from "./audio.js";
import { Performer } from "./speech.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem("pearl." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("pearl." + k, JSON.stringify(v)); } catch {} },
};

// ── face ──────────────────────────────────────────────────────
const face = new Face($("face"));
let last = performance.now();
const ring = $("ring"), rctx = ring.getContext("2d");
const ringSmooth = new Float32Array(72);
let performer = null;
let lastInfo = null;

function drawRing() {
  const W = ring.width, H = ring.height, cx = W / 2, cy = H / 2, inner = W * 0.385, maxH = W * 0.085;
  rctx.clearRect(0, 0, W, H);
  let data = null;
  if (performer && performer.analyser && performer.active && performer.active.src) {
    data = new Uint8Array(performer.analyser.frequencyBinCount);
    performer.analyser.getByteFrequencyData(data);
  }
  const open = face.m.open.v / 18;
  for (let i = 0; i < ringSmooth.length; i++) {
    let raw;
    if (data) raw = data[Math.floor((i / ringSmooth.length) * data.length * 0.55)] / 255;
    else raw = Math.max(0, open * (0.55 + 0.45 * Math.sin(i * 1.7 + performance.now() / 90)));
    ringSmooth[i] = ringSmooth[i] * 0.7 + raw * 0.3;
    const v = ringSmooth[i], a = (i / ringSmooth.length) * Math.PI * 2 - Math.PI / 2;
    const h = Math.max(4, v * maxH);
    rctx.strokeStyle = i % 2 ? `rgba(245,99,10,${0.3 + v * 0.7})` : `rgba(13,51,32,${0.3 + v * 0.7})`;
    rctx.lineWidth = 5; rctx.lineCap = "round";
    rctx.beginPath();
    rctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    rctx.lineTo(cx + Math.cos(a) * (inner + h), cy + Math.sin(a) * (inner + h));
    rctx.stroke();
  }
}

function loop(t) {
  const dt = (t - last) / 1000; last = t;
  if (performer) lastInfo = performer.tick() || lastInfo;
  if (performer && !performer.active && face.state === "speaking") face.setState(mirror ? "speaking" : "idle");
  face.update(dt);
  drawRing();
  brainTick(t);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

document.addEventListener("pointermove", (e) => {
  const r = $("face").getBoundingClientRect();
  face.lookAtScreen(((e.clientX - (r.left + r.width / 2)) / (r.width / 2)) * 0.9, ((e.clientY - (r.top + r.height / 2)) / (r.height / 2)) * 0.8);
});
$("face").addEventListener("click", () => face.gesture(["wink", "laugh", "nod", "blush"][Math.floor(Math.random() * 4)]));

// ── face brain panel ─────────────────────────────────────────
const vbars = VISEMES.map((v) => { const d = document.createElement("div"); $("vbars").appendChild(d); const l = document.createElement("span"); l.textContent = v; $("vlabels").appendChild(l); return d; });
const EMO_LABELS = ["joy", "sadness", "anger", "surprise", "fear", "disgust", "neutral"];
const emoRows = Object.fromEntries(EMO_LABELS.map((e) => {
  const row = document.createElement("div"); row.className = "emo";
  row.innerHTML = `<span>${e}</span><div class="bar"><div></div></div><span>0%</span>`;
  $("emos").appendChild(row);
  return [e, row];
}));
const visDisplay = new Float32Array(VISEMES.length);
let brainLast = 0, lastProbs = null;
function brainTick(t) {
  if (t - brainLast < 50) return;
  brainLast = t;
  const info = performer && performer.active ? lastInfo : null;
  const target = new Float32Array(VISEMES.length);
  if (info && info.probs) info.probs.forEach((p, i) => (target[i] = p));
  else target[VISEMES.indexOf(info ? info.viseme : "sil")] = 1;
  let top = 0;
  for (let i = 0; i < VISEMES.length; i++) {
    visDisplay[i] = visDisplay[i] * 0.45 + target[i] * 0.55;
    if (visDisplay[i] > visDisplay[top]) top = i;
  }
  vbars.forEach((b, i) => { b.style.height = `${Math.max(3, visDisplay[i] * 54)}px`; b.classList.toggle("top", i === top && visDisplay[i] > 0.2); });
  $("visBig").textContent = VISEMES[top];
  $("curWord").textContent = info && info.word ? info.word : "—";
  $("syncInfo").textContent = info ? info.sync || "—" : "—";
  $("srcLabel").textContent = info ? info.source : face.state;
  // emotion bars: model probabilities if we have them, else the live face mix
  const mix = lastProbs || {
    joy: face.emotion.joy + face.emotion.excited * 0.7 + face.emotion.love * 0.6, sadness: face.emotion.sadness + face.emotion.concern * 0.5,
    anger: face.emotion.anger, surprise: face.emotion.surprise + face.emotion.excited * 0.3, fear: face.emotion.fear,
    disgust: face.emotion.disgust, neutral: Math.max(0, 1 - face.emotion.joy - face.emotion.sadness - face.emotion.anger - face.emotion.surprise),
  };
  for (const e of EMO_LABELS) {
    const v = Math.max(0, Math.min(1, mix[e] || 0));
    emoRows[e].children[1].firstChild.style.width = `${v * 100}%`;
    emoRows[e].children[2].textContent = `${Math.round(v * 100)}%`;
  }
}

// ── bubble with word highlighting ────────────────────────────
let bubbleSpans = [];
function renderBubble(text) {
  const b = $("bubble");
  b.textContent = "";
  bubbleSpans = [];
  const re = /\S+/g; let m, pos = 0;
  while ((m = re.exec(text))) {
    if (m.index > pos) b.appendChild(document.createTextNode(text.slice(pos, m.index)));
    const s = document.createElement("span"); s.className = "w"; s.textContent = m[0];
    s.dataset.s = m.index; s.dataset.e = m.index + m[0].length;
    b.appendChild(s); bubbleSpans.push(s); pos = m.index + m[0].length;
  }
  if (pos < text.length) b.appendChild(document.createTextNode(text.slice(pos)));
}
let currentPerf = null;
function highlight(globalChar) {
  for (const s of bubbleSpans) {
    const a = +s.dataset.s, e = +s.dataset.e;
    s.classList.toggle("on", globalChar >= a && globalChar < e);
    s.classList.toggle("done", e <= globalChar);
  }
}

// ── settings ─────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const settings = {
  engine: store.get("engine", "browser"), voice: store.get("voice", ""), rate: store.get("rate", 1), pitch: store.get("pitch", 1.1),
  api: params.get("api") ?? store.get("api", /github\.io$/.test(location.hostname) || location.protocol === "file:" ? "" : location.origin),
  lead: store.get("lead", 45),
};
let backend = { online: false, tts: false };

function voices() { return window.speechSynthesis ? speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang)) : []; }
function pickVoice() {
  const vs = voices();
  return vs.find((v) => v.name === settings.voice)
    || vs.find((v) => v.localService && /zira|aria|jenny|samantha|female|susan|hazel|libby|sonia|natasha/i.test(v.name))
    || vs.find((v) => v.localService) || vs[0] || null;
}
function fillVoices() {
  const sel = $("voiceSel"), vs = voices(), cur = pickVoice();
  sel.innerHTML = "";
  vs.forEach((v) => { const o = document.createElement("option"); o.value = v.name; o.textContent = `${v.name}${v.localService ? "" : " (online)"} — ${v.lang}`; if (cur && v.name === cur.name) o.selected = true; sel.appendChild(o); });
}
if (window.speechSynthesis) { speechSynthesis.onvoiceschanged = fillVoices; fillVoices(); }
$("engineSel").value = settings.engine;
$("rate").value = settings.rate; $("rateVal").textContent = (+settings.rate).toFixed(2);
$("pitch").value = settings.pitch; $("pitchVal").textContent = (+settings.pitch).toFixed(2);
$("apiUrl").value = settings.api;
$("leadMs").value = settings.lead; $("leadVal").textContent = settings.lead;
$("openSettings").onclick = () => $("settings").showModal();
$("closeSettings").onclick = () => { $("settings").close(); checkBackend(); };
$("engineSel").onchange = (e) => store.set("engine", (settings.engine = e.target.value));
$("voiceSel").onchange = (e) => store.set("voice", (settings.voice = e.target.value));
$("rate").oninput = (e) => { store.set("rate", (settings.rate = +e.target.value)); $("rateVal").textContent = settings.rate.toFixed(2); };
$("pitch").oninput = (e) => { store.set("pitch", (settings.pitch = +e.target.value)); $("pitchVal").textContent = settings.pitch.toFixed(2); };
$("apiUrl").onchange = (e) => store.set("api", (settings.api = e.target.value.trim().replace(/\/$/, "")));
$("leadMs").oninput = (e) => { store.set("lead", (settings.lead = +e.target.value)); $("leadVal").textContent = settings.lead; if (performer) performer.lead = settings.lead / 1000; };
$("testApi").onclick = () => checkBackend(true);

async function checkBackend(verbose = false) {
  const chip = $("chipBrain");
  if (!settings.api) { backend = { online: false, tts: false }; chip.textContent = "brain: offline demo"; chip.className = "chip warn"; return; }
  try {
    const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 3500);
    const r = await fetch(`${settings.api}/health`, { signal: ctl.signal }); clearTimeout(to);
    const h = await r.json();
    backend = { online: true, tts: !!(h.elevenlabs && h.elevenlabs.key_set), timed: !!h.tts_timed };
    chip.textContent = `brain: ${h.active_ai_provider || "online"}${backend.tts ? " + ElevenLabs" : ""}`; chip.className = "chip ok";
  } catch (e) {
    backend = { online: false, tts: false };
    chip.textContent = "brain: offline demo"; chip.className = "chip warn";
    if (verbose) $("status").textContent = `backend unreachable: ${settings.api}`;
  }
}
checkBackend();

// ── models ───────────────────────────────────────────────────
let models = {};
(async function boot() {
  const chip = $("chipModels");
  const tasks = {
    lexicon: Lexicon.load("models/lexicon.txt"),
    g2p: G2P.load("models/g2p.json"),
    a2v: Audio2Viseme.load("models/audio2viseme.json"),
    emotion: EmotionModel.load("models/emotion.json"),
    durations: fetch("models/durations.json").then((r) => (r.ok ? r.json() : null)),
  };
  const out = {};
  await Promise.all(Object.entries(tasks).map(async ([k, p]) => {
    try { out[k] = await p; } catch (e) { console.warn(`[pearl] ${k} failed to load:`, e); out[k] = null; }
  }));
  models = out;
  const pron = new Pronouncer(out.lexicon, out.g2p);
  const fx = out.a2v ? new FeatureExtractor(out.a2v.featureConfig) : null;
  performer = new Performer({ face, pron, durations: new Durations(out.durations), emotionModel: out.emotion, a2v: out.a2v, fx,
    hooks: {
      onPerformance: (perf) => { currentPerf = perf; renderBubble(perf.clean); },
      onWord: (si, wi, plan) => {
        if (!currentPerf) return;
        const tok = plan.tokens[plan.words[wi].tok];
        const base = si >= 0 ? currentPerf.sentences[si].start : 0;
        highlight(base + tok.start);
      },
      onEmotion: (emo) => {
        lastProbs = emo.probs ? { ...emo.probs } : null;
        const top = Object.entries(emo.target).sort((a, b) => b[1] - a[1])[0];
        $("moodInfo").textContent = top ? top[0] : "neutral";
        $("moodSrc").textContent = emo.source === "llm-tag" ? "(LLM tag)" : emo.source === "model" ? "(emotion model)" : "";
      },
      onDone: () => {
        highlight(1e9); $("status").textContent = "ready"; face.setState("idle"); setTimeout(() => { lastProbs = null; }, 1500);
        const nw = [...pron.neuralWords.entries()].slice(-4);
        $("pronInfo").textContent = `Pronunciation so far: ${pron.stats.lexicon} words from the CMU dictionary, ${pron.stats.neural} invented by the neural G2P` +
          (nw.length ? ` (${nw.map(([w, p]) => `${w} → ${p.join(" ")}`).join(", ")})` : "") + ".";
      },
      onAnalysis: (a) => { $("audioInfo").textContent = `Network analysed ${a.frames} frames (${a.duration.toFixed(1)} s of audio) in ${Math.round(a.ms)} ms on your CPU.`; },
    } });
  performer.lead = settings.lead / 1000;
  const ok = Object.entries(out).filter(([, v]) => v).map(([k]) => k);
  chip.textContent = ok.length >= 5 ? "face models ✓" : `models: ${ok.length}/5`;
  chip.className = ok.length >= 5 ? "chip ok" : "chip warn";
  const mi = [];
  if (out.lexicon) mi.push(`lexicon ${out.lexicon.size.toLocaleString()} words`);
  const m = (o) => (o && (o.cfg || o.config) && (o.cfg || o.config).metrics) || {};
  if (out.g2p) mi.push(`neural G2P (${(100 * (1 - (m(out.g2p).per ?? 0))).toFixed(1)}% phones right on unseen words)`);
  if (out.a2v) mi.push(`audio→viseme net (${(100 * (m(out.a2v).viseme_acc_tol20ms ?? m(out.a2v).viseme_acc ?? 0)).toFixed(1)}% frames ±20 ms on unseen voices)`);
  if (out.emotion) mi.push(`emotion net (${(100 * (m(out.emotion).teacher_agreement ?? 0)).toFixed(1)}% agreement with RoBERTa teacher)`);
  $("modelInfo").textContent = mi.join(" · ");
  window.pearl = { face, performer, models, pron };
})();

// ── speaking ─────────────────────────────────────────────────
async function speak(text) {
  if (!performer || !text.trim()) return;
  performer.stop();
  $("status").textContent = "speaking…";
  if (settings.engine === "backend" && backend.online && backend.tts) {
    try { await performer.speakBackend(text, settings.api); return; }
    catch (e) { console.warn("[pearl] backend TTS failed, using browser voice", e); }
  }
  await performer.speakBrowser(text, { voice: pickVoice(), rate: +settings.rate, pitch: +settings.pitch });
}

// ── tabs ─────────────────────────────────────────────────────
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("on", p.id === `panel-${b.dataset.tab}`));
}));

// Say
const EXAMPLES = [
  ["Tongue twister", "Peter Piper picked a peck of pickled peppers. She sells seashells by the seashore!"],
  ["Any name", "Hello Nakawa, Musoke, Zorblax and Kwame Nkrumah. Pleased to meet you, Ssekandi!"],
  ["Numbers", "It's 2026, the price is $45.99 and we are 100% ready on the 3rd of November."],
  ["Sad → happy", "[sad] I'm so sorry, I lost the match yesterday. [joy] But guess what? We won the final today!"],
  ["Angry", "I can't believe someone deleted my homework again! This is absolutely unacceptable."],
  ["Surprise", "Wait, what?! You built a robot that dances? No way!"],
  ["Thinking", "[thinking] Hmm, let me think about that for a moment... [joy] Yes, I have the answer."],
  ["Pangram", "The quick brown fox jumps over the lazy dog, while five wizards quickly jump."],
];
EXAMPLES.forEach(([label, text]) => {
  const b = document.createElement("button"); b.textContent = label; b.title = text;
  b.onclick = () => { $("sayText").value = text; speak(text); };
  $("examples").appendChild(b);
});
$("sayBtn").onclick = () => speak($("sayText").value);
$("stopBtn").onclick = () => { performer && performer.stop(); $("status").textContent = "ready"; face.setState("idle"); };
$("sayText").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) speak($("sayText").value); });

// Expression buttons
$("emobar").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.gesture) face.gesture(b.dataset.gesture);
  else { face.setEmotion({ [b.dataset.emo]: 1 }); lastProbs = null; $("moodInfo").textContent = b.dataset.emo; $("moodSrc").textContent = "(button)"; clearTimeout(window.__emoT); window.__emoT = setTimeout(() => face.setEmotion({ neutral: 1 }), 4000); }
});

// Chat
let history = [];
function addMsg(who, text) {
  const d = document.createElement("div"); d.className = `msg ${who}`; d.textContent = text;
  $("msgs").appendChild(d); $("msgs").scrollTop = $("msgs").scrollHeight;
}
function offlineReply(q) {
  const s = q.toLowerCase();
  if (/\b(hi|hello|hey|howdy|good (morning|afternoon|evening))\b/.test(s)) return "[joy] Well hello there, my genius friend! I'm Miss Pearl from Pearl AI Labs in Kampala. Tickety-tock, what shall we talk about?";
  if (/how are you/.test(s)) return "[joy] I'm feeling wonderful, thank you for asking! My face models are warmed up and ready.";
  if (/who are you|what are you|your name/.test(s)) return "[proud] I'm Miss Pearl, the face of Pearl AI Labs in Kampala, Uganda. I was brought to life by Paul and Mr. Divid!";
  if (/joke|funny|laugh/.test(s)) return "[thinking] Why did the computer go to the doctor? [laugh] Because it had a terrible case of the bytes! Tickety-tock!";
  if (/sad|cry|lost|died|sick/.test(s)) return "[sad] Oh dear, I'm so sorry to hear that. [love] Sending you a big warm hug from Kampala.";
  if (/thank/.test(s)) return "[love] You're most welcome, dear! Pearl AI Labs is so proud of your curiosity.";
  if (/angry|mad|annoy/.test(s)) return "[anger] Grr, that sounds frustrating! [joy] But let's take a breath and fix it together.";
  return "[thinking] Hmm, my big brain is offline on this page, so I can't look that up right now. [joy] But try the Say tab, and I'll say absolutely anything you type!";
}
async function sendChat(text) {
  text = text.trim(); if (!text) return;
  addMsg("me", text);
  face.setState("thinking"); $("status").textContent = "thinking…";
  let reply;
  if (backend.online) {
    try {
      const r = await fetch(`${settings.api}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text, history }) });
      const d = await r.json();
      reply = d.tagged || d.reply;
    } catch (e) { reply = null; }
  }
  if (!reply) { await new Promise((r) => setTimeout(r, 600)); reply = offlineReply(text); }
  history.push({ role: "user", content: text }, { role: "assistant", content: reply });
  history = history.slice(-12);
  const shown = reply.replace(/\[[a-z_ -]{2,20}\]\s*/gi, "").replace(/\*([a-z ]{2,24})\*\s*/gi, "");
  addMsg("pearl", shown);
  await speak(reply);
}
$("chatSend").onclick = () => { const v = $("chatInput").value; $("chatInput").value = ""; sendChat(v); };
$("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("chatSend").click(); });
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
$("micBtn").onclick = () => {
  if (!SR) { $("status").textContent = "speech recognition needs Chrome"; return; }
  const rec = new SR(); rec.lang = "en-US"; rec.interimResults = false;
  face.setState("listening"); $("status").textContent = "listening…"; $("micBtn").classList.add("on");
  rec.onresult = (e) => sendChat(e.results[0][0].transcript);
  rec.onend = () => { $("micBtn").classList.remove("on"); if (face.state === "listening") { face.setState("idle"); $("status").textContent = "ready"; } };
  rec.start();
};

// Mirror (live mic → trained network)
let mirror = null;
$("mirrorBtn").onclick = async () => {
  if (mirror) { mirror.stop(); mirror = null; $("mirrorBtn").textContent = "🎙 Start mirror"; $("mirrorBtn").classList.remove("on"); face.setState("idle"); return; }
  if (!models.a2v) { $("mirrorInfo").textContent = "Audio model not loaded."; return; }
  performer.stop();
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
  catch { $("mirrorInfo").textContent = "Microphone permission denied."; return; }
  const ctx = new AudioContext({ sampleRate: 16000 });
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(2048, 1, 1);
  const mute = ctx.createGain(); mute.gain.value = 0;
  src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
  const sink = performer.startStream();
  const fx = new FeatureExtractor(models.a2v.featureConfig);
  let frames = 0, t0 = performance.now(), busy = 0;
  const sv = new StreamingVisemes(models.a2v, fx, (p, rms) => { sink.push(p, rms); frames++; });
  proc.onaudioprocess = (e) => {
    const a = performance.now();
    sv.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    busy += performance.now() - a;
    if (frames && frames % 100 === 0) $("mirrorInfo").textContent = `Live: ${frames} frames through the network · ${(busy / ((performance.now() - t0) / 1000) / 10).toFixed(1)}% CPU · ~${Math.round((models.a2v.context * 10) + 128)} ms latency`;
  };
  mirror = { stop: () => { proc.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop()); ctx.close(); sink.stop(); } };
  $("mirrorBtn").textContent = "■ Stop mirror"; $("mirrorBtn").classList.add("on");
  $("mirrorInfo").textContent = "Listening… say something!";
};

// Audio files
const drop = $("drop");
async function handleFile(file) {
  if (!file || !performer) return;
  performer.stop();
  $("audioInfo").textContent = `Analysing ${file.name}…`;
  renderBubble(`🎧 ${file.name}`);
  try {
    await performer.playAudio(await file.arrayBuffer(), { onProgress: (p) => ($("prog").style.width = `${Math.round(p * 100)}%`) });
  } catch (e) { $("audioInfo").textContent = `Couldn't read that file: ${e.message}`; }
}
$("fileIn").onchange = (e) => handleFile(e.target.files[0]);
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
