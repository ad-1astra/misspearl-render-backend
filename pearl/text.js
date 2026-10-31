// Miss Pearl — text → phonemes → timed visemes.
//
//   "[joy] Hello Dr. Musoke, it's 2026!"
//     ├─ emotion / gesture tags  → face events      ([joy], *laughs*, …)
//     ├─ normalisation           → "hello doctor musoke it's twenty twenty six"
//     ├─ CMU lexicon (125k words) or the trained neural G2P for unknown words
//     ├─ learned phone durations (from Kokoro alignments)
//     └─ viseme timeline (Meta/Oculus 15) with stress, word & sentence info

export const VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"];

const P2V = {
  AA: "aa", AE: "aa", AH: "aa", AO: "oh", AW: ["aa", "ou"], AY: ["aa", "ih"], EH: "E", ER: "RR",
  EY: ["E", "ih"], IH: "ih", IY: "ih", OW: ["oh", "ou"], OY: ["oh", "ih"], UH: "ou", UW: "ou",
  B: "PP", M: "PP", P: "PP", F: "FF", V: "FF", TH: "TH", DH: "TH", T: "DD", D: "DD", K: "kk",
  G: "kk", NG: "kk", CH: "CH", JH: "CH", SH: "CH", ZH: "CH", S: "SS", Z: "SS", N: "nn", L: "nn",
  R: "RR", W: "ou", Y: "ih", HH: null,
};
const VOWELS = new Set(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]);
const CODE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+-./:;<=>?@[]^_{}~";

const FUNCTION_WORDS = new Set(("the a an of to and in is it for on with as at by be are was were that this you i my your " +
  "me we he she they or but so do not from have has had will can just than then there their our its am " +
  "his her them us if").split(" "));

// ─────────────────────────────────────────────────────────────
// Lexicon (CMUdict, compact)
// ─────────────────────────────────────────────────────────────

export class Lexicon {
  constructor(text) {
    const nl = text.indexOf("\n");
    this.tokens = text.slice(0, nl).trim().split(" ");
    this.map = new Map();
    let i = nl + 1;
    while (i < text.length) {
      let j = text.indexOf("\n", i);
      if (j < 0) j = text.length;
      const sp = text.indexOf(" ", i);
      const e = text.charCodeAt(j - 1) === 13 ? j - 1 : j;   // tolerate CRLF
      if (sp > i && sp < e) this.map.set(text.slice(i, sp), text.slice(sp + 1, e));
      i = j + 1;
    }
    this.code = new Map([...CODE_ALPHABET].slice(0, this.tokens.length).map((c, k) => [c, this.tokens[k]]));
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`lexicon: HTTP ${r.status}`);
    return new Lexicon(await r.text());
  }
  get size() { return this.map.size; }
  get(word) {
    const c = this.map.get(word);
    return c ? [...c].map((ch) => this.code.get(ch)) : null;
  }
}

// ─────────────────────────────────────────────────────────────
// Emotion & gesture tags (from the LLM or typed by the user)
// ─────────────────────────────────────────────────────────────

export const TAG_ALIASES = {
  joy: "joy", happy: "joy", smile: "joy", smiling: "joy", glad: "joy", cheerful: "joy", excited: "excited",
  love: "love", warm: "love", tender: "love", proud: "proud",
  sad: "sadness", sadness: "sadness", sorry: "sadness", sigh: "sadness", sighs: "sadness", concerned: "concern", concern: "concern",
  angry: "anger", anger: "anger", annoyed: "anger", frustrated: "anger",
  surprise: "surprise", surprised: "surprise", wow: "surprise", gasp: "surprise", gasps: "surprise", shocked: "surprise",
  fear: "fear", scared: "fear", afraid: "fear", nervous: "fear", worried: "fear",
  disgust: "disgust", disgusted: "disgust", eww: "disgust", gross: "disgust",
  neutral: "neutral", calm: "neutral", serious: "neutral",
  thinking: "thinking", think: "thinking", hmm: "thinking", curious: "curious", wonder: "curious",
  laugh: "laugh", laughs: "laugh", laughing: "laugh", giggle: "laugh", giggles: "laugh", chuckles: "laugh",
  wink: "wink", winks: "wink", smirk: "smirk", smirks: "smirk", nod: "nod", nods: "nod", blush: "blush", blushes: "blush",
  smiles: "joy",
};

/** Strip [tags] and *stage directions*; return clean text + tag positions. */
export function parseTags(text) {
  // markdown noise the TTS shouldn't read (done first so tag positions stay exact)
  text = text.replace(/```[\s\S]*?```/g, " ").replace(/`([^`]*)`/g, "$1")
    .replace(/^#+\s*/gm, "").replace(/^\s*(?:[-•]|\d+\.)\s+/gm, "").replace(/\*\*|__/g, "")
    .replace(/https?:\/\/\S+/g, "this link");
  const marks = [];
  let clean = "";
  const append = (seg) => {
    seg = seg.replace(/\s+/g, " ");
    if (!clean || clean.endsWith(" ")) seg = seg.replace(/^ /, "");
    clean += seg;
  };
  const re = /\[([a-zA-Z_ -]{2,20})\]|\*([a-zA-Z ]{2,24})\*/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    const raw = (m[1] || m[2]).trim().toLowerCase();
    const key = raw.split(/\s+/).find((w) => TAG_ALIASES[w]);
    append(text.slice(last, m.index));
    if (key) marks.push({ pos: clean.length, tag: TAG_ALIASES[key] });
    else if (m[2]) append(m[2]); // *emphasis* that isn't an action → keep words
    last = m.index + m[0].length;
  }
  append(text.slice(last));
  clean = clean.trimEnd();
  for (const mk of marks) mk.pos = Math.min(mk.pos, clean.length);
  return { clean, marks };
}

// ─────────────────────────────────────────────────────────────
// Normalisation: numbers, symbols, abbreviations
// ─────────────────────────────────────────────────────────────

const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
  "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES = [[1e12, "trillion"], [1e9, "billion"], [1e6, "million"], [1e3, "thousand"]];
const ORD = { one: "first", two: "second", three: "third", five: "fifth", eight: "eighth", nine: "ninth", twelve: "twelfth" };

export function intToWords(n) {
  n = Math.floor(Math.abs(n));
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? " " + ONES[n % 10] : "");
  if (n < 1000) return ONES[Math.floor(n / 100)] + " hundred" + (n % 100 ? " and " + intToWords(n % 100) : "");
  for (const [v, name] of SCALES) {
    if (n >= v) return intToWords(Math.floor(n / v)) + " " + name + (n % v ? (n % v < 100 ? " and " : " ") + intToWords(n % v) : "");
  }
  return String(n);
}
function ordinal(words) {
  const parts = words.split(" ");
  const last = parts.pop();
  let o = ORD[last] || (last.endsWith("y") ? last.slice(0, -1) + "ieth" : last + "th");
  return [...parts, o].join(" ");
}
function yearWords(n) {
  const hi = Math.floor(n / 100), lo = n % 100;
  if (n >= 2000 && n < 2010) return intToWords(n);
  return intToWords(hi) + " " + (lo === 0 ? "hundred" : lo < 10 ? "oh " + ONES[lo] : intToWords(lo));
}

const ABBREV = {
  "dr.": "doctor", "mr.": "mister", "mrs.": "missus", "ms.": "miz", "prof.": "professor", "st.": "saint",
  "e.g.": "for example", "i.e.": "that is", "etc.": "et cetera", "vs.": "versus", "approx.": "approximately",
  "&": "and", "+": "plus", "=": "equals", "@": "at", "%": "percent", "°c": "degrees celsius", "°f": "degrees fahrenheit",
  "°": "degrees", "ugx": "u g x", "usd": "u s dollars", "km": "kilometers", "kg": "kilograms", "cm": "centimeters",
};

const TOKEN_RE = /(?:[A-Za-z]\.){2,}|\b(?:[Dd]r|[Mm]rs?|[Mm]s|[Pp]rof|[Ss]t|etc|vs|approx)\.|[A-Za-z][A-Za-z'’]*|\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?|[$£€]?\d[\d,]*(?:\.\d+)?(?:%|st|nd|rd|th|s)?|°[cCfF]?|[&+=@%]|[.,!?;:—–\-…]+|\S/g;

/** Turn one token into spoken words (lowercase). Returns [] for punctuation. */
function speakToken(tok, lexicon) {
  const low = tok.toLowerCase().replace(/’/g, "'");
  if (ABBREV[low]) return ABBREV[low].split(" ");
  if (/^(?:[a-z]\.){2,}$/.test(low)) return low.replace(/\./g, "").split("");          // u.s.a. → u s a
  if (/^[a-z]+\.$/.test(low)) return speakToken(tok.slice(0, -1), lexicon);
  let t = low.match(/^(\d{1,2}):(\d{2})\s?([ap])?/);
  if (t) {
    const h = +t[1], mi = +t[2];
    let w = intToWords(h) + (mi === 0 ? (t[3] ? "" : " o'clock") : mi < 10 ? " oh " + ONES[mi] : " " + intToWords(mi));
    if (t[3]) w += t[3] === "a" ? " a m" : " p m";
    return w.split(" ");
  }
  const money = low.match(/^([$£€])(\d[\d,]*)\.(\d{2})$/);
  if (money) {
    const d = parseInt(money[2].replace(/,/g, ""), 10), c = +money[3];
    const unit = { $: ["dollar", "dollars", "cent", "cents"], "£": ["pound", "pounds", "penny", "pence"], "€": ["euro", "euros", "cent", "cents"] }[money[1]];
    let w = `${intToWords(d)} ${d === 1 ? unit[0] : unit[1]}`;
    if (c) w += ` and ${intToWords(c)} ${c === 1 ? unit[2] : unit[3]}`;
    return w.split(" ");
  }
  let m = low.match(/^([$£€])?(\d[\d,]*)(\.\d+)?(%|st|nd|rd|th|s)?$/);
  if (m) {
    const [, cur, intPart, dec, suf] = m;
    const n = parseInt(intPart.replace(/,/g, ""), 10);
    let w;
    if (!cur && !dec && (!suf || suf === "s") && /^\d{4}$/.test(intPart) && n >= 1100 && n <= 2099) w = yearWords(n);
    else w = intToWords(n);
    if (dec) w += " point " + dec.slice(1).split("").map((d) => ONES[+d]).join(" ");
    if (suf === "st" || suf === "nd" || suf === "rd" || suf === "th") w = ordinal(w);
    if (suf === "s") w = w.endsWith("y") ? w.slice(0, -1) + "ies" : w + "s";
    if (suf === "%") w += " percent";
    if (cur) w += { $: n === 1 ? " dollar" : " dollars", "£": " pounds", "€": " euros" }[cur];
    return w.split(" ");
  }
  if (/^[a-z][a-z']*$/.test(low)) {
    // ALL-CAPS acronym that isn't a dictionary word → spell it (AI → a i)
    if (tok.length <= 5 && tok === tok.toUpperCase() && tok.length > 1 && !(lexicon && lexicon.get(low) && tok.length > 3)) {
      return [...low.replace(/'/g, "")];
    }
    return [low];
  }
  return [];
}

// ─────────────────────────────────────────────────────────────
// Pronunciation
// ─────────────────────────────────────────────────────────────

const LETTER_FALLBACK = { a: "AE1", b: "B", c: "K", d: "D", e: "EH1", f: "F", g: "G", h: "HH", i: "IH1", j: "JH", k: "K",
  l: "L", m: "M", n: "N", o: "AA1", p: "P", q: "K", r: "R", s: "S", t: "T", u: "AH1", v: "V", w: "W", x: "K", y: "Y", z: "Z" };

export class Pronouncer {
  constructor(lexicon, g2p) { this.lexicon = lexicon; this.g2p = g2p; this.stats = { lexicon: 0, neural: 0, fallback: 0 }; this.neuralWords = new Map(); }
  phones(word) {
    const w = word.toLowerCase().replace(/’/g, "'");
    const lex = this.lexicon;
    let p = lex && lex.get(w);
    if (p) { this.stats.lexicon++; return { phones: p, source: "lexicon" }; }
    if (lex && w.endsWith("'s") && lex.get(w.slice(0, -2))) {
      const base = lex.get(w.slice(0, -2));
      const last = base[base.length - 1];
      const suf = ["S", "Z", "SH", "ZH", "CH", "JH"].includes(last) ? ["IH0", "Z"] : ["P", "T", "K", "F", "TH"].includes(last) ? ["S"] : ["Z"];
      this.stats.lexicon++;
      return { phones: [...base, ...suf], source: "lexicon" };
    }
    if (this.g2p) {
      p = this.g2p.phonemize(w);
      if (p.length) { this.stats.neural++; this.neuralWords.set(w, p); return { phones: p, source: "neural" }; }
    }
    this.stats.fallback++;
    return { phones: [...w.replace(/[^a-z]/g, "")].map((c) => LETTER_FALLBACK[c]).filter(Boolean), source: "rules" };
  }
}

// ─────────────────────────────────────────────────────────────
// Durations (learned from Kokoro alignments; sane defaults if absent)
// ─────────────────────────────────────────────────────────────

export class Durations {
  constructor(json) {
    this.phones = {};
    if (json && json.phones) for (const [k, v] of Object.entries(json.phones)) this.phones[k] = v.median ?? v.mean ?? v;
    this.pauses = Object.assign({ ",": 0.18, ".": 0.32, "!": 0.32, "?": 0.32, ";": 0.25, ":": 0.25, "—": 0.2 }, (json && json.pauses) || {});
  }
  of(ph) {
    const base = ph.replace(/\d/, "");
    const st = ph.match(/\d/)?.[0];
    if (this.phones[ph] != null) return this.phones[ph];
    if (VOWELS.has(base)) {
      for (const k of [base + "1", base + "0", base + "2"]) if (this.phones[k] != null) return this.phones[k] * (st === "0" ? 0.7 : 1);
      return st === "0" ? 0.055 : 0.11;
    }
    if (this.phones[base] != null) return this.phones[base];
    return ["P", "B", "T", "D", "K", "G"].includes(base) ? 0.06 : 0.075;
  }
  pause(p) { for (const ch of p) if (this.pauses[ch] != null) return this.pauses[ch]; return 0.15; }
}

// ─────────────────────────────────────────────────────────────
// Planning
// ─────────────────────────────────────────────────────────────

/** Split clean text into sentences, keeping char offsets. */
export function splitSentences(clean) {
  const out = [];
  const re = /[^.!?\n]+(?:[.!?]+["')\]]*|\n|$)/g;
  let m;
  while ((m = re.exec(clean))) {
    const s = m[0];
    if (!s.trim()) { if (m.index === re.lastIndex) re.lastIndex++; continue; }
    const lead = s.length - s.trimStart().length;
    out.push({ text: s.trim(), start: m.index + lead });
    if (m.index === re.lastIndex) re.lastIndex++;
  }
  return out;
}

/** Plan one utterance: spoken string, words, phones and visemes on a nominal timeline. */
export function planUtterance(text, pron, durations) {
  const tokens = [];
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(text))) tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length });

  const words = [];
  let spoken = "";
  const lexiconRef = pron.lexicon;
  tokens.forEach((tok, ti) => {
    const sw = speakToken(tok.text, lexiconRef);
    if (!sw.length) {
      if (/[.,!?;:—–…\-]/.test(tok.text)) {
        spoken = spoken.replace(/ $/, "") + tok.text.replace(/[—–]/g, ",") + " ";
        if (words.length) words[words.length - 1].pauseAfter = (words[words.length - 1].pauseAfter || "") + tok.text;
      }
      return;
    }
    for (const w of sw) {
      const { phones, source } = pron.phones(w);
      words.push({ w, phones, source, tok: ti, charStart: spoken.length, charEnd: spoken.length + w.length,
        func: FUNCTION_WORDS.has(w), emph: tok.text.length > 2 && tok.text === tok.text.toUpperCase() && /[A-Z]/.test(tok.text) });
      spoken += w + " ";
    }
  });
  spoken = spoken.trim();

  // nominal timing
  let t = 0.05;
  const phones = [];
  words.forEach((wd, wi) => {
    wd.start = t;
    const isFinal = wi === words.length - 1 || !!wd.pauseAfter;
    let lastVowel = -1;
    wd.phones.forEach((p, k) => { if (VOWELS.has(p.replace(/\d/, ""))) lastVowel = k; });
    wd.phones.forEach((p, k) => {
      let d = durations.of(p);
      if (wd.func) d *= 0.8;
      if (wd.emph) d *= 1.15;
      if (isFinal && k >= lastVowel) d *= 1.25;
      const base = p.replace(/\d/, "");
      const stress = VOWELS.has(base) ? +(p.match(/\d/)?.[0] ?? 0) : -1;
      phones.push({ p: base, stress, t0: t, t1: t + d, word: wi });
      t += d;
    });
    wd.end = t;
    if (wd.pauseAfter && wi < words.length - 1) t += durations.pause(wd.pauseAfter);
  });
  const plan = { text, spoken, tokens, words, phones, duration: t + 0.08 };
  plan.visemes = phonesToVisemes(phones, words, plan.duration);
  return plan;
}

/** Phones (with timing) → viseme events, inserting silences in gaps. */
export function phonesToVisemes(phones, words, duration) {
  const ev = [];
  let cursor = 0;
  phones.forEach((ph, i) => {
    if (ph.t0 - cursor > 0.04) ev.push({ v: "sil", t0: cursor, t1: ph.t0, stress: -1, word: -1 });
    let v = P2V[ph.p];
    if (v === null || v === undefined) {
      v = "sil";
      for (let j = i + 1; j < phones.length; j++) {
        const n = P2V[phones[j].p];
        if (n) { v = Array.isArray(n) ? n[0] : n; break; }
      }
    }
    const emph = words && words[ph.word] && words[ph.word].emph;
    if (Array.isArray(v)) {
      const mid = ph.t0 + (ph.t1 - ph.t0) * 0.55;
      ev.push({ v: v[0], t0: ph.t0, t1: mid, stress: ph.stress, word: ph.word, emph });
      ev.push({ v: v[1], t0: mid, t1: ph.t1, stress: Math.min(ph.stress, 0), word: ph.word, emph, glide: true });
    } else {
      ev.push({ v, t0: ph.t0, t1: ph.t1, stress: ph.stress, word: ph.word, emph });
    }
    cursor = ph.t1;
  });
  ev.push({ v: "sil", t0: cursor, t1: Math.max(duration, cursor + 0.1), stress: -1, word: -1 });
  return ev;
}

/**
 * Re-time a plan using per-character alignment from the TTS (e.g. ElevenLabs
 * /with-timestamps). Each word gets its true start/end; its phones are spread
 * inside that span in proportion to their learned durations.
 */
export function retimeWithAlignment(plan, alignment, offset = 0) {
  const st = alignment.character_start_times_seconds, en = alignment.character_end_times_seconds;
  if (!st || !en || !st.length) return plan;
  const n = st.length;
  const byWord = new Map();
  plan.phones.forEach((p) => { if (!byWord.has(p.word)) byWord.set(p.word, []); byWord.get(p.word).push(p); });
  plan.words.forEach((wd, wi) => {
    const a = Math.min(wd.charStart, n - 1), b = Math.min(Math.max(wd.charEnd - 1, a), n - 1);
    const t0 = st[a] + offset, t1 = Math.max(en[b] + offset, t0 + 0.04);
    const ps = byWord.get(wi) || [];
    const tot = ps.reduce((s, p) => s + (p.t1 - p.t0), 0) || 1;
    let t = t0;
    ps.forEach((p) => { const d = ((p.t1 - p.t0) / tot) * (t1 - t0); p.t0 = t; p.t1 = t + d; t += d; });
    wd.start = t0; wd.end = t1;
  });
  plan.duration = Math.max(plan.duration, (en[n - 1] || 0) + offset + 0.1);
  plan.visemes = phonesToVisemes(plan.phones, plan.words, plan.duration);
  plan.aligned = true;
  return plan;
}

/** Locate the word whose spoken-string span contains charIndex. */
export function wordAtChar(plan, charIndex) {
  let best = -1;
  for (let i = 0; i < plan.words.length; i++) {
    if (plan.words[i].charStart <= charIndex + 1) best = i; else break;
  }
  return best;
}
