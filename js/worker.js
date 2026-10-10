// Pixanva worker — jalanin engine di thread terpisah, UI tetap responsif.
// Protocol:
//   {type:'load', model}            -> {type:'progress'|'loaded'}
//   {type:'gen', cond, opts}        -> {type:'token'}* -> {type:'done'} / {type:'error'}
//   {type:'chatload'}               -> {type:'chatready', ok}
//   {type:'chat', texts, opts}      -> {type:'ctok'}* -> {type:'cdone', text} / {type:'error'}
//   {type:'pmload'}                 -> {type:'pmready', ok}
//   {type:'pm', text}               -> {type:'pmout', ok, scene, color, mood, orn}
import { fetchModel, Pixanva, mulberry32, sampleToken } from "./engine.js";
import { IDS } from "./data.js";

const models = {};

async function ensureLoaded(id, onProgress) {
  if (models[id]) return models[id];
  // PENTING: URL relatif di worker di-resolve thd lokasi SCRIPT worker (js/),
  // bukan halaman — jadi harus naik 1 folder eksplisit.
  const url = new URL(`../models/${id}.bin`, self.location).href;
  const { meta, W } = await fetchModel(url, onProgress);
  const model = new Pixanva(meta, W);
  models[id] = { meta, W, model };
  return models[id];
}

const condPos = (i) => [i, 0];
const imgPos = (k, G) => [Math.floor(k / G), (k % G) + 16];

function buildCondTokens(cond) {
  const S = IDS.SPECIALS;
  if (cond.uncond) return [S.BOS, S.UNCOND, IDS.GRID_TOKEN[String(cond.G)], S.SEP];
  const t = [S.BOS, S.SEC_SCENE, IDS.SCENE_IDS[cond.scene],
             S.SEC_COLOR, IDS.COLOR_IDS[cond.color]];
  if (cond.mood) t.push(S.SEC_MOOD, IDS.MOOD_IDS[cond.mood]);
  for (const o of cond.orn || []) t.push(S.SEC_ORN, IDS.ORN_IDS[o]);
  t.push(IDS.GRID_TOKEN[String(cond.G)], S.SEP);
  return t;
}

async function generate(msg) {
  const { cond, opts } = msg;
  const G = cond.G;
  const m = await ensureLoaded(opts.model, () => {});
  const model = m.model;
  const t0 = performance.now();

  const toks = buildCondTokens(cond);
  const nCond = toks.length;
  const total = G * G;

  const useCFG = opts.guidance > 1.001;
  const stC = model.newState(nCond + total + 2);
  let stU = null, uToks = null;
  if (useCFG) {
    uToks = buildCondTokens({ ...cond, uncond: true });
    stU = model.newState(uToks.length + total + 2);
  }

  // prefill kondisi
  for (let i = 0; i < nCond; i++) {
    const [px, py] = condPos(i);
    model.step(stC, toks[i], px, py);
  }
  if (stU) {
    for (let i = 0; i < uToks.length; i++) {
      const [px, py] = condPos(i);
      model.step(stU, uToks[i], px, py);
    }
  }

  const rng = mulberry32(cond.seed >>> 0);
  const recent = [];
  const CO = IDS.COLOR_OFFSET;
  const tokens = new Uint8Array(total);
  const nRep = opts.guidance > 1.001 ? 2 : 1;

  // loop melukis: sample -> feed balik
  let logitsC = stC.logits, logitsU = stU ? stU.logits : null;
  const mixed = useCFG ? new Float32Array(model.V) : null;

  for (let k = 0; k < total; k++) {
    let logits;
    if (useCFG) {
      for (let v = 0; v < model.V; v++)
        mixed[v] = logitsU[v] + opts.guidance * (logitsC[v] - logitsU[v]);
      logits = mixed;
    } else {
      logits = logitsC;
    }
    const tok = sampleToken(logits, opts.temp, opts.topk, rng, recent, 1.12);
    const color = tok - CO;
    tokens[k] = color;
    recent.push(tok);
    if (recent.length > 12) recent.shift();

    self.postMessage({ type: "token", i: k, c: color, total });

    if (k < total - 1) {
      const [cx, cy] = imgPos(k, G);
      logitsC = model.step(stC, tok, cx, cy);
      if (stU) {
        const [ux, uy] = imgPos(k, G);
        logitsU = model.step(stU, tok, ux, uy);
      }
    }
    if (k % 16 === 0) self.postMessage({ type: "beat", ms: performance.now() - t0, i: k, total });
  }
  self.postMessage({ type: "done", tokens: Array.from(tokens), ms: performance.now() - t0, G });
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      const m = await ensureLoaded(msg.model, (p, got, total) =>
        self.postMessage({ type: "progress", p, got, total, model: msg.model }));
      self.postMessage({ type: "loaded", model: msg.model, params: m.meta.n_params, step: m.meta.step });
    } else if (msg.type === "gen") {
      await generate(msg);
    } else if (msg.type === "chatload") {
      try {
        await ensureChat();
        self.postMessage({ type: "chatready", ok: true, params: chat.meta.n_params });
      } catch (err) {
        self.postMessage({ type: "chatready", ok: false, msg: String(err) });
      }
    } else if (msg.type === "chat") {
      await chatGenerate(msg);
    } else if (msg.type === "pmload") {
      try {
        await ensurePrompter();
        self.postMessage({ type: "pmready", ok: true, params: pmer.meta.n_params });
      } catch (err) {
        self.postMessage({ type: "pmready", ok: false, msg: String(err) });
      }
    } else if (msg.type === "pm") {
      await prompterParse(msg);
    }
  } catch (err) {
    self.postMessage({ type: "error", msg: String((err && err.stack) || err) });
  }
};

// ===== asisten chat (LLM text dari nol) =====
let chat = null;
async function ensureChat() {
  if (chat) return chat;
  // URL relatif di worker di-resolve thd js/ — naik 1 folder eksplisit
  const binURL = new URL("../models/assistant.bin", self.location).href;
  const { meta, W } = await fetchModel(binURL, () => {});
  const model = new Pixanva(meta, W);
  const vres = await fetch(new URL("../models/chat_vocab.json", self.location).href);
  if (!vres.ok) throw new Error("vocab gak ketemu");
  const vj = await vres.json();
  const stoi = {};
  vj.itos.forEach((w, i) => { if (w) stoi[w] = i; });
  chat = { meta, model, itos: vj.itos, stoi, S: vj.specials };
  return chat;
}
// tokenizer HARUS mirror 1:1 dgn training/chat_data.py
function tokText(s) {
  return s.toLowerCase().match(/[a-z0-9]+|[•=.,:?!-]/g) || [];
}
async function chatGenerate(msg) {
  const { model, itos, stoi, S } = await ensureChat();
  const t0 = performance.now();
  // konteks max 2 user msg terakhir, digabung " sep " (sama kayak format training)
  const texts = (msg.texts || []).slice(-2);
  const uw = tokText(texts.join(" sep "));
  const maxNew = 120;
  const ids = [S.BOS, S.U];
  for (const w of uw) {
    if (stoi[w] !== undefined) ids.push(stoi[w]);
  }
  ids.push(S.A);
  const st = model.newState(ids.length + maxNew + 2);
  let pos = 0;
  for (const t of ids) { model.step(st, t, pos, 0); pos++; }
  const rng = mulberry32((msg.seed || 1) >>> 0);
  const recent = [];
  const out = [];
  for (let i = 0; i < maxNew; i++) {
    const tok = sampleToken(st.logits, msg.temp || 0.82, msg.topk || 24, rng, recent, 1.12);
    if (tok === S.EOS) break;
    recent.push(tok);
    if (recent.length > 24) recent.shift();
    const w = itos[tok];
    if (w) { out.push(w); self.postMessage({ type: "ctok", w }); }
    if (i < maxNew - 1) { model.step(st, tok, pos, 0); pos++; }
  }
  self.postMessage({ type: "cdone", text: out.join(" "), ms: performance.now() - t0 });
}

// ===== prompter custom-prompt (teks bebas → slot tag) =====
// Arsitektur sama dgn chat LLM, cuma output-nya 5 slot fix: scene→color→mood→orn1→orn2.
let pmer = null;
async function ensurePrompter() {
  if (pmer) return pmer;
  const binURL = new URL("../models/prompter.bin?v=v22", self.location).href;
  const { meta, W } = await fetchModel(binURL, () => {});
  const model = new Pixanva(meta, W);
  const vres = await fetch(new URL("../models/prompt_vocab.json", self.location).href);
  if (!vres.ok) throw new Error("vocab prompter gak ketemu");
  const vj = await vres.json();
  const stoi = {};
  vj.itos.forEach((w, i) => { if (w) stoi[w] = i; });
  pmer = { meta, model, itos: vj.itos, stoi, S: vj.specials };
  return pmer;
}
function tokTextPM(s) {
  return s.toLowerCase().match(/[a-z0-9]+/g) || [];
}
async function prompterParse(msg) {
  try {
    const { model, itos, stoi, S } = await ensurePrompter();
    const t0 = performance.now();
    const words = tokTextPM(msg.text || "");
    const ids = [S.BOS];
    for (const w of words) {
      if (stoi[w] !== undefined) ids.push(stoi[w]);
    }
    ids.push(S.A);
    const st = model.newState(ids.length + 8);
    let pos = 0;
    for (const t of ids) { model.step(st, t, pos, 0); pos++; }
    // 5 slot greedy — konsisten & cepat (T pendek, argmax paling stabil)
    const out = [];
    for (let i = 0; i < 5; i++) {
      const lg = st.logits;
      let best = 0, bv = -1e30;
      for (let v = 0; v < model.V; v++) {
        if (lg[v] > bv) { bv = lg[v]; best = v; }
      }
      out.push(itos[best] || "");
      if (i < 4) { model.step(st, best, pos, 0); pos++; }
    }
    const tag = (w) => (w && w[0] === "@" && w !== "@none") ? w.slice(1) : null;
    self.postMessage({
      type: "pmout", ok: true, ms: performance.now() - t0,
      scene: tag(out[0]), color: tag(out[1]),
      mood: tag(out[2]),
      orn: [tag(out[3]), tag(out[4])].filter(Boolean),
    });
  } catch (err) {
    self.postMessage({ type: "pmout", ok: false, msg: String(err) });
  }
}
