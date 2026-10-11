// Pixanva worker — jalanin engine di thread terpisah, UI tetap responsif.
// Protocol:
//   {type:'load', model}            -> {type:'progress'|'loaded'}
//   {type:'gen', cond, opts}        -> {type:'token'}* -> {type:'done'} / {type:'error'}
//   {type:'chatload'}               -> {type:'chatready', ok}
//   {type:'chat', texts, opts}      -> {type:'ctok'}* -> {type:'cdone', text} / {type:'error'}
//   {type:'imjload'}               -> {type:'imjready', ok}
//   {type:'imj', text, G, seed, opts} -> {type:'token'}* -> {type:'done'} / {type:'error'}
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
    } else if (msg.type === "imjload") {
      try {
        await ensureImajin();
        self.postMessage({ type: "imjready", ok: true, params: imj.meta.n_params });
      } catch (err) {
        self.postMessage({ type: "imjready", ok: false, msg: String(err) });
      }
    } else if (msg.type === "imj") {
      await imjGenerate(msg);
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

// ===== imajin custom-prompt — teks bebas MASUK LANGSUNG ke model gambar =====
// Format persis training v4: [BOS] kata.. [A] slot9x(latent, model sendiri yang
// prediksi) [GRID_G] [SEP] kode_gambar..
// Kata yang GAK dikenal vocab → [CHARW] + token karakter (open vocab) — jadi
// prompt NGAWUR pun tetap ngaruh ke gambar. GAK ada lagi "prompt jadi tags":
// gak ada parser, gak ada chip tag — slot-nya laten di dalam model.
let imj = null;
// token GRID = konstanta protokol training (tags.py GRID_TOKEN), GAK ikut vocab
// (vocab json emang gak nyimpen ini — makanya S.GRID bakal undefined)
const IMJ_GT = { 16: 8, 24: 9, 32: 10, 48: 11, 64: 12 };
async function ensureImajin() {
  if (imj) return imj;
  const binURL = new URL("../models/imajin12m.bin?v=v29", self.location).href;
  const { meta, W } = await fetchModel(binURL, () => {});
  const model = new Pixanva(meta, W);
  const vres = await fetch(new URL("../models/imajin_vocab.json?v=v29", self.location).href);
  if (!vres.ok) throw new Error("vocab imajin gak ketemu");
  const vj = await vres.json();
  const stoi = {};
  vj.itos.forEach((w, i) => { if (w) stoi[w] = i; });
  imj = { meta, model, stoi, S: vj.specials, CH: vj.chars || {} };
  return imj;
}
function tokTextImj(s) {
  return s.toLowerCase().match(/[a-z0-9]+/g) || [];
}
// mirror 1:1 dgn training/imajin_data.py encode() — jalur inference (tanpa typo)
function imjEncode(text) {
  const { stoi, S, CH } = imj;
  const ids = [S.BOS];
  for (const w of tokTextImj(text).slice(0, 16)) {
    const wid = stoi[w];
    if (wid !== undefined && wid >= S.WORD_BASE) { ids.push(wid); continue; }
    // kata gak dikenal / bukan kata → char fallback: [CHARW] c1..cn
    ids.push(S.CHARW);
    for (const ch of w) { const c = CH[ch]; if (c !== undefined) ids.push(c); }
    if (ids.length >= 64) break;   // mirror max_txt training
  }
  ids.push(S.A);
  return ids;
}
// slot laten: model yang nyuruh sendiri kapan slot beres (keluar token GRID)
// slot valid = tag lama 109..158 + NONE 162 + subj/attr/scol 163..192
function imjArgmaxSlot(st, model) {
  let best = -1, bv = -1e30;
  const lg = st.logits;
  for (let v = 8; v < 193; v++) {
    if (v >= 13 && v < 109) continue;        // palet — bukan slot
    if (v === 159 || v === 160 || v === 161) continue;  // marker [A]/SEC — bukan nilai slot
    if (lg[v] > bv) { bv = lg[v]; best = v; }
  }
  return best;
}
async function imjGenerate(msg) {
  const { text, G, seed, opts } = msg;
  await ensureImajin();
  const { model, S } = imj;
  const t0 = performance.now();
  const toks = imjEncode(text);
  const total = G * G;
  const useCFG = opts.guidance > 1.001;

  // prefill teks + slot laten (greedy, dihentikan token GRID — persis training)
  const stC = model.newState(toks.length + 14 + total + 2);
  let pos = 0;
  for (const t of toks) { model.step(stC, t, pos, 0); pos++; }
  let gridTok = -1;
  for (let i = 0; i < 12; i++) {
    const s = imjArgmaxSlot(stC, model);
    if (s >= 8 && s <= 12) { gridTok = s; break; }   // model bilang slot-nya udah cukup
    model.step(stC, s, pos, 0); pos++;
  }
  if (gridTok < 0) gridTok = IMJ_GT[G] || 10;        // gagal berhenti → paksa
  model.step(stC, gridTok, pos, 0); pos++;
  model.step(stC, S.SEP, pos, 0); pos++;

  // cabang uncond buat CFG (persis format training: [BOS, UNCOND, GRID, SEP])
  let stU = null;
  if (useCFG) {
    const uToks = [S.BOS, S.UNCOND, (IMJ_GT[G] || 10), S.SEP];
    stU = model.newState(uToks.length + total + 2);
    for (let i = 0; i < uToks.length; i++) model.step(stU, uToks[i], i, 0);
  }

  const rng = mulberry32(seed >>> 0);
  const recent = [];
  const CO = IDS.COLOR_OFFSET;
  const tokens = new Uint8Array(total);
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
      const cx = Math.floor(k / G), cy = (k % G) + 16;
      logitsC = model.step(stC, tok, cx, cy);
      if (stU) logitsU = model.step(stU, tok, cx, cy);
    }
    if (k % 16 === 0) self.postMessage({ type: "beat", ms: performance.now() - t0, i: k, total });
  }
  self.postMessage({ type: "done", tokens: Array.from(tokens), ms: performance.now() - t0, G });
}
