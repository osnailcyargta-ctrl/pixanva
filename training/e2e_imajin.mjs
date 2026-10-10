// E2E Pixanva Imajin v4 — mirror 1:1 dgn worker.js imjGenerate (jalur produksi asli):
// [BOS] kata..(char fallback) [A] slot-laten(auto-greedy s/d GRID) [GRID] [SEP] + CFG.
// Output: e2e_imajin_out.json (tokens + stats) → dirender via render_tokens.py
import { fetchModel, Pixanva, mulberry32, sampleToken } from "../js/engine.js";
import { readFileSync, writeFileSync } from "node:fs";

// ---- shim fetch: serve file lokal lewat Response (node gak support file://) ----
const REPO = new URL("../", import.meta.url);
globalThis.fetch = async (url) => {
  const path = new URL(url, REPO).pathname;
  const buf = readFileSync(path);
  return new Response(buf, { headers: { "content-length": String(buf.length) } });
};

const size = process.argv[2] || "5m";
const modelId = `imajin${size}`;

const { meta, W } = await fetchModel(new URL(`../models/${modelId}.bin`, import.meta.url).href, () => {});
const model = new Pixanva(meta, W);
const vj = JSON.parse(readFileSync(new URL("../models/imajin_vocab.json", import.meta.url), "utf8"));
const stoi = {};
vj.itos.forEach((w, i) => { if (w) stoi[w] = i; });
const S = vj.specials;
const CH = vj.chars || {};
const GT = vj.grid_token;          // { "16": 8, ... }

function tokText(s) {
  return s.toLowerCase().match(/[a-z0-9]+/g) || [];
}
// mirror worker.js imjEncode — kata gak dikenal → [CHARW] + char
function encode(text) {
  const ids = [S.BOS];
  for (const w of tokText(text).slice(0, 16)) {
    const wid = stoi[w];
    if (wid !== undefined && wid >= S.WORD_BASE) { ids.push(wid); continue; }
    ids.push(S.CHARW);
    for (const ch of w) { const c = CH[ch]; if (c !== undefined) ids.push(c); }
    if (ids.length >= 64) break;
  }
  ids.push(S.A);
  return ids;
}
// mirror worker.js imjArgmaxSlot
function argmaxSlot(st) {
  let best = -1, bv = -1e30;
  const lg = st.logits;
  for (let v = 8; v < 193; v++) {
    if (v >= 13 && v < 109) continue;
    if (v === 159 || v === 160 || v === 161) continue;
    if (lg[v] > bv) { bv = lg[v]; best = v; }
  }
  return best;
}

const CASES = [
  "ikan terbang di volkano",
  "naga api raksasa di angkasa malam",
  "kucing neon di kota malam ada petir",
  "burung emas di pantai senja",
  "robot es raksasa di tundra salju",
  "kupu kupu besar di ladang bunga pagi",
  "kucing ninja naik roket",            // kata ngawur → char fallback
  "pantai senja ada perahu",            // landscape polos
  "gw mau gambar keren bro",            // vague
];

const G = parseInt(process.argv[3] || "16", 10);
const CFGS = parseFloat(process.argv[4] || "1.6"), TEMP = 0.9, TOPK = 24;
const out = [];
for (const text of CASES) {
  const t0 = performance.now();
  const toks = encode(text);
  const total = G * G;
  // prefill + slot laten (greedy s/d GRID — persis worker)
  const stC = model.newState(toks.length + 14 + total + 2);
  let pos = 0;
  for (const t of toks) { model.step(stC, t, pos, 0); pos++; }
  const slotsGot = [];
  let gridTok = -1;
  for (let i = 0; i < 12; i++) {
    const s = argmaxSlot(stC);
    if (s >= 8 && s <= 12) { gridTok = s; break; }
    slotsGot.push(vj.itos[s] || `#${s}`);
    model.step(stC, s, pos, 0); pos++;
  }
  if (gridTok < 0) gridTok = GT[String(G)];
  model.step(stC, gridTok, pos, 0); pos++;
  model.step(stC, S.SEP, pos, 0); pos++;

  const uToks = [S.BOS, S.UNCOND, GT[String(G)], S.SEP];
  const stU = model.newState(uToks.length + total + 2);
  for (let i = 0; i < uToks.length; i++) model.step(stU, uToks[i], i, 0);

  const rng = mulberry32((12345 + text.length * 7) >>> 0);
  const recent = [];
  const tokens = new Uint8Array(total);
  let logitsC = stC.logits, logitsU = stU.logits;
  const mixed = new Float32Array(model.V);
  for (let k = 0; k < total; k++) {
    for (let v = 0; v < model.V; v++) mixed[v] = logitsU[v] + CFGS * (logitsC[v] - logitsU[v]);
    const tk = sampleToken(mixed, TEMP, TOPK, rng, recent, 1.12);
    tokens[k] = tk - 13;
    recent.push(tk);
    if (recent.length > 12) recent.shift();
    if (k < total - 1) {
      logitsC = model.step(stC, tk, Math.floor(k / G), (k % G) + 16);
      logitsU = model.step(stU, tk, Math.floor(k / G), (k % G) + 16);
    }
  }
  const uniq = new Set(tokens).size;
  out.push({ text, G, tokens: Array.from(tokens), uniq, slots: slotsGot,
             ms: Math.round(performance.now() - t0) });
  console.log(`[${modelId}] "${text}" → slot[${slotsGot.join(",")}] ${uniq} warna unik, ${out.at(-1).ms}ms`);
}
writeFileSync(new URL("./e2e_imajin_out.json", import.meta.url), JSON.stringify({ G, cases: out }, null, 1));
console.log("OK e2e_imajin_out.json");
