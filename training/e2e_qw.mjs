// E2E test rilis QW: dark12 & heavyqw vs dark & heavy (QQ)
// Mirror logika js/worker.js: prefill cond -> CFG mix -> sampleToken(recent, 1.12)
// Output: JSON token grids buat dirender comparison PNG.
import { readFileSync, writeFileSync } from "node:fs";
import { fetchModel, Pixanva, mulberry32, sampleToken } from "../js/engine.js";
import { IDS } from "../js/data.js";

// ---- shim fetch: serve file lokal lewat Response ----
const REPO = new URL("../", import.meta.url); // .../pixanva/
globalThis.fetch = async (url) => {
  const path = new URL(url, REPO).pathname;
  const buf = readFileSync(path);
  return new Response(buf, { headers: { "content-length": String(buf.length) } });
};

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

async function generate(model, cond, opts) {
  const G = cond.G;
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
  for (let i = 0; i < nCond; i++) {
    const [px, py] = condPos(i);
    model.step(stC, toks[i], px, py);
  }
  if (stU) for (let i = 0; i < uToks.length; i++) {
    const [px, py] = condPos(i);
    model.step(stU, uToks[i], px, py);
  }
  const rng = mulberry32(cond.seed >>> 0);
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
    } else logits = logitsC;
    const tok = sampleToken(logits, opts.temp, opts.topk, rng, recent, 1.12);
    const color = tok - CO;
    tokens[k] = color;
    recent.push(tok);
    if (recent.length > 12) recent.shift();
    if (k < total - 1) {
      const [cx, cy] = imgPos(k, G);
      logitsC = model.step(stC, tok, cx, cy);
      if (stU) { const [ux, uy] = imgPos(k, G); logitsU = model.step(stU, tok, ux, uy); }
    }
  }
  return tokens;
}

const PROMPTS = [
  { scene: "gunung", color: "hangat", mood: "senja", orn: ["burung"], label: "gunung hangat senja" },
  { scene: "gurun", color: "hangat", mood: "siang", orn: ["matahari"], label: "gurun hangat siang" },
  { scene: "kota", color: "neon", mood: "malam", orn: ["awan"], label: "kota neon malam" },
  { scene: "laut", color: "dingin", mood: "senja", orn: ["bulan", "perahu"], label: "laut dingin senja" },
];

const CASES = [
  { id: "dark", file: "models/dark.bin", guidance: 1.7 },
  { id: "dark12", file: "models/dark12.bin", guidance: 1.7 },
  { id: "heavy", file: "models/heavy.bin", guidance: 1.7 },
  { id: "heavyqw", file: "models/heavyqw.bin", guidance: 1.7 },
];

const out = [];
for (const cs of CASES) {
  const t0 = Date.now();
  const { meta, W } = await fetchModel(cs.file);
  const model = new Pixanva(meta, W);
  console.log(`[${cs.id}] load ${(Date.now() - t0) / 1000}s — ${meta.n_params.toLocaleString()} params, step ${meta.step}`);
  for (let i = 0; i < PROMPTS.length; i++) {
    const p = PROMPTS[i];
    const cond = { ...p, G: 16, seed: 77000 + i * 101 };
    const tg = performance.now();
    const tokens = await generate(model, cond, { temp: 0.85, topk: 24, guidance: cs.guidance });
    const uniq = new Set(tokens).size;
    out.push({ model: cs.id, prompt: p.label, seed: cond.seed, G: 16, tokens: Array.from(tokens) });
    console.log(`  "${p.label}" uniq warna=${uniq} (${((performance.now() - tg) / 1000).toFixed(1)}s)`);
  }
}
writeFileSync(new URL("./e2e_qw_out.json", import.meta.url), JSON.stringify(out));
console.log("OK ->", new URL("./e2e_qw_out.json", import.meta.url).pathname);
