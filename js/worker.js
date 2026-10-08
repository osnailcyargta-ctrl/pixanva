// Pixanva worker — jalanin engine di thread terpisah, UI tetap responsif.
// Protocol:
//   {type:'load', model}            -> {type:'progress'|'loaded'}
//   {type:'gen', cond, opts}        -> {type:'token'}* -> {type:'done'} / {type:'error'}
import { fetchModel, Pixanva, mulberry32, sampleToken } from "./engine.js";
import { IDS } from "./data.js";

const models = {};

async function ensureLoaded(id, onProgress) {
  if (models[id]) return models[id];
  const { meta, W } = await fetchModel(`models/${id}.bin`, onProgress);
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
    }
  } catch (err) {
    self.postMessage({ type: "error", msg: String((err && err.stack) || err) });
  }
};
