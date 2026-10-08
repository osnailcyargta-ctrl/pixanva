// Pixanva app — UI logic
import { PALETTE, TAGS, IDS } from "./data.js";

const $ = (s) => document.querySelector(s);
const DEFAULTS = {
  light: { temp: 0.95, topk: 32, cfg: 1.0 },
  dark: { temp: 0.90, topk: 24, cfg: 1.8 },
  heavy: { temp: 0.85, topk: 20, cfg: 2.2 },
};

const state = {
  model: null,          // id model aktif
  meta: null,           // meta.json
  scene: null,
  color: null,
  mood: null,
  orn: [],
  seedLock: false,
  seed: null,
  lastRun: null,        // {cond, opts} utk "ulangi seed ini"
  generating: false,
  worker: null,
  loaded: new Set(),
};

// ---------- worker ----------
function getWorker() {
  if (!state.worker) {
    state.worker = new Worker("js/worker.js", { type: "module" });
    state.worker.onmessage = onWorkerMsg;
    state.worker.onerror = (e) => setStatus("error worker: " + e.message);
  }
  return state.worker;
}

function hex(c) { return PALETTE[c]; }

function onWorkerMsg(e) {
  const m = e.data;
  if (m.type === "progress") {
    $("#loadBar").hidden = false;
    $("#loadBar > div").style.width = (m.p * 100).toFixed(1) + "%";
    setStatus(`memuat bobot ${(m.got / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB…`);
  } else if (m.type === "loaded") {
    state.loaded.add(m.model);
    $("#loadBar").hidden = true;
    setStatus("siap");
    $("#btnGen").disabled = false;
    $("#btnGen").textContent = "Generate";
  } else if (m.type === "token") {
    paintCell(m.i, m.c, currentG);
    if (m.i % 4 === 0 || m.i === m.total - 1) {
      const dt = (performance.now() - genStart) / 1000;
      $("#paintInfo").textContent = `melukis… ${m.i + 1}/${m.total} sel • ${dt.toFixed(1)}s`;
      setStatus("melukis");
    }
  } else if (m.type === "beat") {
    // keepalive opsional
  } else if (m.type === "done") {
    finishGen(m);
  } else if (m.type === "error") {
    console.error(m.msg);
    setStatus("error — cek console");
    $("#btnGen").disabled = false;
    $("#btnGen").textContent = "Generate";
    state.generating = false;
  }
}

// ---------- canvas ----------
let currentG = 32;
function setupCanvas(G) {
  currentG = G;
  const cv = $("#canvas");
  cv.width = G * 2;
  cv.height = G * 2;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0b0d11";
  ctx.fillRect(0, 0, cv.width, cv.height);
  $("#stageEmpty").style.display = "none";
}

function paintCell(i, c, G) {
  const r = Math.floor(i / G), col = i % G;
  const ctx = $("#canvas").getContext("2d");
  ctx.fillStyle = hex(c);
  ctx.fillRect(col * 2, r * 2, 2, 2);
}

function paintFull(tokens, G) {
  setupCanvas(G);
  const ctx = $("#canvas").getContext("2d");
  for (let i = 0; i < tokens.length; i++) {
    const r = Math.floor(i / G), col = i % G;
    ctx.fillStyle = hex(tokens[i]);
    ctx.fillRect(col * 2, r * 2, 2, 2);
  }
}

// ---------- generate ----------
let genStart = 0;
function startGen() {
  if (!state.scene || !state.color) {
    setStatus("pilih pemandangan & warna dulu");
    return;
  }
  if (state.generating) return;
  const model = state.model;
  if (!state.loaded.has(model)) {
    setStatus("memuat model…");
    getWorker().postMessage({ type: "load", model });
    return;
  }
  if (!state.seedLock || !state.seed) state.seed = (Math.random() * 0xffffffff) >>> 0;
  $("#seed").value = state.seed;

  const G = parseInt($("#res").value, 10);
  const d = DEFAULTS[model];
  const opts = {
    model,
    temp: parseFloat($("#temp").value),
    topk: parseInt($("#topk").value, 10),
    guidance: parseFloat($("#cfg").value),
  };
  const cond = {
    scene: state.scene, color: state.color, mood: state.mood,
    orn: [...state.orn], G, seed: state.seed,
  };
  state.lastRun = { cond, opts };
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(G);
  getWorker().postMessage({ type: "gen", cond, opts });
}

function finishGen(m) {
  state.generating = false;
  $("#btnGen").disabled = false;
  $("#btnGen").textContent = "Generate";
  $("#btnDownload").disabled = false;
  $("#btnAgain").disabled = false;
  const dt = (m.ms / 1000).toFixed(1);
  $("#paintInfo").textContent = `${m.G * m.G} sel • ${dt}s • seed ${state.seed}`;
  setStatus("selesai");
  saveGallery(m);
}

// ---------- galeri ----------
function galKey() { return "pixanva_gallery_v1"; }
function loadGallery() {
  try { return JSON.parse(localStorage.getItem(galKey()) || "[]"); }
  catch { return []; }
}
function saveGallery(m) {
  const items = loadGallery();
  const cv = $("#canvas");
  const up = document.createElement("canvas");
  const scale = Math.max(2, 256 / cv.width);
  up.width = cv.width * scale; up.height = cv.height * scale;
  const uc = up.getContext("2d");
  uc.imageSmoothingEnabled = false;
  uc.drawImage(cv, 0, 0, up.width, up.height);
  items.unshift({
    img: up.toDataURL("image/png"),
    prompt: promptText(state.lastRun.cond),
    model: state.model,
    seed: state.seed,
    G: m.G,
    ts: Date.now(),
  });
  while (items.length > 24) items.pop();
  try { localStorage.setItem(galKey(), JSON.stringify(items)); } catch {}
  renderGallery();
}
function renderGallery() {
  const items = loadGallery();
  const el = $("#gallery");
  $("#galCount").textContent = items.length ? `(${items.length})` : "";
  if (!items.length) {
    el.innerHTML = '<div class="gallery-empty">Belum ada karya.<br>Generate yang pertama.</div>';
    return;
  }
  el.innerHTML = "";
  for (const it of items) {
    const img = document.createElement("img");
    img.src = it.img;
    img.title = `${it.prompt}\nmodel: ${it.model} • seed ${it.seed}`;
    img.onclick = () => { paintFromDataURL(it.img, it.G); $("#paintInfo").textContent = it.prompt; };
    el.appendChild(img);
  }
}
function paintFromDataURL(src, G) {
  setupCanvas(G);
  const im = new Image();
  im.onload = () => {
    const ctx = $("#canvas").getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(im, 0, 0, G * 2, G * 2);
  };
  im.src = src;
}

function promptText(cond) {
  const parts = [labelOf("scenes", cond.scene), labelOf("colors", cond.color)];
  if (cond.mood) parts.push(labelOf("moods", cond.mood));
  for (const o of cond.orn) parts.push(labelOf("ornaments", o));
  return parts.join(" • ");
}
function labelOf(group, id) {
  const t = TAGS[group].find((x) => x.id === id);
  return t ? t.label : id;
}

// ---------- model selector ----------
async function initModels() {
  const res = await fetch("models/meta.json");
  state.meta = await res.json();
  const list = $("#modelList");
  list.innerHTML = "";
  for (const m of state.meta.models) {
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.id = m.id;
    card.innerHTML = `
      <div class="mc-name">${m.label} <span class="dot"></span></div>
      <div class="mc-desc">${m.desc}</div>
      <div class="mc-meta">${(m.params / 1e6).toFixed(1)}M param • ${m.L}L × ${m.H}H • d=${m.d} • step ${m.step}${m.val_loss ? ` • val ${m.val_loss.toFixed(2)}` : ""}</div>`;
    card.onclick = () => selectModel(m.id);
    list.appendChild(card);
  }
  selectModel("dark");
}

function selectModel(id) {
  state.model = id;
  document.querySelectorAll(".model-card").forEach((c) =>
    c.classList.toggle("active", c.dataset.id === id));
  const m = state.meta.models.find((x) => x.id === id);
  $("#badgeModel").textContent = m.label;
  $("#badgeInfo").textContent = `${(m.params / 1e6).toFixed(1)}M parameter`;
  const d = DEFAULTS[id];
  $("#temp").value = d.temp; $("#tempVal").textContent = d.temp;
  $("#topk").value = d.topk; $("#topkVal").textContent = d.topk;
  $("#cfg").value = d.cfg; $("#cfgVal").textContent = d.cfg;
  state.loaded.delete(id); // force reload meta utk model ini kalau berganti
  if (state.worker) {
    getWorker().postMessage({ type: "load", model: id });
  }
}

// ---------- chips ----------
function chip(group, el, item, multi, max) {
  const b = document.createElement("button");
  b.className = "chip";
  if (group === "colors") {
    const idx = IDS.COLOR_IDS[item.id] - IDS.COLOR_OFFSET;
    const picks = [idx, (idx + 31) % 96, (idx + 63) % 96];
    b.innerHTML = `<span class="sw">${picks.map((p) => `<i style="background:${PALETTE[p]}"></i>`).join("")}</span>${item.label}`;
  } else {
    b.textContent = item.label;
  }
  b.onclick = () => {
    if (multi) {
      const has = state[group].includes(item.id);
      if (has) state[group] = state[group].filter((x) => x !== item.id);
      else {
        if (state[group].length >= max) return;
        state[group].push(item.id);
      }
      b.classList.toggle("on", !has);
    } else {
      state[group] = state[group] === item.id ? null : item.id;
      el.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
      if (state[group] === item.id) b.classList.add("on");
    }
  };
  el.appendChild(b);
}

function initChips() {
  for (const s of TAGS.scenes) chip("scene", $("#chipsScene"), s, false);
  for (const c of TAGS.colors) chip("color", $("#chipsColor"), c, false);
  for (const mm of TAGS.moods) chip("mood", $("#chipsMood"), mm, false);
  state.scene = TAGS.scenes[0].id;
  $("#chipsScene").firstChild.classList.add("on");
  state.color = TAGS.colors[0].id;
  $("#chipsColor").firstChild.classList.add("on");
  for (const o of TAGS.ornaments) chip("orn", $("#chipsOrn"), o, true, 3);
}

// ---------- misc ----------
function setStatus(s) { $("#statusLine").textContent = s; }

function initEvents() {
  $("#btnGen").onclick = startGen;
  $("#btnDice").onclick = () => {
    state.seed = (Math.random() * 0xffffffff) >>> 0;
    $("#seed").value = state.seed;
  };
  $("#btnLock").onclick = () => {
    state.seedLock = !state.seedLock;
    $("#btnLock").classList.toggle("locked", state.seedLock);
    $("#btnLock").textContent = state.seedLock ? "🔒" : "🔓";
    if (state.seedLock && !state.seed) {
      state.seed = (Math.random() * 0xffffffff) >>> 0;
      $("#seed").value = state.seed;
    }
  };
  $("#seed").oninput = () => {
    const v = parseInt($("#seed").value, 10);
    if (!isNaN(v)) state.seed = v >>> 0;
  };
  $("#btnSkipOrn").onclick = () => {
    state.orn = [];
    $("#chipsOrn").querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
  };
  $("#btnDownload").onclick = () => {
    const cv = $("#canvas");
    const a = document.createElement("a");
    a.download = `pixanva_${state.model}_${state.seed}.png`;
    const up = document.createElement("canvas");
    up.width = cv.width * 4; up.height = cv.height * 4;
    const uc = up.getContext("2d");
    uc.imageSmoothingEnabled = false;
    uc.drawImage(cv, 0, 0, up.width, up.height);
    a.href = up.toDataURL("image/png");
    a.click();
  };
  $("#btnAgain").onclick = () => {
    if (!state.lastRun) return;
    state.seed = state.lastRun.cond.seed; // seed sama → hasil sama
    state.generating = false;
    startGenWith(state.lastRun.cond, state.lastRun.opts);
  };
  $("#temp").oninput = () => $("#tempVal").textContent = $("#temp").value;
  $("#topk").oninput = () => $("#topkVal").textContent = $("#topk").value;
  $("#cfg").oninput = () => $("#cfgVal").textContent = $("#cfg").value;
  $("#btnAbout").onclick = () => $("#aboutModal").hidden = false;
  $("#btnCloseAbout").onclick = () => $("#aboutModal").hidden = true;
  $("#aboutModal").onclick = (e) => { if (e.target === $("#aboutModal")) $("#aboutModal").hidden = true; };
  $("#btnClear").onclick = () => {
    localStorage.removeItem(galKey());
    renderGallery();
  };
}

function startGenWith(cond, opts) {
  if (state.generating) return;
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(cond.G);
  getWorker().postMessage({ type: "gen", cond, opts });
}

// ---------- init ----------
(async function init() {
  initChips();
  initEvents();
  renderGallery();
  try {
    await initModels();
    getWorker().postMessage({ type: "load", model: state.model });
  } catch (e) {
    setStatus("gagal memuat meta: " + e.message);
  }
})();
