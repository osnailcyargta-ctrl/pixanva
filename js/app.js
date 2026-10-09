// Pixanva app — UI logic
import { PALETTE, TAGS, IDS } from "./data.js";

const $ = (s) => document.querySelector(s);
const DEFAULTS = {
  light: { temp: 0.95, topk: 32, cfg: 1.3 },
  dark: { temp: 0.90, topk: 24, cfg: 1.6 },
  dark12: { temp: 0.90, topk: 24, cfg: 1.7 },
  heavy: { temp: 0.85, topk: 24, cfg: 1.6 },
  heavyqw: { temp: 0.85, topk: 24, cfg: 1.7 },
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
    // ?v= rilis — bust cache CDN/browser tiap deploy (Pages cache 10 menit)
    state.worker = new Worker("js/worker.js?v=qw1", { type: "module" });
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
    if (m.i % 6 === 0 || m.i === m.total - 1) {
      blitGrid(curSmooth);
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
// Kanvas display selalu berukuran output final (out px). Token dilukis dulu ke
// buffer grid (2px/sel) lalu di-blit — jadi 32/64/96 native 1:1, 128 dari grid 48
// dihaluskan (bilinear + kuantisasi palet + dither) di akhir.
let currentG = 32;
let outPx = 64;
let curSmooth = false;
let gridCv = null, gridCtx = null;

function resolveGrid(model, out) {
  const m = state.meta.models.find((x) => x.id === model);
  const g48 = !!(m && m.g48);
  if (out <= 32) return { G: 16, out: 32, smooth: false };
  if (out <= 64) return { G: 32, out: 64, smooth: false };
  if (out <= 96) return g48 ? { G: 48, out: 96, smooth: false } : { G: 32, out: 96, smooth: false };
  return g48 ? { G: 48, out: 128, smooth: true } : { G: 32, out: 128, smooth: false };
}

function setupCanvas(G, out, smooth) {
  currentG = G; outPx = out; curSmooth = !!smooth;
  const cv = $("#canvas");
  cv.width = out; cv.height = out;
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#0b0d11";
  ctx.fillRect(0, 0, out, out);
  gridCv = document.createElement("canvas");
  gridCv.width = G * 2; gridCv.height = G * 2;
  gridCtx = gridCv.getContext("2d");
  gridCtx.fillStyle = "#0b0d11";
  gridCtx.fillRect(0, 0, G * 2, G * 2);
  $("#stageEmpty").style.display = "none";
}

function paintCell(i, c, G) {
  if (!gridCtx) return;
  const r = Math.floor(i / G), col = i % G;
  gridCtx.fillStyle = hex(c);
  gridCtx.fillRect(col * 2, r * 2, 2, 2);
}

function blitGrid(smooth) {
  if (!gridCv) return;
  const cv = $("#canvas"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = !!smooth;
  ctx.drawImage(gridCv, 0, 0, cv.width, cv.height);
}

// --- upscale G48 -> 128: bilinear antar sel + kuantisasi palet + dither bayer ---
const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
let PAL_RGB = null;
function palRGB() {
  if (PAL_RGB) return PAL_RGB;
  PAL_RGB = PALETTE.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  return PAL_RGB;
}
function nearestPal(r, g, b) {
  const pal = palRGB();
  let bi = 0, bd = 1e9;
  for (let i = 0; i < pal.length; i++) {
    const dr = r - pal[i][0], dg = g - pal[i][1], db = b - pal[i][2];
    const d = dr * dr + dg * dg + db * db;
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
}
function upscaleBand(tokens, G, out, y0, y1) {
  const ctx = $("#canvas").getContext("2d");
  const img = ctx.createImageData(out, y1 - y0);
  const pal = palRGB();
  const grid = new Uint8Array(G * G);
  for (let i = 0; i < tokens.length; i++) grid[i] = tokens[i];
  for (let y = y0; y < y1; y++) {
    const fy = Math.min(G - 0.001, Math.max(0, (y + 0.5) * G / out - 0.5));
    const y0g = Math.floor(fy), ty = fy - y0g, y1g = Math.min(G - 1, y0g + 1);
    for (let x = 0; x < out; x++) {
      const fx = Math.min(G - 0.001, Math.max(0, (x + 0.5) * G / out - 0.5));
      const x0g = Math.floor(fx), tx = fx - x0g, x1g = Math.min(G - 1, x0g + 1);
      const c00 = pal[grid[y0g * G + x0g]], c10 = pal[grid[y0g * G + x1g]];
      const c01 = pal[grid[y1g * G + x0g]], c11 = pal[grid[y1g * G + x1g]];
      const dth = (BAYER[y & 3][x & 3] / 16 - 0.469) * 42;
      const o = (y - y0) * out + x;
      const r = c00[0] * (1 - tx) * (1 - ty) + c10[0] * tx * (1 - ty) + c01[0] * (1 - tx) * ty + c11[0] * tx * ty + dth;
      const g = c00[1] * (1 - tx) * (1 - ty) + c10[1] * tx * (1 - ty) + c01[1] * (1 - tx) * ty + c11[1] * tx * ty + dth;
      const b = c00[2] * (1 - tx) * (1 - ty) + c10[2] * tx * (1 - ty) + c01[2] * (1 - tx) * ty + c11[2] * tx * ty + dth;
      const pi = nearestPal(r, g, b);
      img.data[o * 4] = pal[pi][0]; img.data[o * 4 + 1] = pal[pi][1];
      img.data[o * 4 + 2] = pal[pi][2]; img.data[o * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, y0);
}
function animateUpscale(tokens, G, out, onDone) {
  const bands = 8, bh = Math.ceil(out / bands);
  let b = 0;
  const step = () => {
    const y0 = b * bh, y1 = Math.min(out, y0 + bh);
    upscaleBand(tokens, G, out, y0, y1);
    b++;
    setStatus("menambah detail…");
    if (b < bands) setTimeout(step, 55);
    else onDone();
  };
  step();
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

  const rg = resolveGrid(model, parseInt($("#res").value, 10));
  const d = DEFAULTS[model];
  const opts = {
    model,
    temp: parseFloat($("#temp").value),
    topk: parseInt($("#topk").value, 10),
    guidance: parseFloat($("#cfg").value),
  };
  const cond = {
    scene: state.scene, color: state.color, mood: state.mood,
    orn: [...state.orn], G: rg.G, seed: state.seed,
  };
  state.lastRun = { cond, opts };
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(rg.G, rg.out, rg.smooth);
  getWorker().postMessage({ type: "gen", cond, opts });
}

function finishGen(m) {
  state.generating = false;
  $("#btnGen").disabled = false;
  $("#btnGen").textContent = "Generate";
  $("#btnDownload").disabled = false;
  $("#btnAgain").disabled = false;
  const dt = (m.ms / 1000).toFixed(1);
  const done = () => {
    $("#paintInfo").textContent = `${outPx}×${outPx}px • ${m.G * m.G} sel • ${dt}s • seed ${state.seed}`;
    setStatus("selesai");
    saveGallery(m);
  };
  if (curSmooth && m.G * 2 !== outPx) {
    animateUpscale(m.tokens, m.G, outPx, done);
  } else {
    blitGrid(false);
    done();
  }
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
    out: outPx,
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
    img.onclick = () => { paintFromDataURL(it.img, it.G, it.out || it.G * 2); $("#paintInfo").textContent = it.prompt; };
    el.appendChild(img);
  }
}
function paintFromDataURL(src, G, out) {
  setupCanvas(G, out || G * 2, out === 128 && G === 48);
  const im = new Image();
  im.onload = () => {
    const ctx = $("#canvas").getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(im, 0, 0, G * 2 >= out ? out : out, out);
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
function modelById(id) { return state.meta.models.find((x) => x.id === id); }
function fmtMeta(m) {
  return `${(m.params / 1e6).toFixed(1)}M param • ${m.L}L × ${m.H}H • d=${m.d}` +
    (m.val_loss ? ` • val ${m.val_loss.toFixed(2)}` : "");
}

async function initModels() {
  const res = await fetch("models/meta.json");
  state.meta = await res.json();
  selectModel("heavyqw"); // default: rilis terbaru (sekaligus render kartu + popup)
  showStep(1);
}

// kartu model aktif di sidebar
function renderModelNow() {
  const m = modelById(state.model);
  if (!m) return;
  $("#modelNow").innerHTML = `
    <div class="model-card active">
      <div class="mc-name">${m.label}${m.isNew ? ' <span class="tag-new">BARU</span>' : ""}</div>
      <div class="mc-desc">${m.desc}</div>
      <div class="mc-meta">${fmtMeta(m)} • step ${m.step}</div>
    </div>`;
}

// isi popup More models — terbaru dulu, model lama tetap bisa dipilih
function renderModelsModal() {
  const list = $("#mmList");
  list.innerHTML = "";
  for (const m of [...state.meta.models].reverse()) {
    const card = document.createElement("div");
    card.className = "mm-item" + (m.id === state.model ? " active" : "");
    card.innerHTML = `
      <div class="mm-top">
        <span class="mm-name">${m.label}</span>
        ${m.isNew ? '<span class="tag-new">BARU</span>' : ""}
        ${m.id === state.model ? '<span class="tag-cur">aktif</span>' : ""}
      </div>
      <div class="mc-desc">${m.desc}</div>
      <div class="mc-meta">${fmtMeta(m)} • step ${m.step}</div>`;
    card.onclick = () => {
      selectModel(m.id);
      $("#modelsModal").hidden = true;
    };
    list.appendChild(card);
  }
}

function selectModel(id) {
  state.model = id;
  const m = modelById(id);
  $("#badgeModel").textContent = m.label;
  $("#badgeInfo").textContent = `${(m.params / 1e6).toFixed(1)}M parameter`;
  const d = DEFAULTS[id] || DEFAULTS.heavy;
  $("#temp").value = d.temp; $("#tempVal").textContent = d.temp;
  $("#topk").value = d.topk; $("#topkVal").textContent = d.topk;
  $("#cfg").value = d.cfg; $("#cfgVal").textContent = d.cfg;
  renderModelNow();
  renderModelsModal();
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
      // single-select: selalu terpilih — klik chip yg sama gak boleh me-null (wajib ada pilihan)
      state[group] = item.id;
      el.querySelectorAll(".chip").forEach((c) => c.classList.remove("on"));
      b.classList.add("on");
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
  // auto-lanjut ke step berikutnya saat single-select dipilih
  ["#chipsScene", "#chipsColor", "#chipsMood"].forEach((sel) => {
    $(sel).addEventListener("click", () => wizAdvance());
  });
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
  $("#btnBack").onclick = () => showStep(curStep - 1);
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
  // ----- modals: more models / pengaturan / tentang -----
  $("#btnMoreModels").onclick = () => $("#modelsModal").hidden = false;
  $("#btnCloseModels").onclick = () => $("#modelsModal").hidden = true;
  $("#modelsModal").onclick = (e) => { if (e.target === $("#modelsModal")) $("#modelsModal").hidden = true; };

  const openSettings = () => { $("#settingsModal").hidden = false; };
  $("#btnSettings").onclick = openSettings;
  $("#btnSettingsTop").onclick = openSettings;
  $("#btnCloseSettings").onclick = () => $("#settingsModal").hidden = true;
  $("#settingsModal").onclick = (e) => { if (e.target === $("#settingsModal")) $("#settingsModal").hidden = true; };

  const openAbout = () => { $("#settingsModal").hidden = true; $("#aboutModal").hidden = false; };
  $("#btnAboutSettings").onclick = openAbout;
  $("#btnCloseAbout").onclick = () => $("#aboutModal").hidden = true;
  $("#aboutModal").onclick = (e) => { if (e.target === $("#aboutModal")) $("#aboutModal").hidden = true; };

  // bersihkan galeri — konfirmasi 2 langkah (klik lagi dlm 3 dtk)
  let clearArm = null;
  $("#btnClearSettings").onclick = () => {
    const b = $("#btnClearSettings");
    if (clearArm) {
      clearTimeout(clearArm); clearArm = null;
      localStorage.removeItem(galKey());
      renderGallery();
      b.textContent = "beres";
      setTimeout(() => (b.textContent = "Bersihkan"), 1200);
    } else {
      b.textContent = "Yakin? klik lagi";
      clearArm = setTimeout(() => { clearArm = null; b.textContent = "Bersihkan"; }, 3000);
    }
  };

  // ----- toggle CapCut (default mati, persist di localStorage) -----
  $("#swCapcut").onclick = () =>
    applyCapcut(!document.body.classList.contains("capcut"));

  // ----- bottom nav (mode capcut) -----
  const setCapTab = (t) => {
    $("#capBtnBuat").classList.toggle("on", t === "buat");
    $("#capBtnGaleri").classList.toggle("on", t === "galeri");
    $("#capBtnSetelan").classList.toggle("on", false);
  };
  $("#capBtnBuat").onclick = () => {
    document.body.classList.remove("cc-gal");
    setCapTab("buat");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  $("#capBtnGaleri").onclick = () => {
    const on = document.body.classList.toggle("cc-gal");
    setCapTab(on ? "galeri" : "buat");
  };
  $("#capBtnSetelan").onclick = () => $("#settingsModal").hidden = false;
  $("#sheetHandle").onclick = () => {
    document.body.classList.remove("cc-gal");
    setCapTab("buat");
  };
}

function applyCapcut(on) {
  document.body.classList.toggle("capcut", on);
  if (!on) document.body.classList.remove("cc-gal");
  $("#swCapcut").classList.toggle("on", on);
  $("#swCapcut").setAttribute("aria-pressed", on ? "true" : "false");
  try { localStorage.setItem("pixanva_capcut", on ? "1" : "0"); } catch {}
}

function startGenWith(cond, opts) {
  if (state.generating) return;
  const rg = resolveGrid(opts.model, cond.outPx || 64);
  cond.G = rg.G;
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(rg.G, rg.out, rg.smooth);
  getWorker().postMessage({ type: "gen", cond, opts });
}

// ---------- wizard ----------
const WIZ_STEPS = 4;
let curStep = 1;
const WIZ_HINT = {
  1: "milih subjek pemandangan",
  2: "milih warna dominan",
  3: "milih suasana / waktu",
  4: "hiasan tambahan — bisa dilewati",
};
function showStep(n) {
  curStep = Math.max(1, Math.min(WIZ_STEPS, n));
  document.querySelectorAll(".step").forEach((s) =>
    s.classList.toggle("active", +s.dataset.step === curStep));
  $("#btnBack").disabled = curStep <= 1;
  $("#wizHint").textContent = WIZ_HINT[curStep];
  const dots = $("#wizDots");
  if (dots.children.length !== WIZ_STEPS) {
    dots.innerHTML = "";
    for (let i = 0; i < WIZ_STEPS; i++) {
      const d = document.createElement("span");
      d.className = "wdot";
      dots.appendChild(d);
    }
  }
  [...dots.children].forEach((d, i) => {
    d.className = "wdot" + (i + 1 === curStep ? " cur" : i + 1 < curStep ? " done" : "");
  });
}
function wizAdvance() {
  if (curStep < WIZ_STEPS) setTimeout(() => showStep(curStep + 1), 230);
}

// ---------- init ----------
(async function init() {
  initChips();
  initEvents();
  renderGallery();
  // mode capcut: default MATI kecuali user udah nyala-in sebelumnya
  let cc = false;
  try { cc = localStorage.getItem("pixanva_capcut") === "1"; } catch {}
  applyCapcut(cc);
  try {
    await initModels();
    getWorker().postMessage({ type: "load", model: state.model });
  } catch (e) {
    setStatus("gagal memuat meta: " + e.message);
  }
  showStep(1);
})();
