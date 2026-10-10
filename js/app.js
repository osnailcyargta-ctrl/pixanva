// Pixanva app — UI logic
import { PALETTE, TAGS, IDS } from "./data.js";

const $ = (s) => document.querySelector(s);
const DEFAULTS = {
  light: { temp: 0.95, topk: 32, cfg: 1.3 },
  light13: { temp: 0.93, topk: 28, cfg: 1.5 },
  dark: { temp: 0.90, topk: 24, cfg: 1.6 },
  dark12: { temp: 0.90, topk: 24, cfg: 1.7 },
  dark15: { temp: 0.88, topk: 22, cfg: 1.8 },
  heavy: { temp: 0.85, topk: 24, cfg: 1.6 },
  heavyqw: { temp: 0.85, topk: 24, cfg: 1.7 },
  heavyqr: { temp: 0.84, topk: 24, cfg: 1.8 },
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
  lastRun: null,        // {cond, opts, out} utk "ulangi seed ini"
  generating: false,
  worker: null,
  loaded: new Set(),
  varMode: false,       // variasi ×4 aktif?
  varQ: null,           // antrean variasi yg lagi jalan
  varRes: null,         // hasil 4 variasi terakhir (buat klik-pilih)
  painters: null,       // painter per sel variasi
  selVar: -1,
  genfx: true,          // overlay loading blur (persist, default nyala)
  aiLLM: false,         // LLM chat lokal siap? (fallback: rule-based)
  loadTries: {},        // percobaan load gagal per model (buat retry otomatis)
  pmMode: false,        // mode custom prompt aktif? (teks bebas → imajin langsung)
  imjReady: false,      // model imajin ke-load?
  imjLoading: false,    // imajin lagi ke-load?
  imjTries: 0,          // retry load imajin
  pendingCustom: null,  // prompt yang nunggu imajin ke-load
  pmText: "",           // teks custom terakhir
};

// ---------- worker ----------
function getWorker() {
  if (!state.worker) {
    // ?v= rilis — bust cache CDN/browser tiap deploy (Pages cache 10 menit)
    state.worker = new Worker("js/worker.js?v=v27", { type: "module" });
    state.worker.onmessage = onWorkerMsg;
    state.worker.onerror = (e) => setStatus("error worker: " + e.message);
  }
  return state.worker;
}

function hex(c) { return PALETTE[c] || PALETTE[0]; }

function onWorkerMsg(e) {
  const m = e.data;
  if (m.type === "progress") {
    $("#loadBar").hidden = false;
    $("#loadBar > div").style.width = (m.p * 100).toFixed(1) + "%";
    setStatus(`memuat bobot ${(m.got / 1e6).toFixed(1)} / ${(m.total / 1e6).toFixed(1)} MB…`);
  } else if (m.type === "loaded") {
    state.loaded.add(m.model);
    delete state.loadTries[m.model];
    $("#loadBar").hidden = true;
    $("#loadErr").hidden = true;
    setStatus("siap");
    $("#btnGen").disabled = false;
    $("#btnGen").textContent = "Generate";
  } else if (m.type === "token") {
    onToken(m);
  } else if (m.type === "beat") {
    // keepalive opsional
  } else if (m.type === "done") {
    if (state.varQ) finishVarCell(m);
    else finishGen(m);
  } else if (m.type === "error") {
    console.error(m.msg);
    if (state.generating) {
      // generate-nya gagal — pulihkan UI + pesan manusiawi (mobile gak punya console)
      state.generating = false;
      state.varQ = null;
      AIState.busy = false;
      hideOverlay();
      $("#btnGen").disabled = false;
      $("#btnGen").textContent = "Generate";
      setStatus("generate gagal — coba lagi");
    } else {
      loadFailed(m.msg);
    }
  } else if (m.type === "chatready") {
    state.aiLLM = !!m.ok;
    if (m.ok) {
      $("#aiStatus").textContent = `LLM lokal siap (${(m.params / 1e6).toFixed(1)}M) ✦`;
    }
  } else if (m.type === "ctok") {
    aiStream(m.w);
  } else if (m.type === "cdone") {
    aiFinish(m.text);
  } else if (m.type === "imjready") {
    state.imjLoading = false;
    if (m.ok) {
      state.imjReady = true;
      state.imjTries = 0;
      if (state.pmMode && !state.generating) setStatus(`imajin siap (${(m.params / 1e6).toFixed(1)}M) ✍️`);
      if (state.pendingCustom) {
        const t = state.pendingCustom;
        state.pendingCustom = null;
        if (state.pmMode && !state.generating) { $("#pmInput").value = t; startGenCustom(); }
      }
    } else {
      // load imajin gagal — retry otomatis 3x (mobile gak punya console)
      state.imjTries++;
      if (state.imjTries <= 3) {
        setTimeout(() => {
          state.imjLoading = true;
          getWorker().postMessage({ type: "imjload" });
        }, 900 * state.imjTries);
      } else if (state.pmMode && !state.generating) {
        setStatus("model custom gagal ke-load — coba refresh halaman");
      }
    }
  }
}

// ---------- load model anti-gagal: retry otomatis + kartu error ramah mobile ----------
// mobile gak punya console — semua kegagalan load ditangani di UI, gak ada lagi "cek log"
function loadModel() {
  setStatus("memuat model…");
  getWorker().postMessage({ type: "load", model: state.model });
}
function shortErr(msg) {
  const s = String(msg || "");
  if (/timeout|abort/i.test(s)) return "koneksi lambat";
  if (/fetch|network|http|load/i.test(s)) return "jaringan";
  return "error tak terduga";
}
function loadFailed(msg) {
  const tries = (state.loadTries[state.model] || 0) + 1;
  state.loadTries[state.model] = tries;
  if (tries <= 3) {
    setStatus(`model gagal ke-load — nge-retry otomatis (${tries}/3)…`);
    setTimeout(loadModel, 900 * tries);
    return;
  }
  $("#loadErrMsg").textContent =
    `Model gagal ke-load 3x (${shortErr(msg)}). Biasanya cuma jaringan lagi lemot — ` +
    "ketuk Coba lagi, atau pilih model yang lebih ringan di More models.";
  $("#loadErr").hidden = false;
  setStatus("model gagal ke-load");
}
async function bootModels() {
  $("#loadErr").hidden = true;
  try {
    await initModels();
    loadModel();
    // LLM asisten: load di belakang (kalau bin-nya udah ada)
    getWorker().postMessage({ type: "chatload" });
    // imajin custom-prompt: load di belakang juga (kalau bin-nya udah ada)
    getWorker().postMessage({ type: "imjload" });
  } catch (e) {
    console.error(e);
    $("#loadErrMsg").textContent =
      "Gagal memuat daftar model (" + shortErr(e.message) + "). Ketuk Coba lagi buat nge-retry.";
    $("#loadErr").hidden = false;
    setStatus("gagal memuat daftar model");
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
  // balikin tampilan ke mode tunggal + bersihin sisa mode variasi
  $("#varGrid").hidden = true;
  $("#varGrid").innerHTML = "";
  $("#canvas").style.display = "";
  state.painters = null;
  clearRevealNow();
  brushHide();
  hideOverlay();
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

// routing token: ke kanvas utama / ke sel variasi yg lagi dilukis
let revCtxs = [];
function onToken(m) {
  const q = state.varQ;
  if (q && state.painters) {
    const p = state.painters[Math.min(q.cur, 3)];
    paintCellP(p, m.i, m.c);
    if (state.genfx) {
      // blur di seluruh grid (1 elemen) + gambar tetap di-paint live di baliknya;
      // tiap baris kelar → langsung ke-reveal sinematik tanpa nunggu selesai
      $("#genProgFill").style.width =
        (((q.cur + (m.i + 1) / m.total) / 4) * 100).toFixed(1) + "%";
      requestBlit();
      if ((m.i + 1) % p.G === 0) {
        const y = Math.floor(m.i / p.G);
        $("#paintInfo").textContent = `melukis variasi ${q.cur + 1}/4 … baris ${y + 1}/${p.G}`;
        enqueueRow({ grid: p.gridCv, ctx: revCtxs[Math.min(q.cur, 3)], G: p.G, out: p.out, y, cell: q.cur, dx: 0 });
      }
    } else if (m.i % 6 === 0 || m.i === m.total - 1) {
      blitPainter(p);
      brushFollow(q.cur, m.i, p.G);
      $("#paintInfo").textContent = `melukis variasi ${q.cur + 1}/4 … ${m.i + 1}/${m.total} sel`;
      setStatus("melukis");
    }
    return;
  }
  paintCell(m.i, m.c, currentG);
  if (state.genfx) {
    $("#genProgFill").style.width = ((m.i + 1) / m.total * 100).toFixed(1) + "%";
    requestBlit();
    if ((m.i + 1) % currentG === 0) {
      const y = Math.floor(m.i / currentG);
      const dt = (performance.now() - genStart) / 1000;
      $("#paintInfo").textContent = `melukis… baris ${y + 1}/${currentG} • ${dt.toFixed(1)}s`;
      setStatus("melukis");
      enqueueRow({ grid: gridCv, ctx: $("#revealCv").getContext("2d"), G: currentG, out: outPx, y, cell: -1, dx: 0 });
    }
  } else if (m.i % 6 === 0 || m.i === m.total - 1) {
    blitGrid(curSmooth);
    brushFollow(-1, m.i, currentG);
    const dt = (performance.now() - genStart) / 1000;
    $("#paintInfo").textContent = `melukis… ${m.i + 1}/${m.total} sel • ${dt.toFixed(1)}s`;
    setStatus("melukis");
  }
}

// painter terpisah per sel variasi (canvas kecil sendiri)
function makePainter(cv, G, out) {
  const gc = document.createElement("canvas");
  gc.width = G * 2; gc.height = G * 2;
  const gctx = gc.getContext("2d");
  gctx.fillStyle = "#0b0d11"; gctx.fillRect(0, 0, G * 2, G * 2);
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#0b0d11"; ctx.fillRect(0, 0, out, out);
  return { cv, ctx, gridCv: gc, gridCtx: gctx, G, out };
}
function paintCellP(p, i, c) {
  const r = Math.floor(i / p.G), col = i % p.G;
  p.gridCtx.fillStyle = hex(c);
  p.gridCtx.fillRect(col * 2, r * 2, 2, 2);
}
function blitPainter(p) {
  p.ctx.imageSmoothingEnabled = false;
  p.ctx.drawImage(p.gridCv, 0, 0, p.out, p.out);
}

// ---------- kuas virtual: nyusulin posisi lukis ----------
function brushShow() { $("#brush").hidden = false; }
function brushHide() { $("#brush").hidden = true; }
function brushTo(fx, fy, hard) {
  const b = $("#brush");
  b.classList.toggle("hard", !!hard);
  b.style.left = (fx * 100).toFixed(2) + "%";
  b.style.top = (fy * 100).toFixed(2) + "%";
}
// cellIdx: 0..3 = posisi grid 2×2 mode variasi; -1 = kanvas tunggal
function brushFollow(cellIdx, i, G) {
  const col = i % G, row = Math.floor(i / G);
  let x, y;
  if (cellIdx >= 0) {
    x = (cellIdx % 2) * 50 + ((col + 1) / G) * 50;
    y = Math.floor(cellIdx / 2) * 50 + ((row + 1) / G) * 50;
  } else {
    x = ((col + 1) / G) * 100;
    y = ((row + 1) / G) * 100;
  }
  brushTo(x / 100, y / 100);
}

// --- palet RGB (dipakai layer reveal sinematik) ---
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

// ---------- reveal sinematik per layer: kasar → detail, kuas nyusul sweep ----------
// balik lagi sesuai request — pas gambar selese, gambarnya ke-draw ulang layer demi layer
function layerCanvas(tokens, G, block, out, soft) {
  const g = Math.max(1, Math.ceil(G / block));
  const small = document.createElement("canvas");
  small.width = g; small.height = g;
  const sc = small.getContext("2d");
  const img = sc.createImageData(g, g);
  const pal = palRGB();
  for (let by = 0; by < g; by++) for (let bx = 0; bx < g; bx++) {
    let r = 0, gg = 0, bb = 0, n = 0;
    for (let y = by * block; y < Math.min(G, (by + 1) * block); y++)
      for (let x = bx * block; x < Math.min(G, (bx + 1) * block); x++) {
        // token di luar range warna (model kadang nge-sample token tag) → fallback hitam
        const c = pal[tokens[y * G + x]] || pal[0];
        r += c[0]; gg += c[1]; bb += c[2]; n++;
      }
    const pi = nearestPal(r / n, gg / n, bb / n);
    const o = (by * g + bx) * 4;
    img.data[o] = pal[pi][0]; img.data[o + 1] = pal[pi][1];
    img.data[o + 2] = pal[pi][2]; img.data[o + 3] = 255;
  }
  sc.putImageData(img, 0, 0);
  const big = document.createElement("canvas");
  big.width = out; big.height = out;
  const bc = big.getContext("2d");
  // 128 (grid 48 → non-integer): smooth biar gak muncul pixel aneh tak rata
  bc.imageSmoothingEnabled = !!soft;
  bc.drawImage(small, 0, 0, out, out);
  return big;
}

function cinematicReveal(p, tokens, onDone, delayMs = 0, region = { x: 0, y: 0, w: 1, h: 1 }) {
  const layers = [8, 4, 2, 1];
  const cvs = layers.map((b) => layerCanvas(tokens, p.G, b, p.out, p.smooth));
  let li = 0;
  const start = () => {
    // tab gak kelihatan → timer di-throttle browser: skip sweep, langsung final
    if (document.hidden) {
      p.ctx.imageSmoothingEnabled = !!p.smooth;
      p.ctx.drawImage(cvs[cvs.length - 1], 0, 0);
      if (onDone) onDone();
      return;
    }
    brushShow();
    const stepLayer = () => {
      if (li >= layers.length) { brushHide(); if (onDone) onDone(); return; }
      const lc = cvs[li];
      const dur = 400 + li * 130; // lebih pelan biar kelihatan layer demi layer
      const t0 = performance.now();
      // setTimeout loop (bukan rAF) — tetap jalan walau tab lagi gak fokus
      const anim = () => {
        const f = Math.min(1, (performance.now() - t0) / dur);
        const x = Math.max(1, Math.ceil(f * p.out));
        p.ctx.drawImage(lc, 0, 0, x, p.out, 0, 0, x, p.out);
        brushTo(
          region.x + f * region.w,
          region.y + region.h * (0.5 + 0.32 * Math.sin(f * Math.PI * 2)),
          true
        );
        if (f < 1) setTimeout(anim, 16);
        else { li++; stepLayer(); }
      };
      anim();
    };
    stepLayer();
  };
  if (delayMs > 0) setTimeout(start, delayMs);
  else start();
}

// ---------- overlay loading (blur SEKALI di seluruh gambar + spinner + progress) ----------
function showOverlay() {
  if (!state.genfx) return;
  $("#genOverlay").hidden = false;
  $("#genProgFill").style.width = "0%";
  $("#canvasBox").classList.add("generating");
  if (!state.varQ) {
    const rv = $("#revealCv");
    rv.hidden = false;
    rv.width = outPx; rv.height = outPx;
    rv.getContext("2d").clearRect(0, 0, outPx, outPx);
    rv.style.opacity = "1";
  }
  brushShow();
}
function hideOverlay() {
  $("#genOverlay").hidden = true;
  $("#canvasBox").classList.remove("generating");
}

// ---------- reveal per baris: tiap baris kelar → langsung ke-draw sinematik ----------
// animasinya jalan SEWAKTU generate — gak perlu nunggu 2 dtk di akhir
const revQ = [];
let revBusy = false;
let rafPending = false;

function liveBlit() {
  rafPending = false;
  if (!state.generating || !state.genfx) return;
  const q = state.varQ;
  if (q && state.painters) blitPainter(state.painters[Math.min(q.cur, 3)]);
  else blitGrid(curSmooth);
}
function requestBlit() {
  if (!state.genfx || rafPending) return;
  rafPending = true;
  requestAnimationFrame(liveBlit);
}

function enqueueRow(job) {
  revQ.push(job);
  pumpReveal();
}
function pumpReveal() {
  if (revBusy) return;
  const job = revQ.shift();
  if (!job) return;
  revBusy = true;
  // tab gak kelihatan / antrian numpuk → gambar langsung tanpa animasi biar gak numpuk
  if (document.hidden || revQ.length >= 3) {
    drawRow(job, 1.02);
    revBusy = false;
    pumpReveal();
    return;
  }
  const t0 = performance.now();
  const dur = 150;
  const step = () => {
    const f = Math.min(1, (performance.now() - t0) / dur);
    drawRow(job, f);
    const bp = brushPosRow(job, f);
    brushTo(bp.x, bp.y, true);
    if (f < 1) setTimeout(step, 16);
    else { drawRow(job, 1.03); revBusy = false; pumpReveal(); }
  };
  step();
}
// f boleh > 1 — fudge tipis biar tepi antar baris gak bolong
function drawRow(j, f) {
  const fc = Math.min(1, f);
  const h = j.out / j.G;
  j.ctx.drawImage(
    j.grid, 0, j.y * 2, Math.max(0.02, j.G * fc), 2,
    j.dx || 0, j.y * h, Math.max(0.02, j.out * fc), h
  );
}
function brushPosRow(j, f) {
  const fy = (j.y + 0.5) / j.G;
  if (j.cell >= 0) {
    return { x: (j.cell % 2) * 0.5 + f * 0.5, y: Math.floor(j.cell / 2) * 0.5 + fy * 0.5 };
  }
  return { x: f, y: fy };
}
function clearRevealNow() {
  revQ.length = 0;
  revBusy = false;
  const rv = $("#revealCv");
  rv.hidden = true;
  rv.getContext("2d").clearRect(0, 0, rv.width, rv.height);
  rv.style.opacity = "1";
  const vr = $("#varReveal");
  vr.hidden = true;
  vr.querySelectorAll("canvas").forEach((c) => c.getContext("2d").clearRect(0, 0, c.width, c.height));
}

// ---------- generate ----------
let genStart = 0;
function startGen() {
  if (state.pmMode) { startGenCustom(); return; }
  if (!state.scene || !state.color) {
    setStatus("pilih pemandangan & warna dulu");
    return;
  }
  proceedStartGen();
}

// mode custom: teks bebas → model imajin → LANGSUNG dilukis (gak lewat tag).
// Gak ada parser, gak ada chip — modelnya yang ngerti teks lo beneran.
function startGenCustom() {
  const txt = ($("#pmInput").value || "").trim();
  if (!txt) { setStatus("ketik dulu mau gambar apa ✍️"); return; }
  if (state.generating) return;
  if (!state.imjReady) {
    // model imajin belum ke-load — jangan diem, kasih status + auto-jalan pas siap
    state.pendingCustom = txt;
    if (!state.imjLoading) {
      state.imjLoading = true;
      getWorker().postMessage({ type: "imjload" });
    }
    setStatus(`model imajin lagi ke-load…${state.imjTries ? ` (coba ${state.imjTries}/3)` : ""} — bentar lagi langsung jalan`);
    return;
  }
  state.pmText = txt;
  if (!state.seedLock || !state.seed) state.seed = (Math.random() * 0xffffffff) >>> 0;
  $("#seed").value = state.seed;
  const rg = resolveGrid("imajin", parseInt($("#res").value, 10));
  const opts = {
    temp: parseFloat($("#temp").value),
    topk: parseInt($("#topk").value, 10),
    guidance: parseFloat($("#cfg").value),
  };
  state.lastRun = { cond: { text: txt, seed: state.seed }, opts, out: rg.out, custom: true };
  if (state.varMode) { startVarRun(rg, opts, txt); return; }
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(rg.G, rg.out, rg.smooth);
  showOverlay();
  setStatus(`imajin: "${txt}"`);
  getWorker().postMessage({ type: "imj", text: txt, G: rg.G, seed: state.seed, opts });
}

function proceedStartGen() {
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
  state.lastRun = { cond, opts, out: rg.out };
  if (state.varMode) { startVarRun(rg, opts); return; }
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(rg.G, rg.out, rg.smooth);
  showOverlay();
  getWorker().postMessage({ type: "gen", cond, opts });
}

function finishGen(m) {
  state.varQ = null;
  state.generating = false;
  $("#btnGen").disabled = false;
  $("#btnGen").textContent = "Generate";
  $("#btnDownload").disabled = false;
  $("#btnAgain").disabled = false;
  hideOverlay();
  const dt = (m.ms / 1000).toFixed(1);
  const done = () => {
    $("#paintInfo").textContent = `${outPx}×${outPx}px • ${m.G * m.G} sel • ${dt}s • seed ${state.seed}`;
    setStatus("selesai");
    saveOneCanvas($("#canvas"), state.lastRun.cond, state.seed, m.G, outPx);
  };
  // reveal sinematik SELALU main pas gambar selese (animasi hasil — gak ikut toggle blur loading)
  clearRevealNow();
  const ctx = $("#canvas").getContext("2d");
  ctx.fillStyle = "#0b0d11";
  ctx.fillRect(0, 0, outPx, outPx);
  cinematicReveal({ G: m.G, out: outPx, smooth: curSmooth, ctx }, m.tokens, done);
}

// ---------- galeri ----------
function galKey() { return "pixanva_gallery_v1"; }
function loadGallery() {
  try { return JSON.parse(localStorage.getItem(galKey()) || "[]"); }
  catch { return []; }
}
function saveOneCanvas(cv, cond, seed, G, out) {
  const items = loadGallery();
  const up = document.createElement("canvas");
  const scale = Math.max(2, 256 / cv.width);
  up.width = cv.width * scale; up.height = cv.height * scale;
  const uc = up.getContext("2d");
  uc.imageSmoothingEnabled = false;
  uc.drawImage(cv, 0, 0, up.width, up.height);
  items.unshift({
    img: up.toDataURL("image/png"),
    prompt: promptText(cond),
    model: cond.text ? "imajin" : state.model,
    seed,
    G,
    out,
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
    ctx.imageSmoothingEnabled = (out || G * 2) > G * 2; // 128 dari 48: halus, sisanya crisp
    ctx.drawImage(im, 0, 0, G * 2 >= out ? out : out, out);
  };
  im.src = src;
}

function promptText(cond) {
  if (cond.text) return cond.text;   // custom prompt — tampil persis yang user ketik
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
  const res = await fetch("models/meta.json?v=v27");
  state.meta = await res.json();
  const mains = state.meta.models.filter((m) => m.main).sort((a, b) => (b.ri || 0) - (a.ri || 0));
  selectModel(mains.length ? mains[0].id : "heavyqw"); // default: model utama terbaru
  showStep(1);
}

// 3 kartu model utama (versi terbaru tiap keluarga) — nanggung di sidebar
function renderMainModels() {
  const box = $("#modelNow");
  const mains = state.meta.models.filter((m) => m.main && !m.hidden);
  box.innerHTML = '<div class="model-list main3">' + mains.map((m) => `
    <div class="model-card ${m.id === state.model ? "active" : ""}" data-id="${m.id}">
      <div class="mc-name"><span>${m.label}</span>${m.isNew ? ' <span class="tag-new">BARU</span>' : ""}</div>
      <div class="mc-desc">${m.desc}</div>
      <div class="mc-meta">${fmtMeta(m)}${m.released ? ` • rilis ${m.released}` : ""}</div>
    </div>`).join("") + "</div>";
  box.querySelectorAll(".model-card").forEach((c) => (c.onclick = () => selectModel(c.dataset.id)));
}

// sort popup More models — terbaru / terpintar + reverse (persist di localStorage)
const mmSort = { key: "baru", rev: false };
function loadMmSort() {
  try {
    const s = JSON.parse(localStorage.getItem("pixanva_mmsort") || "null");
    if (s) { mmSort.key = s.key === "pintar" ? "pintar" : "baru"; mmSort.rev = !!s.rev; }
  } catch {}
}
function saveMmSort() {
  try { localStorage.setItem("pixanva_mmsort", JSON.stringify(mmSort)); } catch {}
}
function sortOthers() {
  const arr = [...state.meta.models.filter((m) => !m.main && !m.hidden)];
  if (mmSort.key === "baru") arr.sort((a, b) => (b.ri || 0) - (a.ri || 0));
  else arr.sort((a, b) => (b.params - a.params) || ((b.step || 0) - (a.step || 0)));
  if (mmSort.rev) arr.reverse();
  return arr;
}

// isi popup More models — cuma model selain 3 utama, sesuai sort
function renderModelsModal() {
  const list = $("#mmList");
  const sorted = sortOthers();
  $("#mmSub").textContent = sorted.length
    ? `${sorted.length} model lama & versi sebelumnya — tetap bisa dipilih`
    : "semua model udah nanggung di layar utama";
  $("#sortNew").classList.toggle("on", mmSort.key === "baru");
  $("#sortSmart").classList.toggle("on", mmSort.key === "pintar");
  $("#btnRevSort").classList.toggle("on", mmSort.rev);
  list.innerHTML = "";
  for (const m of sorted) {
    const card = document.createElement("div");
    card.className = "mm-item" + (m.id === state.model ? " active" : "");
    card.innerHTML = `
      <div class="mm-top">
        <span class="mm-name">${m.label}</span>
        ${m.isNew ? '<span class="tag-new">BARU</span>' : ""}
        ${m.id === state.model ? '<span class="tag-cur">aktif</span>' : ""}
      </div>
      <div class="mc-desc">${m.desc}</div>
      <div class="mc-meta">${fmtMeta(m)} • step ${m.step}${m.released ? ` • rilis ${m.released}` : ""}</div>`;
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
  renderMainModels();
  renderModelsModal();
  renderCapModels();
  state.loaded.delete(id); // force reload meta utk model ini kalau berganti
  if (state.worker) {
    getWorker().postMessage({ type: "load", model: id });
  }
}

// strip pilihan model buat tema CapCut — sidebar ke-hidden di tema itu,
// jadi pemilih modelnya pindah ke atas panel "Buat gambar"
function renderCapModels() {
  const el = $("#capModels");
  if (!el || !state.meta) return;
  const all = state.meta.models.filter((m) => !m.hidden)
    .sort((a, b) => (b.ri || 0) - (a.ri || 0) || (b.step || 0) - (a.step || 0));
  el.innerHTML = all.map((m) =>
    `<button class="chip cm-chip ${m.id === state.model ? "on" : ""}" data-id="${m.id}">` +
    `${m.label.replace(/^Pixanva\s+/, "")}${m.isNew ? " ★" : ""}</button>`
  ).join("") + '<button class="chip cm-chip" data-all="1">Semua…</button>';
  el.querySelectorAll(".cm-chip").forEach((c) => (c.onclick = () => {
    if (c.dataset.all) { $("#modelsModal").hidden = false; return; }
    selectModel(c.dataset.id);
  }));
}

// ---------- chips ----------
function chip(group, el, item, multi, max) {
  const b = document.createElement("button");
  b.className = "chip";
  b.dataset.id = item.id;
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
  // skip step 2 & 3 — gak usah lewat satu-satu: warna diacak / suasana dibuang
  const R = (a) => a[Math.floor(Math.random() * a.length)];
  $("#btnSkipColor").onclick = () => {
    state.color = R(TAGS.colors).id;
    setChipsOn("#chipsColor", state.color);
    showStep(3);
  };
  $("#btnSkipMood").onclick = () => {
    state.mood = null;
    setChipsOn("#chipsMood", null);
    showStep(4);
  };
  $("#btnSkipAll").onclick = () => {
    if (state.generating) return;
    if (!state.scene) state.scene = R(TAGS.scenes).id;
    if (!state.color) state.color = R(TAGS.colors).id;
    state.mood = null;
    state.orn = [];
    setChipsOn("#chipsColor", state.color);
    setChipsOn("#chipsMood", null);
    setChipsOn("#chipsOrn", []);
    startGen();
  };
  // retry load model dari kartu error
  $("#btnLoadRetry").onclick = () => {
    state.loadTries = {};
    $("#loadErr").hidden = true;
    if (!state.meta) bootModels();
    else loadModel();
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
  // ----- prompt dadu: acak semua tag + langsung generate -----
  $("#btnDadu").onclick = () => {
    if (state.generating) return;
    const R = (a) => a[Math.floor(Math.random() * a.length)];
    state.scene = R(TAGS.scenes).id;
    state.color = R(TAGS.colors).id;
    state.mood = Math.random() < 0.85 ? R(TAGS.moods).id : null;
    const pool = [...TAGS.ornaments];
    state.orn = [];
    const n = Math.random() < 0.5 ? 0 : Math.random() < 0.72 ? 1 : 2;
    for (let i = 0; i < n; i++)
      state.orn.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0].id);
    setChipsOn("#chipsScene", state.scene);
    setChipsOn("#chipsColor", state.color);
    setChipsOn("#chipsMood", state.mood);
    setChipsOn("#chipsOrn", state.orn);
    showStep(1);
    setStatus("dadu: " + promptText({ scene: state.scene, color: state.color, mood: state.mood, orn: state.orn }));
    startGen();
  };
  // ----- toggle variasi ×4 -----
  $("#btnVar").onclick = () => {
    state.varMode = !state.varMode;
    $("#btnVar").classList.toggle("on", state.varMode);
    setStatus(state.varMode ? "mode variasi ×4 nyala — 1 prompt jadi 4 gambar" : "mode variasi mati");
  };
  $("#temp").oninput = () => $("#tempVal").textContent = $("#temp").value;
  $("#topk").oninput = () => $("#topkVal").textContent = $("#topk").value;
  $("#cfg").oninput = () => $("#cfgVal").textContent = $("#cfg").value;
  // ----- modals: more models / pengaturan / tentang -----
  $("#btnMoreModels").onclick = () => $("#modelsModal").hidden = false;
  $("#btnModelsTop").onclick = () => $("#modelsModal").hidden = false;
  $("#btnCloseModels").onclick = () => $("#modelsModal").hidden = true;
  $("#modelsModal").onclick = (e) => { if (e.target === $("#modelsModal")) $("#modelsModal").hidden = true; };
  // ----- sort popup more models: terbaru / terpintar + reverse -----
  $("#sortNew").onclick = () => { mmSort.key = "baru"; saveMmSort(); renderModelsModal(); };
  $("#sortSmart").onclick = () => { mmSort.key = "pintar"; saveMmSort(); renderModelsModal(); };
  $("#btnRevSort").onclick = () => { mmSort.rev = !mmSort.rev; saveMmSort(); renderModelsModal(); };

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

  // ----- toggle layar loading blur (default nyala, persist; reaktif mid-generate) -----
  $("#swGenfx").onclick = () => applyGenfx(!state.genfx);

  // ----- mode tag / custom -----
  $("#modeTag").onclick = () => applyMode("tag");
  $("#modeCustom").onclick = () => applyMode("custom");
  $("#pmDice").onclick = () => {
    $("#pmInput").value = PM_EXAMPLES[Math.floor(Math.random() * PM_EXAMPLES.length)];
  };
  $("#pmInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startGen();
  });

  // ----- bottom nav (tema capcut) -----
  const setCapTab = (t) => {
    $("#capBtnBuat").classList.toggle("on", t === "buat");
    $("#capBtnGaleri").classList.toggle("on", t === "galeri");
    $("#capBtnSetelan").classList.toggle("on", false);
  };
  $("#capBtnBuat").onclick = () => {
    setCapTab("buat");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  $("#capBtnGaleri").onclick = () => {
    setCapTab("galeri");
    const el = document.querySelector(".gallery-sec");
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  $("#capBtnSetelan").onclick = () => $("#settingsModal").hidden = false;
}

// ---------- tema tampilan (8 tema, persist, migrasi dari toggle capcut lama) ----------
const THEMES = [
  { id: "classic-dark", name: "Classic (Dark)", dots: ["#0e1014", "#6c7bff", "#39d98a"] },
  { id: "classic-light", name: "Classic (Light)", dots: ["#eef1f7", "#5661f2", "#ffb648"] },
  { id: "metal", name: "Classic (Metalic)", dots: ["#26292f", "#aeb9c9", "#7f8b9c"] },
  { id: "paper", name: "Paper Doodle", dots: ["#f6efdd", "#33291d", "#e8590c"] },
  { id: "capcut", name: "CapCut Typshit", dots: ["#000000", "#22e0c4", "#3d7eff"] },
  { id: "neon", name: "Pixanva Neon", dots: ["#07070f", "#19e3ff", "#ff0dca"] },
  { id: "sunset", name: "Senja Jam", dots: ["#1f1014", "#ff8a5c", "#ff5c7a"] },
  { id: "hacker", name: "Hacker Terminal", dots: ["#040704", "#26ff7d", "#0fca55"] },
];
function applyTheme(id, save = true) {
  const t = THEMES.find((x) => x.id === id) || THEMES[0];
  document.body.dataset.theme = t.id;
  if (t.id !== "capcut") document.body.classList.remove("cc-gal");
  if (save) { try { localStorage.setItem("pixanva_theme", t.id); } catch {} }
  renderThemePicker();
  setStatus(`tema: ${t.name}`);
}
function renderThemePicker() {
  const g = $("#themeGrid");
  if (!g) return;
  g.innerHTML = "";
  for (const t of THEMES) {
    const b = document.createElement("button");
    b.className = "theme-card" + (document.body.dataset.theme === t.id ? " on" : "");
    b.innerHTML = `<span class="tc-dots">${t.dots.map((d) => `<i style="background:${d}"></i>`).join("")}</span><span>${t.name}</span>`;
    b.onclick = () => applyTheme(t.id);
    g.appendChild(b);
  }
}

// ---------- layar loading toggle (reaktif) ----------
function applyGenfx(on) {
  state.genfx = !!on;
  $("#swGenfx").classList.toggle("on", state.genfx);
  $("#swGenfx").setAttribute("aria-pressed", state.genfx ? "true" : "false");
  // REAKTIF walau lagi generate: toggle mid-run langsung ngefek
  if (!state.genfx) {
    hideOverlay();
    if (state.generating) clearRevealNow();
  } else if (state.generating) {
    showOverlay();
  }
  try { localStorage.setItem("pixanva_genfx", state.genfx ? "1" : "0"); } catch {}
}

// sync tampilan chip wizard dari state (dipakai dadu)
function setChipsOn(sel, ids) {
  const want = Array.isArray(ids) ? ids : ids == null ? [] : [ids];
  $(sel).querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("on", want.includes(c.dataset.id)));
}

function startGenWith(cond, opts) {
  if (state.generating) return;
  if (cond.text) {
    // custom prompt (imajin) — ulang run lama (tombol lagi)
    const out = (state.lastRun && state.lastRun.out) || 64;
    const rg = resolveGrid("imajin", out);
    cond.G = rg.G;
    state.varQ = null;
    state.generating = true;
    $("#btnGen").disabled = true;
    $("#btnGen").textContent = "Melukis…";
    genStart = performance.now();
    setupCanvas(rg.G, rg.out, rg.smooth);
    showOverlay();
    getWorker().postMessage({ type: "imj", text: cond.text, G: rg.G, seed: cond.seed, opts });
    return;
  }
  const out = (state.lastRun && state.lastRun.out) || cond.outPx || 64;
  const rg = resolveGrid(opts.model, out);
  cond.G = rg.G;
  state.varQ = null;
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis…";
  genStart = performance.now();
  setupCanvas(rg.G, rg.out, rg.smooth);
  showOverlay();
  getWorker().postMessage({ type: "gen", cond, opts });
}

// ---------- variasi ×4: satu prompt, 4 seed, grid 2×2 ----------
function startVarRun(rg, opts, customText) {
  const seeds = [0, 1, 2, 3].map((i) =>
    state.seedLock && state.seed
      ? (state.seed + i * 101) >>> 0
      : (Math.random() * 0xffffffff) >>> 0);
  state.seed = seeds[0];
  $("#seed").value = seeds[0];
  const conds = seeds.map((s) => customText
    ? { text: customText, seed: s }
    : {
        scene: state.scene, color: state.color, mood: state.mood,
        orn: [...state.orn], G: rg.G, seed: s,
      });
  state.varQ = { conds, opts, results: [], cur: 0, rg, custom: !!customText };
  currentG = rg.G; outPx = rg.out; curSmooth = rg.smooth;
  const vg = $("#varGrid");
  vg.innerHTML = ""; vg.hidden = false;
  $("#canvas").style.display = "none";
  $("#stageEmpty").style.display = "none";
  // lapisan reveal per sel (di atas grid yang di-blur sekali)
  const vr = $("#varReveal");
  vr.innerHTML = "";
  vr.style.opacity = "1";
  revCtxs = conds.map(() => {
    const cv = document.createElement("canvas");
    cv.width = rg.out; cv.height = rg.out;
    vr.appendChild(cv);
    return cv.getContext("2d");
  });
  vr.hidden = false;
  state.painters = conds.map((c, i) => {
    const cell = document.createElement("div");
    cell.className = "var-cell";
    cell.dataset.i = i;
    cell.title = "klik buat jadikan variasi utama";
    const cv = document.createElement("canvas");
    cv.width = rg.out; cv.height = rg.out;
    const tag = document.createElement("span");
    tag.className = "vc-tag";
    tag.textContent = "#" + (i + 1);
    cell.appendChild(cv);
    cell.appendChild(tag);
    vg.appendChild(cell);
    return makePainter(cv, rg.G, rg.out);
  });
  [...vg.children].forEach((c, i) => (c.onclick = () => selectVar(i)));
  state.selVar = -1;
  $("#btnDownload").disabled = true;
  $("#btnAgain").disabled = true;
  state.generating = true;
  $("#btnGen").disabled = true;
  $("#btnGen").textContent = "Melukis ×4…";
  genStart = performance.now();
  showOverlay();
  if (customText) {
    getWorker().postMessage({ type: "imj", text: customText, G: rg.G, seed: conds[0].seed, opts });
  } else {
    getWorker().postMessage({ type: "gen", cond: conds[0], opts });
  }
}

function finishVarCell(m) {
  const q = state.varQ;
  if (!q) return;
  const i = q.cur;
  const p = state.painters[i];
  q.results.push({ tokens: m.tokens.slice(), seed: q.conds[i].seed });
  if (q.rg.smooth && q.rg.G * 2 !== q.rg.out) {
    // 128 dari grid 48: halus bilinear, bukan dither (pixel gak aneh)
    p.ctx.imageSmoothingEnabled = true;
    p.ctx.drawImage(p.gridCv, 0, 0, q.rg.out, q.rg.out);
  } else blitPainter(p);
  q.cur++;
  if (q.cur < 4) {
    const c = q.conds[q.cur];
    if (q.custom) {
      getWorker().postMessage({ type: "imj", text: c.text, G: q.rg.G, seed: c.seed, opts: q.opts });
    } else {
      getWorker().postMessage({ type: "gen", cond: c, opts: q.opts });
    }
    return;
  }
  // semua sel beres — reveal sinematik per layer rame-rame (balik kayak dulu)
  state.generating = false;
  state.varQ = null;
  $("#btnGen").disabled = false;
  $("#btnGen").textContent = "Generate";
  hideOverlay();
  clearRevealNow();
  const dt = ((performance.now() - genStart) / 1000).toFixed(1);
  $("#paintInfo").textContent = `4 variasi • ${dt}s — klik salah satu buat jadi utama`;
  setStatus("4 variasi siap");
  const cond0 = q.custom
    ? { text: q.conds[0].text }
    : { scene: state.scene, color: state.color, mood: state.mood, orn: [...state.orn] };
  q.results.forEach((r, i2) => {
    saveOneCanvas(state.painters[i2].cv, cond0, r.seed, q.rg.G, q.rg.out);
    // reveal per layer rame-rame — SELALU, gak ikut toggle blur loading
    const p2 = state.painters[i2];
    p2.ctx.fillStyle = "#0b0d11";
    p2.ctx.fillRect(0, 0, p2.out, p2.out);
    cinematicReveal(
      { G: q.rg.G, out: q.rg.out, smooth: q.rg.smooth, ctx: p2.ctx }, r.tokens, null,
      350 + i2 * 190,
      { x: (i2 % 2) * 0.5, y: Math.floor(i2 / 2) * 0.5, w: 0.5, h: 0.5 }
    );
  });
  state.varRes = { results: q.results, conds: q.conds, opts: q.opts, rg: q.rg };
}

function selectVar(i) {
  const V = state.varRes;
  if (!V || !V.results[i] || state.generating) return;
  const r = V.results[i];
  const rg = V.rg;
  setupCanvas(rg.G, rg.out, rg.smooth); // balik ke tampilan tunggal
  for (let k = 0; k < rg.G * rg.G; k++) paintCell(k, r.tokens[k], rg.G);
  // 128: halus bilinear; 32/64/96: pixel crisp
  blitGrid(rg.smooth);
  state.seed = r.seed;
  $("#seed").value = r.seed;
  state.lastRun = { cond: { ...V.conds[i] }, opts: V.opts, out: rg.out };
  $("#btnDownload").disabled = false;
  $("#btnAgain").disabled = false;
  $("#paintInfo").textContent = `${rg.out}×${rg.out}px • variasi #${i + 1} • seed ${r.seed}`;
  setStatus(`variasi #${i + 1} jadi utama`);
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

// ---------- asisten prompt: fab moveable + chat ----------
// Otak: LLM lokal ±6.9M (assistant.bin, dilatih dari nol) kalau ke-load;
// kalau belum/kegalan → fallback rule-based di bawah.
const AIState = { last: null, neg: [], busy: false, lastUserText: "", prevUser: "" };

// sinonim → id tag (dari data.js) buat nangkep maksud user dari chat
const SYN = {
  scenes: {
    gunung: ["gunung", "pegunungan", "puncak", "bukit", "mountain", "everest", "rinjani"],
    laut: ["laut", "lautan", "samudra", "ocean", "segara", "laut dalam"],
    hutan: ["hutan", "rimba", "jungle", "forest", "pepohonan", "hutan belantara"],
    kota: ["kota", "urban", "city", "gedung", "downtown", "metropolis", "skyline"],
    gurun: ["gurun", "desert", "sahara", "duna", "pasiran", "gurun pasir"],
    angkasa: ["angkasa", "galaksi", "galaxy", "planet", "nebula", "orbit", "luar angkasa", "antariksa"],
    aurora: ["aurora", "northern lights", "polar", "kutub utara"],
    pantai: ["pantai", "beach", "pesisir", "tebing laut", "tepian"],
    danau: ["danau", "telaga", "lake", "situ"],
    sawah: ["sawah", "ladang", "rice field", "petak sawah", "pertanian", "perkebunan"],
    kanjon: ["kanjon", "canyon", "ngarai", "jurang", "antelope"],
    volkano: ["volkano", "volcano", "gunung api", "vulkan", "lava", "erupsi", "magma", "krakatau"],
    bunga: ["bunga", "flower", "sakura", "ladang bunga", "tulip", "mawar", "sunflower"],
    terjun: ["air terjun", "terjun", "waterfall", "curug", "coban"],
    salju: ["salju", "snow", "tundra", "es kutub", "arctic", "bingkai es", "mountain salju"],
    awan: ["lautan awan", "di atas awan", "cloud sea", "atas awan"],
  },
  colors: {
    hangat: ["hangat", "warm", "oranye", "keemasan"],
    dingin: ["dingin", "cool", "cold", "biru dingin"],
    neon: ["neon", "cyberpunk", "cyber", "futuristik", "glow", "synthwave"],
    pastel: ["pastel", "lembut warna", "soft warna", "candy"],
    monokrom: ["monokrom", "monochrome", "hitam putih", "abu abu", "grayscale", "b&w"],
    bumi: ["bumi", "earth tone", "coklat", "tanah", "natural warna"],
    tropis: ["tropis", "tropical", "jungle vibe", "bali"],
    gelap: ["gelap", "dark", "galap", "remang", "gothic"],
    cerah: ["cerah", "bright", "terang", "vivid", " colorful"],
    vintage: ["vintage", "retro", "jadul", "old school", "film look", "analog"],
    es: ["es", "ice", "beku", "gletser", "glacier", "es biru"],
    smaragd: ["smaragd", "emerald", "zamrud", "hijau zamrud"],
  },
  moods: {
    pagi: ["pagi", "morning", "fajar", "sunrise", "subuh", "early morning"],
    siang: ["siang", "noon", "midday", "terik"],
    senja: ["senja", "sore", "sunset", "golden hour", "maghrib", "dusk"],
    malam: ["malam", "night", "tengah malam", "midnight", "dini hari", "late night"],
    kabut: ["kabut", "fog", "mist", "berkabut", "misty", "foggy"],
    badai: ["badai", "storm", "hujan", "angin kencang", "tornado", "stormy"],
    mystic: ["mystic", "mistis", "magis", "magic", "fantasy", "gaib", "legendaris", "epik", "epic"],
    mimpi: ["mimpi", "dreamy", "dream", "surreal", "halus", "soft mood"],
  },
  orns: {
    bintang: ["bintang", "star", "starry", "berbintang"],
    bulan: ["bulan", "moon", "crescent", "purnama", "bulan sabit"],
    matahari: ["matahari", "sun", "matahari terbenam", "sunset sun"],
    awan: ["awan", "cloud", "mendung", "berawan"],
    burung: ["burung", "bird", "kawanan burung", "flock"],
    perahu: ["perahu", "boat", "kapal", "sampan", "sailboat", "perahu layar"],
    balon: ["balon", "balon udara", "hot air balloon", "zeppelin"],
    kupu: ["kupu", "kupu kupu", "butterfly", "kupu2"],
    petir: ["petir", "lightning", "kilat", "thunder", "halilintar"],
    pelangi: ["pelangi", "rainbow", "pelangi"],
    meteor: ["meteor", "meteor shower", "hujan meteor", "komet", "comet"],
    salju: ["hujan salju", "snowing", "salju turun", "snow fall"],
    pohon: ["pohon", "tree", "pine", "cemara", "pohon cemara"],
  },
};
const LBL = {};
const LBLREV = {};
for (const g of ["scenes", "colors", "moods", "ornaments"])
  for (const t of TAGS[g]) {
    LBL[t.id] = t.label;
    LBLREV[t.label.toLowerCase()] = t.id;
  }

const AI_LINES = {
  ack: ["Oke bro, gw racik dulu 🔎", "Sip, ide bagus — bentar ya 🎨", "Noted! gw susun 3 opsi ✨", "Oke, gw gali ide dulu 🔍"],
  deliver: ["Nih 3 pilihannya 👇", "Nih gw racik 3, tinggal pilih 👇", "3 opsi siap — klik salah satu 👇"],
  reask: ["Oke gw racik ulang 🔄", "Bentar, gw ganti resepnya 🔄", "Oke, versi lain ya 🔄"],
  notopic: [
    "Hmm gw gak nangkep maksudnya 😅 — santai, nih gw racik 3 opsi enak dulu.",
    "Gw belum paham nih 😄 tapi tenang — 3 opsi acak udah gw siapin di bawah.",
  ],
};

function aiNorm(s) {
  return " " + s.toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ").trim() + " ";
}
function synIds(norm, group) {
  const out = [];
  for (const [id, words] of Object.entries(SYN[group])) {
    for (const w of words) {
      if (norm.includes(" " + w + " ")) { out.push(id); break; }
    }
  }
  return out;
}
function aiTopic(norm) {
  const sc = synIds(norm, "scenes");
  const co = synIds(norm, "colors");
  const mo = synIds(norm, "moods");
  const or = synIds(norm, "orns");
  return {
    scene: sc[0] || null, color: co[0] || null, mood: mo[0] || null,
    orn: or.filter((x) => x !== sc[0]).slice(0, 2),
    any: sc.length + co.length + mo.length + or.length > 0,
  };
}
function aiParseNeg(norm) {
  const found = new Set();
  const re = /(?:jangan|jgn|ga usah|gak usah|gausah|tanpa|without|exclude)\s+(?:ada\s+|tag\s+|tags?\s+|pakai\s+|pake\s+)?([a-z0-9\s]+)/g;
  let m;
  while ((m = re.exec(norm))) {
    const chunk = " " + m[1] + " ";
    for (const g of Object.keys(SYN))
      for (const [id, words] of Object.entries(SYN[g]))
        if (words.some((w) => chunk.includes(" " + w + " "))) found.add(id);
  }
  return [...found];
}
function aiFieldNeg() {
  const raw = ($("#aiNegInput").value || "").toLowerCase();
  if (!raw.trim()) return [];
  const out = new Set();
  for (const part of raw.split(/[,;]/)) {
    const p = aiNorm(part);
    if (!p.trim()) continue;
    for (const g of Object.keys(SYN))
      for (const [id, words] of Object.entries(SYN[g]))
        if (words.some((w) => p.includes(" " + w + " ")) || p.includes(" " + (LBL[id] || "").toLowerCase() + " "))
          out.add(id);
  }
  return [...out];
}
const aiR = (a) => a[Math.floor(Math.random() * a.length)];
function aiSets(topic, negs) {
  const neg = new Set(negs);
  const pool = (g) => TAGS[g].filter((t) => !neg.has(t.id));
  const scenes = pool("scenes"), colors = pool("colors"), moods = pool("moods"), orns = pool("ornaments");
  const rec = [
    { nm: "Set 1 — aman & enak", mood: ["senja", "pagi", "siang", "mimpi"], orn: ["awan", "burung", "perahu", "pohon", "matahari"] },
    { nm: "Set 2 — dramatis", mood: ["malam", "badai", "kabut", "mystic"], orn: ["petir", "bulan", "meteor", "bintang"] },
    { nm: "Set 3 — dreamy", mood: ["mimpi", "mystic", "kabut", "pagi"], orn: ["pelangi", "kupu", "balon", "bintang"] },
  ];
  return rec.map((r, i) => {
    const scene = (topic.scene && !neg.has(topic.scene)) ? topic.scene
      : (scenes.length ? aiR(scenes).id : "gunung");
    const color = (topic.color && !neg.has(topic.color)) ? topic.color
      : (colors.length ? aiR(colors).id : "hangat");
    let mood = (i === 0 && topic.mood && !neg.has(topic.mood)) ? topic.mood : null;
    if (!mood) {
      const c = r.mood.filter((x) => !neg.has(x) && moods.some((t) => t.id === x));
      mood = c.length ? aiR(c) : (moods.length ? aiR(moods).id : null);
    }
    const orn = topic.orn.filter((x) => !neg.has(x) && x !== scene && orns.some((t) => t.id === x));
    if (orn.length < 2 && Math.random() < 0.8) {
      const c = r.orn.filter((x) => !neg.has(x) && x !== scene && orns.some((t) => t.id === x) && !orn.includes(x));
      if (c.length) orn.push(aiR(c));
    }
    return { name: r.nm, scene, color, mood, orn: orn.slice(0, 2) };
  });
}
function aiPromptText(set) {
  return [LBL[set.scene], LBL[set.color], set.mood ? LBL[set.mood] : null,
    ...(set.orn || []).map((o) => LBL[o])].filter(Boolean).join(" • ");
}
function aiApplySet(set) {
  if (!set.scene || !set.color) return;
  state.scene = set.scene;
  state.color = set.color;
  state.mood = set.mood;
  state.orn = [...(set.orn || [])];
  setChipsOn("#chipsScene", state.scene);
  setChipsOn("#chipsColor", state.color);
  setChipsOn("#chipsMood", state.mood);
  setChipsOn("#chipsOrn", state.orn);
  showStep(1);
  setStatus("ide asisten: " + aiPromptText(set));
  const g = $("#btnGen");
  g.classList.remove("pulse"); void g.offsetWidth;
  g.classList.add("pulse");
  setTimeout(() => g.classList.remove("pulse"), 3600);
  $("#aiPanel").hidden = true;
}
function aiAddMsg(text, who = "ai") {
  const d = document.createElement("div");
  d.className = "ai-msg" + (who === "me" ? " me" : "");
  d.textContent = text;
  $("#aiMsgs").appendChild(d);
  $("#aiMsgs").scrollTop = 1e9;
}
function aiAddOpts(sets) {
  const wrap = document.createElement("div");
  wrap.className = "ai-opts";
  for (const s of sets) {
    const b = document.createElement("button");
    b.className = "ai-opt";
    b.innerHTML = `<b>${s.name}</b><span></span>`;
    b.querySelector("span").textContent = aiPromptText(s);
    b.onclick = () => aiApplySet(s);
    wrap.appendChild(b);
  }
  $("#aiMsgs").appendChild(wrap);
  $("#aiMsgs").scrollTop = 1e9;
}
function aiReply(text, isReask = false) {
  if (AIState.busy) return;
  const norm = aiNorm(text || "");
  const negs = [...new Set([...aiParseNeg(norm), ...aiFieldNeg()])];
  const topic = aiTopic(norm);
  // "lagi / ganti / versi lain" tanpa topik baru → treat sbg reask
  if (!isReask && AIState.last && !topic.any && /\b(lagi|reask|ganti|versi lain|ulang|other)\b/.test(norm)) {
    return aiReply("", true);
  }
  if (!isReask && text) AIState.lastUserText = text;
  const cur = isReask
    ? ((AIState.lastUserText || "kasih ide prompt dong") + " lagi dong versi lain")
    : (text || "");
  let curFull = cur;
  if (negs.length) curFull += " jangan " + negs.map((n) => (LBL[n] || n).toLowerCase()).join(" ");
  AIState.neg = negs;
  AIState.last = isReask ? AIState.last : topic;

  // ---- jalur LLM lokal (assistant.bin) ----
  if (state.aiLLM && !state.generating) {
    AIState.busy = true;
    aiStreamEl = null;
    aiStreamBuf = "";
    $("#aiStatus").textContent = "LLM mikir…";
    const ctx = AIState.prevUser ? [AIState.prevUser, curFull] : [curFull];
    AIState.prevUser = curFull;
    getWorker().postMessage({
      type: "chat", texts: ctx, temp: 0.82, topk: 24,
      seed: (Math.random() * 0xffffffff) >>> 0,
    });
    return;
  }
  if (state.aiLLM && state.generating) {
    aiAddMsg("Model gambarnya lagi sibuk melukis 😄 nih gw racik manual dulu:");
  }

  // ---- fallback rule-based ----
  AIState.busy = true;
  const T = isReask ? (AIState.last && AIState.last.any ? AIState.last : topic) : topic;
  if (isReask) aiAddMsg(aiR(AI_LINES.reask));
  else aiAddMsg(aiR(AI_LINES.ack));
  if (negs.length) aiAddMsg("Oh iya — tag " + negs.map((n) => LBL[n] || n).join(", ") + " gw ilangin dari semua opsi 🚫");
  if (!T.any && !isReask) aiAddMsg(aiR(AI_LINES.notopic));
  else if (T.scene) aiAddMsg(`"${LBL[T.scene]}" ya — enak nih subjeknya, gw padu mood & ornamen buat 3 varian.`);
  setTimeout(() => {
    const sets = aiSets(T, negs);
    aiAddMsg(aiR(AI_LINES.deliver));
    aiAddOpts(sets);
    $("#aiStatus").textContent = "3 set siap — klik buat pasang";
    AIState.busy = false;
  }, 340);
}

// rapikan spasi tanda baca dari tokenizer kata ("1 ." → "1.", "sip ," → "sip,")
function aiPretty(s) {
  return s.replace(/\s*sep\s*/g, " ")
    .replace(/\s+([.,:!?=•])/g, "$1");
}

// streaming token LLM → bubble sementara; pas selesai di-re-render jadi bersih
let aiStreamEl = null;
let aiStreamBuf = "";
function aiStream(w) {
  if (!aiStreamEl) {
    aiStreamEl = document.createElement("div");
    aiStreamEl.className = "ai-msg";
    $("#aiMsgs").appendChild(aiStreamEl);
  }
  aiStreamBuf += (aiStreamBuf ? " " : "") + w;
  aiStreamEl.textContent = aiPretty(aiStreamBuf);
  $("#aiMsgs").scrollTop = 1e9;
}
function aiFinish(text) {
  if (aiStreamEl) { aiStreamEl.remove(); aiStreamEl = null; }
  const buf = aiPretty((text || aiStreamBuf || ""));
  aiStreamBuf = "";
  // output model datar (tokenizer gak punya newline) → split sebelum "1." / "2." / "3."
  const lines = buf.split(/\n+|(?<=\S) (?=[123][.)]\s)/);
  const sets = [], bubbles = [];
  // token label → id tag: toleran teks nyangkut ("salju pilih yang sreg" → Salju + sisa jadi bubble)
  const aiLabelOf = (tok) => {
    const t = tok.replace(/\s*-\s*/g, "-").trim();
    if (LBLREV[t.toLowerCase()]) return { id: LBLREV[t.toLowerCase()], rest: "" };
    const words = t.split(" ");
    for (let n = Math.min(2, words.length); n >= 1; n--) {
      const cand = words.slice(0, n).join(" ").toLowerCase();
      if (LBLREV[cand]) return { id: LBLREV[cand], rest: words.slice(n).join(" ").trim() };
    }
    return null;
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    const mm = line.match(/^([123])[.)]\s*(.+?)\s*=\s*(.+)$/);
    if (mm) {
      const ids = [], rest = [];
      for (const tk of mm[3].split("•")) {
        const L = aiLabelOf(tk);
        if (L) { ids.push(L.id); if (L.rest) rest.push(L.rest); }
      }
      if (ids.length >= 2) {
        sets.push({ name: mm[2], ids });
        for (const r of rest) bubbles.push(r);
        continue;
      }
    }
    bubbles.push(line);
  }
  for (const b of bubbles) {
    if (/^(nih|ini 3|3 opsi|tinggal pilih|klik salah)/i.test(b)) continue;
    if (/cerita aja|sebut aja|coba ketik|misal /i.test(b)) continue; // jangan ada tutorial
    aiAddMsg(b);
  }
  if (sets.length) {
    aiAddOpts(sets.map((s) => ({
      name: s.name, scene: s.ids[0], color: s.ids[1],
      mood: s.ids[2] || null, orn: s.ids.slice(3),
    })));
    $("#aiStatus").textContent = "3 set siap — klik buat pasang";
  } else {
    aiAddMsg("(hmm jawaban modelnya rancu — gw racik manual ya 🙏)");
    const T = AIState.last && AIState.last.any ? AIState.last : aiTopic("");
    aiAddOpts(aiSets(T, AIState.neg));
    $("#aiStatus").textContent = "3 set siap — klik buat pasang";
  }
  AIState.busy = false;
}
function aiSendMsg() {
  const inp = $("#aiInput");
  const v = inp.value.trim();
  if (!v) return;
  aiAddMsg(v, "me");
  inp.value = "";
  $("#aiStatus").textContent = "ngirim…";
  aiReply(v);
}
// posisi panel: nempel di sisi fab yang ada ruang (kiri fab → panel kanan, dst)
function placeAIPanel() {
  const panel = $("#aiPanel"), fab = $("#aiFab");
  if (innerWidth <= 640) { panel.style.left = panel.style.right = panel.style.top = ""; return; }
  const wasHidden = panel.hidden;
  panel.hidden = false;
  panel.style.visibility = "hidden";
  panel.style.left = "0px"; panel.style.top = "0px"; panel.style.right = "auto";
  const fr = fab.getBoundingClientRect();
  const pw = panel.offsetWidth, ph = panel.offsetHeight;
  const left = (innerWidth - fr.right >= pw + 24)
    ? fr.right + 14
    : Math.max(12, fr.left - pw - 14);
  const top = Math.max(12, Math.min(innerHeight - ph - 12, fr.top - 24));
  panel.style.left = Math.round(left) + "px";
  panel.style.top = Math.round(top) + "px";
  panel.style.visibility = "";
  panel.hidden = wasHidden;
}
function toggleAIPanel(force) {
  const panel = $("#aiPanel");
  const show = force !== undefined ? force : panel.hidden;
  if (show) {
    placeAIPanel();
    panel.hidden = false;
    // animasi popup muncul — scale + naik + fade (di-retrigger tiap buka)
    panel.classList.remove("pop"); void panel.offsetWidth;
    panel.classList.add("pop");
    setTimeout(() => $("#aiInput").focus(), 60);
  } else {
    panel.hidden = true;
  }
}
function initAssistant() {
  const fab = $("#aiFab");
  // posisi tersimpan + clamp ke viewport
  let pos = null;
  try { pos = JSON.parse(localStorage.getItem("pixanva_fab") || "null"); } catch {}
  const clampXY = (x, y) => [
    Math.max(6, Math.min(innerWidth - 62, x)),
    Math.max(6, Math.min(innerHeight - 62, y)),
  ];
  if (pos) {
    const [x, y] = clampXY(pos.x, pos.y);
    fab.style.left = x + "px"; fab.style.top = y + "px";
  } else {
    fab.style.right = "18px"; fab.style.bottom = "18px";
  }
  let drag = null, moved = false;
  fab.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, ox: fab.offsetLeft, oy: fab.offsetTop };
    moved = false;
    fab.classList.add("hit"); // hit: kenceng dipencet
    try { fab.setPointerCapture(e.pointerId); } catch {}
  });
  fab.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 7) moved = true;
    if (moved) {
      const [x, y] = clampXY(drag.ox + dx, drag.oy + dy);
      fab.style.left = x + "px"; fab.style.top = y + "px";
      fab.style.right = "auto"; fab.style.bottom = "auto";
    }
  });
  const fabRelease = () => fab.classList.remove("hit"); // lepas → mantul balik mulus
  fab.addEventListener("pointerup", () => {
    fabRelease();
    if (drag && moved) {
      try { localStorage.setItem("pixanva_fab", JSON.stringify({ x: fab.offsetLeft, y: fab.offsetTop })); } catch {}
    } else if (drag && !moved) {
      toggleAIPanel();
      // denyut ring sekali pas panel kebuka
      fab.classList.remove("ping"); void fab.offsetWidth;
      fab.classList.add("ping");
      setTimeout(() => fab.classList.remove("ping"), 660);
    }
    drag = null;
  });
  fab.addEventListener("pointercancel", fabRelease);
  $("#aiClose").onclick = () => toggleAIPanel(false);
  $("#aiSend").onclick = aiSendMsg;
  $("#aiInput").addEventListener("keydown", (e) => { if (e.key === "Enter") aiSendMsg(); });
  $("#aiReask").onclick = () => {
    if (!AIState.last) {
      aiAddMsg("Belum ada topik yang dibahas 😄 cerita dulu mau gambar apa.");
      return;
    }
    $("#aiStatus").textContent = "ngirim…";
    aiReply("", true);
  };
  $("#aiNegBtn").onclick = () => {
    const row = $("#aiNegRow");
    row.hidden = !row.hidden;
    $("#aiNegBtn").classList.toggle("on", !row.hidden);
    if (!row.hidden) $("#aiNegInput").focus();
  };
  // klik di luar panel & fab → tutup
  document.addEventListener("pointerdown", (e) => {
    const panel = $("#aiPanel");
    if (panel.hidden) return;
    if (!panel.contains(e.target) && !fab.contains(e.target)) toggleAIPanel(false);
  });
}

// ganti mode tag <-> custom (persist)
const PM_EXAMPLES = [
  "ikan terbang di volkano",
  "naga api raksasa di angkasa malam",
  "kucing neon di kota malam ada petir",
  "robot es raksasa di tundra salju",
  "burung emas di pantai senja",
  "kupu kupu kristal di hutan mystic",
  "balon udara terbang di danau pagi",
  "kapal di laut badai ada petir",
  "rumah pohon di hutan kabut",
  "gajah raksasa di sawah senja",
];
function applyMode(m, save = true) {
  state.pmMode = m === "custom";
  const panel = $("#panel");
  if (panel) panel.classList.toggle("custom", state.pmMode);
  $("#modeTag").classList.toggle("on", !state.pmMode);
  $("#modeCustom").classList.toggle("on", state.pmMode);
  if (save) { try { localStorage.setItem("pixanva_pm_mode", state.pmMode ? "custom" : "tag"); } catch {} }
  setStatus(state.pmMode
    ? "mode custom — ketik apa aja, AI nggambar langsung ✍️"
    : "mode tag — pilih tag step-by-step");
}

// ---------- init ----------
(async function init() {
  initChips();
  initEvents();
  renderGallery();
  initAssistant();
  // bunuh universal select (hitbox biru / drag text) — inputs tetap bisa diketik
  document.addEventListener("selectstart", (e) => {
    const t = e.target;
    if (t && /^(INPUT|TEXTAREA)$/.test(t.tagName)) return;
    e.preventDefault();
  });
  // tema: pakai tema tersimpan; migrasi dari toggle capcut lama
  let th = null;
  try { th = localStorage.getItem("pixanva_theme"); } catch {}
  if (!th) { try { if (localStorage.getItem("pixanva_capcut") === "1") th = "capcut"; } catch {} }
  applyTheme(th || "classic-dark", false);
  // mode prompt: tag / custom (persist)
  let pmSaved = "tag";
  try { pmSaved = localStorage.getItem("pixanva_pm_mode") || "tag"; } catch {}
  applyMode(pmSaved === "custom" ? "custom" : "tag", false);
  // layar loading blur: default NYALA kecuali user mati-in sebelumnya
  let fx = true;
  try { fx = localStorage.getItem("pixanva_genfx") !== "0"; } catch {}
  applyGenfx(fx);
  loadMmSort();
  await bootModels();
  showStep(1);
})();
