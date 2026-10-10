// Pixanva engine — transformer decoder-only, from scratch, tanpa framework.
// Mirror 1:1 dari training/model.py (numpy). Jalan di Web Worker.

export class FP16 {
  // decode float16 -> float32 manual (kompatibel semua browser)
  static decode(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) >> 15;
      const e = (h & 0x7c00) >> 10;
      const m = h & 0x03ff;
      let f;
      if (e === 0) f = m * 5.960464477539063e-8;          // subnormal
      else if (e === 31) f = m ? NaN : 1e38;               // inf
      else f = Math.pow(2, e - 15) * (1 + m / 1024);
      out[i] = s ? -f : f;
    }
    return out;
  }
}

// PRNG deterministik (seed -> deret acak) — mulberry32
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export async function fetchModel(url, onProgress) {
  // retry 3x — GitHub Pages di seluler suka gagal load tiba-tiba (jaringan pinggir jalan)
  let lastErr = null;
  for (let att = 0; att < 3; att++) {
    try {
      return await fetchOnce(url, onProgress);
    } catch (e) {
      lastErr = e;
      if (att < 2) await new Promise((r) => setTimeout(r, 700 * (att + 1)));
    }
  }
  throw lastErr;
}

async function fetchOnce(url, onProgress) {
  // timeout 30 dtk per percobaan — koneksi gantung jangan bikin nunggu selamanya
  const ac = new AbortController();
  const tmr = setTimeout(() => ac.abort(), 30000);
  try {
    return await download(url, onProgress, ac.signal);
  } finally {
    clearTimeout(tmr);
  }
}

async function download(url, onProgress, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error("HTTP " + res.status + " — " + url);
  const total = +(res.headers.get("content-length") || 0);
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress && total) onProgress(got / total, got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  // [u32 metaLen][meta][fp16 weights]
  const dv = new DataView(buf.buffer);
  const metaLen = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + metaLen)));
  const wbuf = buf.subarray(4 + metaLen);
  const nHalf = Math.floor(wbuf.length / 2);
  const u16 = new Uint16Array(nHalf);
  for (let i = 0; i < nHalf; i++) u16[i] = wbuf[i * 2] | (wbuf[i * 2 + 1] << 8);
  const weights = FP16.decode(u16);
  // potong per-param sesuai urutan & shape
  const cfg = meta.cfg;
  const d = cfg.d, L = cfg.L, m = cfg.mlp;
  const shapes = { "emb": [cfg.V, d], "lnf.g": [d], "lnf.b": [d] };
  for (let l = 0; l < L; l++) {
    const p = `l${l}.`;
    Object.assign(shapes, {
      [p + "ln1.g"]: [d], [p + "ln1.b"]: [d],
      [p + "wqkv"]: [d, 3 * d], [p + "bqkv"]: [3 * d],
      [p + "wo"]: [d, d], [p + "bo"]: [d],
      [p + "ln2.g"]: [d], [p + "ln2.b"]: [d],
      [p + "w1"]: [d, m], [p + "b1"]: [m],
      [p + "w2"]: [m, d], [p + "b2"]: [d],
    });
  }
  const W = {};
  let ptr = 0;
  for (const name of meta.order) {
    const sh = shapes[name];
    // shape bisa [r,c] (2D) atau [n] (1D utk ln/bias) — jangan r*c mentah!
    const n = sh.length === 2 ? sh[0] * sh[1] : sh[0];
    W[name] = weights.subarray(ptr, ptr + n);
    ptr += n;
  }
  return { meta, W };
}

export class Pixanva {
  constructor(meta, W) {
    this.meta = meta;
    this.W = W;
    this.cfg = meta.cfg;
    const c = this.cfg;
    this.V = c.V; this.d = c.d; this.L = c.L; this.H = c.H;
    this.hd = c.d / c.H; this.m = c.mlp;
    this.hdHalf = Math.floor(this.hd / 2);
    this.hdHalfY = this.hd - this.hdHalf;
    // freq RoPE (sama dgn numpy: theta^(-2i/n) dalam blok)
    // hd dibagi 2 blok: x-block (bx=hd/2), y-block (by=hd-bx). Tiap blok di-pair setengah.
    const theta = c.theta || 1000;
    const bx = this.hd >> 1, by = this.hd - bx;
    this.bx = bx; this.by = by;
    this.n2x = bx >> 1; this.n2y = by >> 1;
    this.fx = new Float32Array(this.n2x);
    for (let i = 0; i < this.n2x; i++) this.fx[i] = Math.pow(theta, -2 * i / bx);
    this.fy = new Float32Array(this.n2y);
    for (let i = 0; i < this.n2y; i++) this.fy[i] = Math.pow(theta, -2 * i / by);
  }

  // RoPE per blok: x-block pair (i, i+n2x), y-block pair (bx+i, bx+n2y+i)
  rope(v, pos, x, y) {
    const { bx, n2x, n2y, fx, fy } = this;
    for (let i = 0; i < n2x; i++) {
      const c = Math.cos(x * fx[i]), s = Math.sin(x * fx[i]);
      const a = v[pos + i], b = v[pos + i + n2x];
      v[pos + i] = a * c - b * s;
      v[pos + i + n2x] = a * s + b * c;
    }
    for (let i = 0; i < n2y; i++) {
      const c = Math.cos(y * fy[i]), s = Math.sin(y * fy[i]);
      const p1 = bx + i, p2 = bx + n2y + i;
      const a = v[pos + p1], b = v[pos + p2];
      v[pos + p1] = a * c - b * s;
      v[pos + p2] = a * s + b * c;
    }
  }

  ln(out, o, x, g, b, n) {
    let mu = 0;
    for (let i = 0; i < n; i++) mu += x[i];
    mu /= n;
    let va = 0;
    for (let i = 0; i < n; i++) { const t = x[i] - mu; va += t * t; }
    va /= n;
    const sg = 1 / Math.sqrt(va + 1e-5);
    for (let i = 0; i < n; i++) out[o + i] = (x[i] - mu) * sg * g[i] + b[i];
  }

  // forward 1 token. state: {T, K:[Float32Array/L], V:[...], x: Float32Array(d)}
  // ids: token id, pos2d: {x, y}. return logits Float32Array(V)
  step(state, token, px, py) {
    const { d, L, H, hd, m, V, W } = this;
    const T = state.T;
    // embedding
    const x = state.x;
    const emb = W["emb"];
    for (let i = 0; i < d; i++) x[i] = emb[token * d + i];

    const q = state.q || (state.q = new Float32Array(d)); // q per head dirope
    const h = state.h || (state.h = new Float32Array(d));
    const hh = state.hh || (state.hh = new Float32Array(m));
    const qkv = state.qkv || (state.qkv = new Float32Array(3 * d));

    for (let l = 0; l < L; l++) {
      const p = `l${l}.`;
      // LN1
      this.ln(h, 0, x, W[p + "ln1.g"], W[p + "ln1.b"], d);
      // qkv = h @ wqkv + bqkv  (wqkv: (d, 3d) row-major)
      const wq = W[p + "wqkv"], bq = W[p + "bqkv"];
      for (let j = 0; j < 3 * d; j++) qkv[j] = bq[j];
      for (let i = 0; i < d; i++) {
        const hv = h[i];
        if (hv === 0) continue;
        const ro = i * 3 * d;
        for (let j = 0; j < 3 * d; j++) qkv[j] += hv * wq[ro + j];
      }
      // simpan K,V ke cache; q di-rope (HASIL ROPE PER HEAD DISIMPAN —
      // bug lama: cuma head-0 ke-rope, head lain baca buffer nol → output monokrom)
      const K = state.K[l], Vc = state.V[l];
      K.set(qkv.subarray(d, 2 * d), T * d);
      Vc.set(qkv.subarray(2 * d, 3 * d), T * d);
      const qr = state.qr || (state.qr = new Float32Array(d));
      for (let hh2 = 0; hh2 < H; hh2++) {
        const off = hh2 * hd;
        q.set(qkv.subarray(off, off + hd));
        this.rope(q, 0, px, py);
        qr.set(q.subarray(0, hd), off);
        const kseg = K.subarray(T * d + off, T * d + off + hd);
        this.rope(kseg, 0, px, py);
      }
      // attention: untuk tiap head, q·K cache (semua t<=T) → softmax → AV
      const attnOut = state.attn || (state.attn = new Float32Array(d));
      const scale = 1 / Math.sqrt(hd);
      const scores = state.scores || (state.scores = new Float32Array(4096 + 24));
      for (let hh2 = 0; hh2 < H; hh2++) {
        const off = hh2 * hd;
        let mx = -1e30;
        for (let t = 0; t <= T; t++) {
          let s = 0;
          const kr = t * d + off;
          for (let i = 0; i < hd; i++) s += qr[off + i] * K[kr + i];
          s *= scale;
          scores[t] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let t = 0; t <= T; t++) { const e = Math.exp(scores[t] - mx); scores[t] = e; sum += e; }
        const inv = 1 / sum;
        const or = off; // output segmen attnOut[off..off+hd]
        for (let i = 0; i < hd; i++) attnOut[or + i] = 0;
        for (let t = 0; t <= T; t++) {
          const wgt = scores[t] * inv;
          if (wgt < 1e-12) continue;
          const vr = t * d + off;
          for (let i = 0; i < hd; i++) attnOut[or + i] += wgt * Vc[vr + i];
        }
      }
      // proj + residual
      const wo = W[p + "wo"], bo = W[p + "bo"];
      for (let i = 0; i < d; i++) h[i] = bo[i];
      for (let i = 0; i < d; i++) {
        const av = attnOut[i];
        if (av === 0) continue;
        const ro = i * d;
        for (let j = 0; j < d; j++) h[j] += av * wo[ro + j];
      }
      for (let i = 0; i < d; i++) x[i] += h[i];
      // LN2 + MLP
      this.ln(h, 0, x, W[p + "ln2.g"], W[p + "ln2.b"], d);
      const w1 = W[p + "w1"], b1 = W[p + "b1"], w2 = W[p + "w2"], b2 = W[p + "b2"];
      for (let j = 0; j < m; j++) hh[j] = b1[j];
      for (let i = 0; i < d; i++) {
        const hv = h[i];
        if (hv === 0) continue;
        const ro = i * m;
        for (let j = 0; j < m; j++) hh[j] += hv * w1[ro + j];
      }
      // GELU tanh
      for (let j = 0; j < m; j++) {
        const v = hh[j];
        hh[j] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
      }
      for (let i = 0; i < d; i++) h[i] = b2[i];
      for (let j = 0; j < m; j++) {
        const av = hh[j];
        if (av === 0) continue;
        const ro = j * d;
        for (let i = 0; i < d; i++) h[i] += av * w2[ro + i];
      }
      for (let i = 0; i < d; i++) x[i] += h[i];
    }
    state.T = T + 1;
    // final LN + tied head
    this.ln(h, 0, x, W["lnf.g"], W["lnf.b"], d);
    const logits = state.logits || (state.logits = new Float32Array(V));
    for (let v = 0; v < V; v++) {
      let s = 0;
      const ro = v * d;
      for (let i = 0; i < d; i++) s += h[i] * emb[ro + i];
      logits[v] = s;
    }
    return logits;
  }

  newState(maxT) {
    const { d, L } = this;
    const T = Math.min(maxT || 4200, 4400);
    const st = { T: 0, x: new Float32Array(d), K: [], V: [] };
    for (let l = 0; l < L; l++) {
      st.K.push(new Float32Array(T * d));
      st.V.push(new Float32Array(T * d));
    }
    return st;
  }
}

// sampling: temperature + top-k + repetition penalty (disamakan dgn training/eval.py)
export function sampleToken(logits, temp, topK, rng, recent, repPenalty) {
  const V = logits.length;
  const adj = new Float32Array(V);
  for (let i = 0; i < V; i++) adj[i] = logits[i] / temp;
  if (recent && repPenalty && repPenalty > 1) {
    const seen = new Set(recent);
    for (const t of seen) {
      const c = recent.filter((r) => r === t).length;
      adj[t] -= 0.12 * (1 + c);
    }
  }
  // top-k indices
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => adj[b] - adj[a]);
  const k = Math.min(topK, V);
  const cut = idx.slice(0, k);
  let mx = -1e30;
  for (const i of cut) mx = Math.max(mx, adj[i]);
  let sum = 0;
  const ps = cut.map(i => { const e = Math.exp(adj[i] - mx); sum += e; return e; });
  let r = rng() * sum;
  for (let i = 0; i < cut.length; i++) {
    r -= ps[i];
    if (r <= 0) return cut[i];
  }
  return cut[cut.length - 1];
}
