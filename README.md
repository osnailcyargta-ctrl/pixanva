# Pixanva

**AI image generator berbasis transformer — dibangun from scratch, jalan 100% di browser.**

Tanpa PyTorch. Tanpa TensorFlow. Tanpa pre-trained model. Seluruh arsitektur, attention,
RoPE 2D, backprop manual, optimizer, dan engine inferensi JavaScript ditulis dari nol —
bisa lo baca satu-satu di folder [`training/`](training/) dan [`js/`](js/).

## Cara pakai

1. Pilih model (Light 1.0 / Dark 1.0 / Heavy QQ)
2. Ikuti wizard: **Pemandangan → Warna → Suasana → Ornamen** (ornamen bisa di-skip)
3. Pilih resolusi 32×32 sampai 128×128, atur seed (acak otomatis, bisa dikunci)
4. Tekan **Generate** — Pixanva melukis sel demi sel di depan lo

## Model

| Model | Parameter | Keterangan |
|---|---|---|
| Pixanva Light 1.0 | 0.8M | Cepat & ringan, paling sederhana |
| Pixanva Dark 1.0 | 2.7M | Seimbang |
| Pixanva Heavy QQ | 6.3M | Paling kuat & paling lambat. Versi heavy ngikutin urutan qwerty: qq → qw → … → qm → wq → … |

## Arsitektur

- **Decoder-only transformer** (pre-LN, GELU, attention multi-head)
- **RoPE 2D faktoris** — posisi kondisi di (i, 0), sel gambar di (baris, kolom+16).
  Satu model bisa generate semua resolusi (grid 16/32/48 sel)
- **Kondisioning lewat prefix tokens**: tag pemandangan/warna/suasana/ornamen di-encode
  jadi token dan dikondisikan via attention + **classifier-free guidance** (12% kondisi
  dibuang saat training)
- **Vocab 159**: 96 warna palet tetap + 50 tag + special tokens
- Data training **digenerate prosedural** on-the-fly (16 pemandangan × 12 skema warna ×
  8 suasana × 14 ornamen, variasi seed tak terbatas) — tag dan piksel dijamin selalu aligned
- Training: numpy murni, backprop diturunin manual (lulus gradcheck 108/108), AdamW,
  cosine schedule, loss scaling anti-denormal

## Jalan di GitHub Pages

Repo ini statis — tinggal aktifkan:

**Settings → Pages → Deploy from a branch → `main` / `(root)` → Save**

## Struktur

```
index.html          UI
css/style.css       tema gelap
js/engine.js        transformer inference (fp16, KV-cache, RoPE 2D, CFG, sampling)
js/worker.js       Web Worker — melukis token demi token
js/app.js           wizard, galeri, kanvas
js/data.js          palet + tag (auto-generate dari Python)
models/*.bin        bobot fp16 + meta
training/*.py       seluruh ML stack from scratch (numpy)
preview/*.png       bukti data & hasil eval
```
