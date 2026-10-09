# Pixanva

**AI image generator berbasis transformer — dibangun from scratch, jalan 100% di browser.**

Tanpa PyTorch. Tanpa TensorFlow. Tanpa pre-trained model. Seluruh arsitektur, attention,
RoPE 2D, backprop manual, optimizer, dan engine inferensi JavaScript ditulis dari nol —
bisa lo baca satu-satu di folder [`training/`](training/) dan [`js/`](js/).

## Cara pakai

1. Pilih model di sidebar — atau klik **More models** buat lihat semua versi (yang lama tetap bisa dipilih)
2. Ikuti wizard: **Pemandangan → Warna → Suasana → Ornamen** (ornamen bisa di-skip)
3. Pilih resolusi 32×32 sampai 128×128, atur seed (acak otomatis, bisa dikunci)
4. Tekan **Generate** — Pixanva melukis sel demi sel di depan lo
5. **Pengaturan** (gear): toggle **menu gaya CapCut** (UI berubah jadi editor ala CapCut —
   default mati), bersihkan galeri, tentang Pixanva

## Model

| Model | Parameter | Keterangan |
|---|---|---|
| Pixanva Light 1.0 | 813.696 | Cepat & ringan, paling sederhana (sengaja — ini versi "paling cacat"nya) |
| Pixanva Dark 1.0 | 2.700.096 | Seimbang, dukung grid 48 (96/128px) |
| Pixanva Dark 1.2 | 2.700.096 | Update menengah (+0.2): dilatih lanjutan 400+ step, lebih taat tag & komposisi lebih rapi |
| Pixanva Heavy QQ | 6.359.296 | Paling kuat & paling lambat |
| Pixanva Heavy QW | 6.359.296 | Update menengah (+0.2): dilatih lanjutan 450+ step + G48 lebih banyak |

Versi Light/Dark pakai desimal. Heavy pakai skema **QWERTY** — dua huruf sesuai layout keyboard
(qq, qw, qe … qm, lalu wq, ww …), dan sekarang tiap naik satu huruf = **+0.2 versi**.
Model lama gak pernah dihapus — semua bisa dipilih lewat popup **More models**.

Training: light 800 step (val 0.68) • dark 800+30 G48 (val 0.713) • **dark 1.2** 1230+45 G48
(val **0.681**) • heavy 1400+30 G48 (val 0.717) • **heavy QW** 1880+50 G48 (val **0.687**).


## Changelog vQW

- **2 model baru**: Pixanva **Dark 1.2** (warm-restart dari 1.0, +400 step bulk + 45 G48 —
  val 0.713 → 0.681) dan Pixanva **Heavy QW** (warm-restart dari QQ, +450 step bulk + 50 G48 —
  val 0.717 → 0.687). Keduanya lebih taat tag & komposisinya lebih rapi
  (banding: `preview/compare_*.png`, prompt & seed identik).
- **Tombol More models**: popup semua model yang pernah rilis — model lama tetap bisa dipilih.
- **Pengaturan** (gear di sidebar & topbar): toggle **menu gaya CapCut** (hitam pekat, aksen
  gradasi cyan-biru, bottom nav Buat/Galeri/Setelan, galeri & modal jadi bottom sheet —
  default mati, pilihan kesimpan), bersihkan galeri (konfirmasi 2 langkah),
  dan Tentang Pixanva pindah ke sini.
- Default sampling Dark 1.2 & Heavy QW: CFG 1.7, suhu 0.88/0.85.

## Changelog vQQ

- **FIX kritis engine.js**: slice bobot tensor 1-D (LayerNorm/bias) salah hitung (`r*c` dgn
  `c=undefined` → NaN) — seluruh forward sebelumnya menghasilkan NaN. Ini penyebab hasil
  blank/monokrom.
- **FIX kritis engine.js**: RoPE hanya diterapkan ke head pertama; head lain memakai query nol
  sehingga attention rata-rata tanpa arah. Sekarang semua head di-RoPE.
- Repetition penalty sampling disamakan dengan `training/eval.py` (window 12, 0.12×(1+count))
  — menghilangkan output "nyangkut satu warna".
- Heavy dilatih ulang sampai 1400 step + finetune grid 48×48 (output 96/128px).
- Seed sekarang acak tiap generate (bisa dikunci), wizard 4 langkah pilih-otomatis-lanjut,
  resolusi 32/64/96/128 dengan mapping grid 16/32/48, 128px = G48 dihaluskan
  (bilinear + kuantisasi palet + dither bayer).

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
