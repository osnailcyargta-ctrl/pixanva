"""Pixanva subjects — objek yang BENERAN digambar di grid (ikan, naga, robot, ...).
Dipanggil render.py SETELAH color grading (biar warna eksplisit — "ikan merah" tetap merah),
SEBELUM fog/glow mood (biar subjek kena atmosfer dikit).
Semua shape prosedural: aman di G16 (3-6 px) sampai G48.
"""
import numpy as np
from render import hsv

# ---- klasifikasi perilaku penempatan ----
SKY_SCENES = {"angkasa", "awan"}
WATER_SCENES = {"laut", "danau", "pantai", "kanjon", "terjun"}
SWIMMERS = {"ikan", "kapal", "naga"}
FLYERS = {"burung", "kupu", "balon", "naga"}   # default udara walau gak ada "terbang"
GROUND_ONLY = {"pohon", "bunga", "rumah"}       # gak mau berenang, gak terbang default

# ---- warna kata → HSV (subjek eksplisit, tahan grading karena digambar SETELAH grade) ----
COLOR_WORDS = {
    "merah": (0.00, 0.85, 0.90), "biru": (0.60, 0.80, 0.85), "hijau": (0.36, 0.80, 0.80),
    "hitam": (0.0, 0.0, 0.12),   "putih": (0.0, 0.02, 0.98), "ungu": (0.76, 0.70, 0.75),
    "pink": (0.92, 0.65, 0.95),  "oranye": (0.07, 0.90, 0.95), "kuning": (0.14, 0.85, 0.95),
    "perak": (0.60, 0.06, 0.88), "emas": (0.115, 0.72, 0.95), "es": (0.52, 0.30, 0.96),
}
ATTR_COLOR = {
    "emas": (0.115, 0.72, 0.95),
    "es": (0.52, 0.30, 0.96),
    "neon": None,     # hue acak full sat — dipilih rng
    "kristal": None,  # gradasi magenta→cyan per y
}

# ---- default warna per subjek: (primary, secondary) fungsi rng ----
def _c_ikan(rng):  return hsv(0.55, 0.45, 0.92), hsv(0.08, 0.75, 0.85)
def _c_burung(rng): return hsv(0.09, 0.70, 0.85), hsv(0.60, 0.30, 0.30)
def _c_naga(rng):  return hsv(0.36, 0.70, 0.72), hsv(0.13, 0.85, 0.92)
def _c_kucing(rng): return hsv(0.07, 0.65, 0.85), hsv(0.06, 0.55, 0.45)
def _c_robot(rng): return hsv(0.62, 0.08, 0.72), hsv(0.52, 0.85, 0.95)
def _c_kapal(rng): return hsv(0.05, 0.55, 0.38), hsv(0.0, 0.02, 0.96)
def _c_kupu(rng):  return hsv(rng.uniform(0, 1), 0.85, 0.95), hsv(rng.uniform(0, 1), 0.80, 0.90)
def _c_pohon(rng): return hsv(0.36, 0.60, 0.30), hsv(0.07, 0.55, 0.30)
def _c_bunga(rng): return hsv(rng.choice([0.98, 0.13, 0.75, 0.55]), 0.85, 0.95), hsv(0.33, 0.70, 0.45)
def _c_balon(rng): return hsv(rng.choice([0.0, 0.55, 0.12]), 0.85, 0.92), hsv(0.09, 0.70, 0.55)
def _c_rumah(rng): return hsv(0.09, 0.45, 0.88), hsv(0.02, 0.70, 0.45)
def _c_gajah(rng): return hsv(0.08, 0.06, 0.62), hsv(0.08, 0.05, 0.45)

DRAW = {}  # diisi di bawah: id -> fn(cv, G, rng, x, y, s, c1, c2, fly)

# ---------- helper gambar kecil ----------

def _px(cv, x, y, c):
    x, y = int(round(x)), int(round(y))
    if 0 <= x < cv.shape[1] and 0 <= y < cv.shape[0]:
        cv[y, x] = c

def _ell(cv, cx, cy, rx, ry, c):
    G = cv.shape[0]
    x0, x1 = int(max(0, cx - rx - 1)), int(min(G - 1, cx + rx + 1))
    y0, y1 = int(max(0, cy - ry - 1)), int(min(G - 1, cy + ry + 1))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    m = ((xx - cx) / max(rx, 0.6)) ** 2 + ((yy - cy) / max(ry, 0.6)) ** 2 <= 1.0
    sub = cv[y0:y1 + 1, x0:x1 + 1]
    sub[m] = c

def _rect(cv, x0, y0, x1, y1, c):
    a, b = int(round(min(x0, x1))), int(round(max(x0, x1)))
    d, e = int(round(min(y0, y1))), int(round(max(y0, y1)))
    cv[max(0, d):e + 1, max(0, a):b + 1] = c

def _poly(cv, pts, c):
    """poly convex (list (x,y)) — fill via tanda cross product (robust utk shape kecil)."""
    G = cv.shape[0]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = int(max(0, min(xs) - 1)), int(min(G - 1, max(xs) + 1))
    y0, y1 = int(max(0, min(ys) - 1)), int(min(G - 1, max(ys) + 1))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.ones(xx.shape, dtype=bool)
    sign = 0
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]; bx, by = pts[(i + 1) % n]
        cr = (bx - ax) * (yy - ay) - (by - ay) * (xx - ax)
        s = np.sign(cr)
        if sign == 0:
            if (s != 0).any():
                sign = s[(s != 0)][0]
        inside &= (s == sign) | (s == 0)
    sub = cv[y0:y1 + 1, x0:x1 + 1]
    sub[inside] = c

def _line(cv, x0, y0, x1, y1, c, thick=1):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(n):
        t = i / max(n - 1, 1)
        _px(cv, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, c)
        if thick > 1:
            _px(cv, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t + 1, c)

def _mix(c, other, t):
    return np.asarray(c) * (1 - t) + np.asarray(other) * t

# ---------- bentuk subjek ----------
# kontrak: fn(cv, G, rng, x, y, s, c1, c2, fly) — (x,y) titik pusat badan, s = radius dasar

def _wings(cv, G, x, y, s, c, up=True):
    """sayap segitiga kiri-kanan (buat subjek yang dipaksa terbang)."""
    dy = -s * 1.5 if up else s * 0.9
    _poly(cv, [(x - s * 0.4, y), (x - s * 1.7, y + dy * 0.8), (x - s * 0.5, y + s * 0.5)], c)
    _poly(cv, [(x + s * 0.4, y), (x + s * 1.7, y + dy * 0.8), (x + s * 0.5, y + s * 0.5)], c)

def _streaks(cv, G, x, y, s, sky):
    """garis gerak di belakang subjek terbang."""
    c = _mix(sky, hsv(0.0, 0.0, 1.0), 0.55)
    for k in range(2):
        yy = y + (k - 0.5) * s * 0.8
        _line(cv, x - s * 2.2, yy, x - s * 1.1, yy, c)

def s_ikan(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x, y, s, s * 0.55, c1)
    d = 1 if rng.random() < 0.5 else -1
    _poly(cv, [(x - d * s * 0.9, y), (x - d * s * 1.8, y - s * 0.55), (x - d * s * 1.8, y + s * 0.55)], c2)
    _px(cv, x + d * s * 0.55, y - s * 0.15, hsv(0.0, 0.0, 0.08))
    _poly(cv, [(x + d * s * 0.1, y - s * 0.5), (x + d * s * 0.45, y - s * 1.1), (x + d * s * 0.6, y - s * 0.45)], c2)
    if fly:
        _wings(cv, G, x, y, s * 0.9, _mix(c2, hsv(0, 0, 1), 0.25))

def s_burung(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x, y, s * 0.6, s * 0.42, c1)
    dy = -s * 1.1
    _poly(cv, [(x - s * 0.2, y - s * 0.1), (x - s * 1.6, y + dy), (x - s * 0.7, y + s * 0.3)], c2)
    _poly(cv, [(x + s * 0.2, y - s * 0.1), (x + s * 1.6, y + dy), (x + s * 0.7, y + s * 0.3)], c2)
    _px(cv, x + s * 0.7, y - s * 0.1, hsv(0.10, 0.85, 0.9))
    _line(cv, x - s * 0.55, y + s * 0.35, x - s * 1.0, y + s * 0.8, c2)

def s_naga(cv, G, rng, x, y, s, c1, c2, fly):
    n = 7
    for i in range(n):
        t = i / (n - 1)
        bx = x - s * 1.4 + t * s * 2.8
        by = y + np.sin(t * 6.0 + rng.uniform(0, 1)) * s * 0.45
        _ell(cv, bx, by, s * 0.42, s * 0.36, c1 if i % 2 == 0 else _mix(c1, hsv(0.13, 0.85, 0.9), 0.35))
        if i in (1, 3, 5):
            _poly(cv, [(bx, by - s * 0.3), (bx - s * 0.2, by - s * 0.85), (bx + s * 0.2, by - s * 0.85)],
                  hsv(0.13, 0.85, 0.95))
    hx = x + s * 1.5
    hy = y + np.sin(6.0 + rng.uniform(0, 1)) * s * 0.2 - s * 0.25
    _ell(cv, hx, hy, s * 0.58, s * 0.46, c1)
    _poly(cv, [(hx - s * 0.2, hy - s * 0.4), (hx - s * 0.1, hy - s * 0.9), (hx + s * 0.15, hy - s * 0.4)], c2)
    _px(cv, hx + s * 0.28, hy - s * 0.12, hsv(0.0, 0.9, 0.98))
    if fly or rng.random() < 0.6:
        _wings(cv, G, x, y - s * 0.3, s * 1.15, _mix(c2, hsv(0, 0, 1), 0.15))

def s_kucing(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x, y, s * 0.95, s * 0.65, c1)
    hx, hy = x + s * 0.75, y - s * 0.55
    _ell(cv, hx, hy, s * 0.48, s * 0.42, c1)
    _poly(cv, [(hx - s * 0.42, hy - s * 0.25), (hx - s * 0.5, hy - s * 0.95), (hx - s * 0.05, hy - s * 0.45)], c2)
    _poly(cv, [(hx + s * 0.42, hy - s * 0.25), (hx + s * 0.5, hy - s * 0.95), (hx + s * 0.05, hy - s * 0.45)], c2)
    _px(cv, hx + s * 0.18, hy - s * 0.05, hsv(0.5, 0.6, 0.2))
    _px(cv, hx + s * 0.38, hy - s * 0.05, hsv(0.5, 0.6, 0.2))
    _line(cv, x - s * 0.9, y + s * 0.2, x - s * 1.5, y - s * 0.4, c1, thick=1)

def s_robot(cv, G, rng, x, y, s, c1, c2, fly):
    _rect(cv, x - s * 0.7, y - s * 0.6, x + s * 0.7, y + s * 1.0, c1)
    _rect(cv, x - s * 0.45, y - s * 1.35, x + s * 0.45, y - s * 0.7, c1)
    _line(cv, x, y - s * 1.35, x, y - s * 1.8, c2)
    _px(cv, x, y - s * 1.9, c2)
    _px(cv, x - s * 0.2, y - s * 1.0, c2)
    _px(cv, x + s * 0.2, y - s * 1.0, c2)
    _rect(cv, x - s * 1.05, y - s * 0.4, x - s * 0.75, y + s * 0.5, c2)
    _rect(cv, x + s * 0.75, y - s * 0.4, x + s * 1.05, y + s * 0.5, c2)
    _rect(cv, x - s * 0.5, y + s * 1.0, x - s * 0.2, y + s * 1.35, c2)
    _rect(cv, x + s * 0.2, y + s * 1.0, x + s * 0.5, y + s * 1.35, c2)

def s_kapal(cv, G, rng, x, y, s, c1, c2, fly):
    _poly(cv, [(x - s * 1.1, y - s * 0.35), (x + s * 1.1, y - s * 0.35), (x + s * 0.7, y + s * 0.3), (x - s * 0.7, y + s * 0.3)], c1)
    _line(cv, x, y - s * 0.35, x, y - s * 1.7, c1, thick=1)
    _poly(cv, [(x + s * 0.1, y - s * 1.6), (x + s * 1.0, y - s * 0.5), (x + s * 0.1, y - s * 0.5)], c2)
    _px(cv, x - s * 0.15, y - s * 1.8, c2)

def s_kupu(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x - s * 0.5, y - s * 0.4, s * 0.55, s * 0.45, c1)
    _ell(cv, x + s * 0.5, y - s * 0.4, s * 0.55, s * 0.45, c2)
    _ell(cv, x - s * 0.4, y + s * 0.35, s * 0.42, s * 0.36, c2)
    _ell(cv, x + s * 0.4, y + s * 0.35, s * 0.42, s * 0.36, c1)
    _line(cv, x, y - s * 0.5, x, y + s * 0.7, hsv(0.0, 0.0, 0.15))

def s_pohon(cv, G, rng, x, y, s, c1, c2, fly):
    _rect(cv, x - s * 0.16, y - s * 0.2, x + s * 0.16, y + s * 0.9, c2)
    _ell(cv, x, y - s * 0.7, s * 0.75, s * 0.7, c1)
    _ell(cv, x - s * 0.45, y - s * 0.35, s * 0.5, s * 0.45, _mix(c1, hsv(0.33, 0.7, 0.4), 0.5))
    _ell(cv, x + s * 0.45, y - s * 0.35, s * 0.5, s * 0.45, _mix(c1, hsv(0.30, 0.7, 0.22), 0.5))

def s_bunga(cv, G, rng, x, y, s, c1, c2, fly):
    _line(cv, x, y + s * 1.2, x, y - s * 0.1, c2, thick=1)
    for k in range(6):
        a = k / 6 * 6.283
        _ell(cv, x + np.cos(a) * s * 0.55, y - s * 0.35 + np.sin(a) * s * 0.55, s * 0.42, s * 0.42, c1)
    _ell(cv, x, y - s * 0.35, s * 0.3, s * 0.3, hsv(0.12, 0.9, 0.95))

def s_balon(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x, y - s * 0.4, s * 0.75, s * 1.0, c1)
    _ell(cv, x - s * 0.35, y - s * 0.6, s * 0.28, s * 0.4, _mix(c1, hsv(0, 0, 1), 0.35))
    _line(cv, x - s * 0.3, y + s * 0.55, x - s * 0.18, y + s * 0.9, c2)
    _line(cv, x + s * 0.3, y + s * 0.55, x + s * 0.18, y + s * 0.9, c2)
    _rect(cv, x - s * 0.25, y + s * 0.9, x + s * 0.25, y + s * 1.2, c2)

def s_rumah(cv, G, rng, x, y, s, c1, c2, fly):
    _rect(cv, x - s * 0.9, y - s * 0.2, x + s * 0.9, y + s * 0.8, c1)
    _poly(cv, [(x - s * 1.05, y - s * 0.2), (x, y - s * 1.15), (x + s * 1.05, y - s * 0.2)], c2)
    _rect(cv, x - s * 0.2, y + s * 0.15, x + s * 0.2, y + s * 0.8, c2)
    _px(cv, x + s * 0.45, y + s * 0.1, hsv(0.13, 0.6, 0.95))

def s_gajah(cv, G, rng, x, y, s, c1, c2, fly):
    _ell(cv, x, y, s * 1.05, s * 0.75, c1)
    hx, hy = x + s * 0.95, y - s * 0.35
    _ell(cv, hx, hy, s * 0.55, s * 0.55, c1)
    _ell(cv, hx - s * 0.1, hy + s * 0.05, s * 0.3, s * 0.4, c2)
    _line(cv, hx + s * 0.3, hy + s * 0.3, hx + s * 0.45, hy + s * 1.0, c1, thick=1)
    _px(cv, hx + s * 0.55, hy + s * 0.35, hsv(0.1, 0.05, 0.95))
    _rect(cv, x - s * 0.7, y + s * 0.55, x - s * 0.4, y + s * 1.0, c2)
    _rect(cv, x + s * 0.3, y + s * 0.55, x + s * 0.6, y + s * 1.0, c2)

DRAW.update({
    "ikan": s_ikan, "burung": s_burung, "naga": s_naga, "kucing": s_kucing,
    "robot": s_robot, "kapal": s_kapal, "kupu": s_kupu, "pohon": s_pohon,
    "bunga": s_bunga, "balon": s_balon, "rumah": s_rumah, "gajah": s_gajah,
})

SUBJ_DEFAULT_COLOR = {
    "ikan": _c_ikan, "burung": _c_burung, "naga": _c_naga, "kucing": _c_kucing,
    "robot": _c_robot, "kapal": _c_kapal, "kupu": _c_kupu, "pohon": _c_pohon,
    "bunga": _c_bunga, "balon": _c_balon, "rumah": _c_rumah, "gajah": _c_gajah,
}

SUBJECTS = list(DRAW.keys())
ATTRS = ["terbang", "raksasa", "kecil", "neon", "emas", "api", "es", "kristal"]

# ---------- efek atribut global (sekitar subjek) ----------

def _fx_fire(cv, G, rng, x, y, s):
    glow = hsv(0.05, 0.85, 0.55)
    _ell(cv, x, y, s * 1.7, s * 1.5, _mix(cv[int(np.clip(y, 0, G - 1)), int(np.clip(x, 0, G - 1))], glow, 0.5))
    n = int(rng.integers(10, 19))
    for _ in range(n):
        px = x + rng.uniform(-1.3, 1.3) * s
        py = y + rng.uniform(-1.5, 0.7) * s
        h = rng.uniform(0.02, 0.10)
        v = rng.uniform(0.85, 1.0)
        _px(cv, px, py, hsv(h, 0.95, v))
    for _ in range(int(rng.integers(2, 5))):   # bara naik di atas
        _px(cv, x + rng.uniform(-0.8, 0.8) * s, y + rng.uniform(-2.0, -1.4) * s,
            hsv(0.09, 0.9, rng.uniform(0.9, 1.0)))

def _fx_ice(cv, G, rng, x, y, s):
    for _ in range(int(rng.integers(4, 9))):
        _px(cv, x + rng.uniform(-1.3, 1.3) * s, y + rng.uniform(-1.1, 1.1) * s, hsv(0.55, 0.05, 1.0))

def _fx_neon(cv, G, rng, x, y, s, col):
    _ell(cv, x, y, s * 1.5, s * 1.3, _mix(cv[int(np.clip(y, 0, G - 1)), int(np.clip(x, 0, G - 1))], col, 0.38))

# ---------- placement utama ----------

def draw_subject(cv, G, rng, cond, ctx, sk):
    subj = cond["subj"]
    sid = subj["id"]
    if sid not in DRAW:
        return
    attrs = list(subj.get("attrs", []))
    fly = "terbang" in attrs or cond.get("scene") in SKY_SCENES or sid in FLYERS
    giant = "raksasa" in attrs
    tiny = "kecil" in attrs

    # ---- posisi ----
    water = ctx.get("water", False)
    wy = ctx.get("water_y", None)
    dock = False
    if fly:
        x = rng.uniform(0.28, 0.72) * G
        y = rng.uniform(0.13, 0.42) * G if cond.get("scene") not in SKY_SCENES else rng.uniform(0.15, 0.60) * G
    elif water and sid in SWIMMERS and wy is not None:
        x = rng.uniform(0.3, 0.7) * G
        y = (wy + rng.uniform(1.5, max(2.0, (G - wy) * 0.6))) if sid == "ikan" else wy - 0.2
        y = min(y, G - 1.5)
    elif cond.get("scene") in WATER_SCENES:
        # non-penyelam di scene air → berdiri di dermaga kecil (digambar abis skala kehitung)
        x = rng.uniform(0.3, 0.7) * G
        y = G * 0.78
        dock = True
    else:
        gy = ctx.get("ground")
        x = rng.uniform(0.3, 0.7) * G
        y = (gy[int(x)] if gy is not None else G - 1) + 0.5
    if giant:
        x = np.clip(x, G * 0.34, G * 0.66)

    # ---- skala ----
    s = G * rng.uniform(0.14, 0.19)
    if giant:
        s *= 1.9
    if tiny:
        s *= 0.55
    s = max(s, 1.4)

    # subjek darat jangan kepotong tepi bawah grid
    if not fly and not (water and sid in SWIMMERS and wy is not None):
        y = min(y, G - 1 - s * 0.75)

    # ---- warna ----
    colw = subj.get("col")
    if colw in COLOR_WORDS:
        c1 = hsv(*COLOR_WORDS[colw])
        c2 = _mix(c1, hsv(0.0, 0.0, 0.15), 0.45)
    else:
        c1, c2 = SUBJ_DEFAULT_COLOR[sid](rng)
    if "neon" in attrs and colw not in COLOR_WORDS:
        c1 = hsv(rng.uniform(0, 1), 1.0, 0.95)
        c2 = hsv((c1[0] + 0.5) % 1.0, 1.0, 0.9)
    if "kristal" in attrs and colw not in COLOR_WORDS:
        c1 = hsv(0.85, 0.55, 0.95)
        c2 = hsv(0.52, 0.60, 0.95)

    sky_col = sk[1]

    if dock:
        _ell(cv, x, y + s * 1.1, s * 1.1, s * 0.28, _mix(sky_col, hsv(0.07, 0.55, 0.25), 0.7))

    # bayangan di tanah/air kalau terbang
    if fly and cond.get("scene") not in SKY_SCENES:
        sh_y = ctx.get("ground")[int(np.clip(x, 0, G - 1))] if ctx.get("ground") is not None else (wy or G * 0.85)
        _ell(cv, x, min(sh_y + 0.5, G - 1), s * 0.9, s * 0.22, _mix(sky_col, hsv(0, 0, 0), 0.35))

    # efek belakang
    if "api" in attrs:
        _fx_fire(cv, G, rng, x, y, s)
    elif "neon" in attrs:
        _fx_neon(cv, G, rng, x, y, s, c1)

    DRAW[sid](cv, G, rng, x, y, s, c1, c2, fly)

    # efek depan / detail
    if "kristal" in attrs:
        _line(cv, x - s, y - s, x + s * 0.4, y + s, _mix(c2, hsv(0, 0, 1), 0.4))
        _line(cv, x - s * 0.4, y - s, x + s, y + s * 0.6, _mix(c1, hsv(0, 0, 1), 0.5))
    if "es" in attrs:
        _fx_ice(cv, G, rng, x, y, s)
    if fly:
        _streaks(cv, G, x, y, s, sky_col)
        # sayap ekstra cuma buat subjek yang drawernya gak punya sayap sendiri
        if sid in ("kucing", "robot", "gajah", "rumah", "kapal", "pohon", "bunga", "balon"):
            _wings(cv, G, x, y - s * 0.2, s * 0.85, _mix(c2, hsv(0, 0, 1), 0.3))
    if "api" in attrs:
        for _ in range(int(rng.integers(4, 8))):
            _px(cv, x + rng.uniform(-1.0, 1.0) * s, y + rng.uniform(-1.6, -0.9) * s,
                hsv(rng.uniform(0.03, 0.09), 0.9, rng.uniform(0.85, 1.0)))
