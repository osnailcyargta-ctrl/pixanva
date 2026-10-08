"""Pixanva procedural renderer — (tag, gambar) SELALU aligned.
Tiap scene digambar di grid G x G cell (1 cell = 1 warna palet), lalu di-upscale 2x
saat decode. Tag warna = color grading HSV global -> menjamin tag nempel di piksel.
"""
import colorsys
import numpy as np
import palette as PA
import tags as TG

# ---------- helpers ----------

def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1)))

def sines(rng, G, n=3, lo=1.0, hi=4.0):
    """kurva halus period-1 di atas G titik, amplitudo ~1"""
    x = np.arange(G) / G * 2 * np.pi
    y = np.zeros(G)
    for _ in range(n):
        f = rng.uniform(lo, hi)
        a = 1.0 / (1.0 + f)
        y += a * np.sin(f * x + rng.uniform(0, 6.28))
    y -= y.min()
    y = y / max(y.max(), 1e-6) * 2 - 1
    return y

def vgrad(top, bot, n):
    t = np.linspace(0, 1, n)[:, None, None]
    return top[None, None, :] * (1 - t) + bot[None, None, :] * t

def fill_below(cv, ycol, color, mix_sky=None, sky=None):
    """isi dari ycol[x]..bawah dengan color (bisa per-x array (G,3))"""
    G = cv.shape[0]
    x = np.arange(G)
    yy = np.arange(G)[:, None]
    mask = yy >= ycol[None, :]
    c = np.asarray(color, dtype=float)
    if c.ndim == 1:
        c = np.broadcast_to(c, (G, 3))
    if mix_sky and sky is not None:
        m = np.broadcast_to(mix_sky, (G, 1))
        c = c * (1 - m) + sky * m
    cv[mask] = c[np.repeat(x[None, :], G, 0)[mask]]
    return mask

def blob(cv, cx, cy, r, color, squash=1.0, soft=0.0):
    G = cv.shape[0]
    yy, xx = np.mgrid[0:G, 0:G]
    d2 = ((xx - cx) ** 2 + ((yy - cy) / squash) ** 2) / max(r * r, 1e-6)
    m = d2 <= 1.0
    if soft > 0:
        t = np.clip((1.0 - d2) / max(soft, 1e-6), 0, 1)
        cv[:] = cv * (1 - (m * t)[..., None]) + color * (m * t)[..., None]
    else:
        cv[m] = color

SKY = {
    "pagi":   ((0.56, 0.45, 0.72), (0.09, 0.32, 0.93)),
    "siang":  ((0.585, 0.60, 0.85), (0.575, 0.28, 0.96)),
    "senja":  ((0.76, 0.52, 0.42), (0.07, 0.85, 0.92)),
    "malam":  ((0.665, 0.68, 0.09), (0.63, 0.52, 0.20)),
    "badai":  ((0.61, 0.16, 0.28), (0.60, 0.10, 0.44)),
    "mystic": ((0.745, 0.58, 0.26), (0.48, 0.52, 0.52)),
    "mimpi":  ((0.80, 0.38, 0.78), (0.10, 0.42, 0.95)),
}

def sky_for(mood, rng):
    spec = next((m[2] for m in TG.MOODS if m[0] == mood), None)
    key = spec["sky"] if spec else ["siang", "senja", "malam"][int(rng.integers(3))]
    t, b = SKY[key]
    return hsv(*t), hsv(*b), key

def stars(cv, rng, density, G, y_max=0.65):
    if density <= 0:
        return
    n = int(rng.integers(4, 16) * density * G / 16)
    for _ in range(max(n, 0)):
        x, y = int(rng.integers(0, G)), int(rng.integers(0, max(int(G * y_max), 1)))
        c = hsv(0.13, 0.25, rng.uniform(0.8, 1.0)) if rng.random() < 0.25 else hsv(0.58, 0.05, rng.uniform(0.85, 1.0))
        cv[y, x] = c
        if rng.random() < 0.2 and x + 1 < G:
            cv[y, x + 1] = c * 0.8

# ---------- scenes ----------
# tiap scene: cv digambar, return ctx {"ground": ycol|None, "water": bool, "glow": [(x,y,r,c)]}

def sc_gunung(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    cols = [hsv(0.60 + rng.uniform(-0.05, 0.05), 0.35, 0.55 - i * 0.14) for i in range(3)]
    base = G * rng.uniform(0.45, 0.6)
    for i in range(3):
        amp = G * rng.uniform(0.10, 0.22)
        ycol = (base + i * G * 0.10 + amp * sines(rng, G)).astype(int)
        fill_below(cv, np.clip(ycol, 0, G - 1), cols[i], mix_sky=0.15 * (2 - i) / 2, sky=bot)
        if i == 2 and rng.random() < 0.5:
            peak = ycol.min()
            for x in range(G):
                if ycol[x] - peak < G * 0.08:
                    cv[ycol[x]:ycol[x] + 2, x] = hsv(0.58, 0.06, 0.93)
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_laut(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    hz = int(G * rng.uniform(0.38, 0.52))
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    sea_top, sea_bot = hsv(0.56, 0.55, 0.30), hsv(0.57, 0.60, 0.16)
    cv[hz:] = vgrad(sea_top, sea_bot, G - hz)
    for k in range(int((G - hz) * 0.6)):
        y = hz + int(rng.integers(0, max(1, G - hz)))
        cv[y, :] = cv[y, :] * 0.86 + hsv(0.55, 0.40, 0.55) * 0.14
    sx = int(rng.integers(2, G - 2))
    cv[hz:, sx] = cv[hz:, sx] * 0.6 + hsv(0.12, 0.55, 0.9) * 0.4
    ctx["water"] = True
    ctx["water_y"] = hz
    return ctx

def sc_hutan(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    layers = int(rng.integers(2, 4))
    for i in range(layers):
        base = G * (0.35 + 0.22 * i)
        dark = hsv(0.38 + rng.uniform(-0.04, 0.04), 0.55, 0.28 - i * 0.06)
        n = int(rng.integers(4, 9))
        for t in range(n):
            tx = int(rng.integers(0, G))
            th = int(G * rng.uniform(0.14, 0.26))
            w = max(int(rng.integers(1, 3)), 1)
            ty = int(base + rng.uniform(-G * 0.05, G * 0.05))
            for r in range(th):
                yy = ty - r
                if 0 <= yy < G:
                    ww = max(int((r / th) * w * 2.2) + 1, 1)
                    cv[yy, max(0, tx - ww):min(G, tx + ww + 1)] = dark
    mist = int(rng.integers(0, 3))
    cv[int(G * 0.5):] = cv[int(G * 0.5):] * 0.75 + bot * 0.25
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_kota(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    night = key in ("malam", "senja", "mystic")
    x = 0
    while x < G:
        w = int(rng.integers(2, max(3, G // 6)))
        h = int(G * rng.uniform(0.18, 0.55))
        c = hsv(0.62, 0.20, rng.uniform(0.10, 0.22) if night else rng.uniform(0.35, 0.55))
        cv[G - h:, x:x + w] = c
        if night:
            for wy in range(max(G - h + 1, 0), G - 1):
                for wx in range(x, min(x + w, G)):
                    if rng.random() < 0.28:
                        cv[wy, wx] = hsv(0.12, 0.55, rng.uniform(0.75, 0.95))
        x += w + (1 if rng.random() < 0.5 else 0)
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_gurun(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    d1 = hsv(0.11, 0.55, 0.62)
    d2 = hsv(0.09, 0.60, 0.45)
    base = G * rng.uniform(0.55, 0.65)
    y1 = (base + G * 0.14 * sines(rng, G)).astype(int)
    fill_below(cv, np.clip(y1, 0, G - 1), d1)
    y2 = (base + G * 0.28 + G * 0.12 * sines(rng, G)).astype(int)
    fill_below(cv, np.clip(y2, 0, G - 1), d2)
    ctx["ground"] = np.clip(y2, 0, G - 1)
    return ctx

def sc_angkasa(cv, G, rng, sk, orn, ctx):
    cv[:] = vgrad(hsv(0.68, 0.65, 0.07), hsv(0.62, 0.55, 0.13), G)
    stars(cv, rng, 2.5, G, y_max=1.0)
    nc = int(rng.integers(1, 3))
    nebula_c = [hsv(0.85, 0.6, 0.45), hsv(0.50, 0.6, 0.45), hsv(0.62, 0.5, 0.4)]
    for i in range(nc):
        cx, cy, r = rng.uniform(0, G), rng.uniform(0, G), rng.uniform(G * 0.2, G * 0.42)
        col = nebula_c[int(rng.integers(3))]
        soft = 0.35
        yy, xx = np.mgrid[0:G, 0:G]
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        t = np.clip(1 - d2, 0, 1) ** 1.6 * (1 - soft) + soft * np.clip(1 - d2, 0, 1) ** 0.5
        cv[:] = cv * (1 - t[..., None] * 0.55) + col * (t[..., None] * 0.55)
    px, py, pr = rng.uniform(G * 0.25, G * 0.75), rng.uniform(G * 0.2, G * 0.6), rng.uniform(G * 0.10, G * 0.2)
    pc = hsv(rng.uniform(0, 1), 0.55, 0.65)
    blob(cv, px, py, pr, pc)
    yy, xx = np.mgrid[0:G, 0:G]
    sh = ((xx - px) ** 2 + (yy - py) ** 2) / (pr * pr)
    cv[(sh > 0.35) & (sh <= 1.0)] = cv[(sh > 0.35) & (sh <= 1.0)] * 0.72
    if rng.random() < 0.5:
        ring = ((xx - px) ** 2 / (pr * 2.2) ** 2 + (yy - py) ** 2 / (pr * 0.5) ** 2)
        cv[(ring > 0.72) & (ring < 1.05)] = cv[(ring > 0.72) & (ring < 1.05)] * 0.6 + hsv(0.11, 0.35, 0.8) * 0.4
    return ctx

def sc_aurora(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(hsv(0.66, 0.70, 0.08), hsv(0.63, 0.50, 0.18), G)
    stars(cv, rng, 1.0, G)
    yy, xx = np.mgrid[0:G, 0:G]
    for i in range(int(rng.integers(2, 4))):
        ph, fr = rng.uniform(0, 6.28), rng.uniform(1.2, 2.6)
        cx = (xx / G) * fr * 6.28 + ph
        band = np.exp(-((yy / G - (0.18 + 0.22 * np.sin(cx / 2)) ) ** 2) / (2 * 0.045 ** 2))
        col = [hsv(0.40, 0.85, 0.85), hsv(0.48, 0.80, 0.80), hsv(0.78, 0.70, 0.75)][i % 3]
        cv[:] = cv * (1 - (band * 0.75)[..., None]) + col * (band * 0.75)[..., None]
    base = G * rng.uniform(0.72, 0.85)
    ycol = (base + G * 0.10 * sines(rng, G)).astype(int)
    fill_below(cv, np.clip(ycol, 0, G - 1), hsv(0.65, 0.45, 0.10))
    ctx["ground"] = np.clip(ycol, 0, G - 1)
    return ctx

def sc_pantai(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    hz = int(G * rng.uniform(0.30, 0.42))
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    sea = hsv(0.52, 0.65, 0.55)
    cv[hz:hz + int(G * 0.25)] = sea * (1 - 0.15 * (np.arange(int(G * 0.25)) / max(int(G * 0.25), 1))[:, None, None])
    sand = hsv(0.12, 0.45, 0.85)
    sy = hz + int(G * 0.25)
    cv[sy:] = sand
    cv[sy:, :] = sand * (1 - 0.12 * rng.uniform(0, 1))
    ctx["water"] = True
    ctx["water_y"] = hz
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_danau(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    hz = int(G * rng.uniform(0.42, 0.55))
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    base = G * rng.uniform(0.18, 0.30)
    ycol = (hz - base * 0.4 + base * sines(rng, G)).astype(int)
    mtn = hsv(0.60, 0.30, 0.38)
    fill_below(cv, np.clip(ycol, 0, hz), mtn, mix_sky=0.2, sky=bot)
    wl = hsv(0.57, 0.45, 0.35)
    cv[hz:] = wl
    n = G - hz
    refl = cv[max(hz - 1 - n + 1, 0):hz, :][::-1, :]
    if refl.shape[0] < n:
        pad = np.repeat(refl[:1], n - refl.shape[0], axis=0)
        refl = np.concatenate([pad, refl], 0)
    cv[hz:] = cv[hz:] * 0.55 + refl * 0.35
    for k in range(int(G * 0.1) + 2):
        y = hz + int((k / (G * 0.1 + 2)) * (G - hz - 1))
        cv[y, :] = cv[y, :] * 0.8 + hsv(0.55, 0.2, 0.85) * 0.2
    ctx["water"] = True
    ctx["water_y"] = hz
    return ctx

def sc_sawah(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    hz = int(G * rng.uniform(0.18, 0.3))
    cv[:] = vgrad(top, bot, G)
    y = hz
    band = 0
    while y < G:
        h = max(int(rng.integers(2, max(3, G // 7))), 2)
        col = hsv(0.24 + rng.uniform(-0.05, 0.05), 0.60, 0.55 + 0.06 * (band % 3))
        yy = np.arange(y, min(y + h, G))
        curve = (G * 0.02 * np.sin(np.arange(G) / G * rng.uniform(2, 5) * 6.28 + band)).astype(int)
        for i, yyv in enumerate(yy):
            cv[yyv, :] = col
            cv[yyv, :] = cv[yyv, :] * 0.9
            sh = np.clip(curve + y, 0, G - 1)
            if yyv == min(y + h - 1, G - 1):
                cv[yyv, :] = cv[yyv, :] * 0.82
        y += h
        band += 1
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_kanjon(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    y = int(G * rng.uniform(0.25, 0.4))
    band = 0
    while y < G:
        h = max(int(rng.integers(2, max(3, G // 6))), 2)
        col = hsv(0.06 + rng.uniform(-0.02, 0.03), 0.62, 0.60 - band * 0.05)
        yy = np.arange(y, min(y + h, G))
        w1 = sines(rng, G) * G * 0.05
        for yyv in yy:
            cv[yyv, :] = col
        y += h
        band += 1
    riv = int(G * rng.uniform(0.82, 0.92))
    cv[riv:, :] = hsv(0.53, 0.55, 0.60)
    ctx["water"] = True
    ctx["water_y"] = riv
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_volkano(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(hsv(0.66, 0.60, 0.12), hsv(0.05, 0.45, 0.28), G)
    stars(cv, rng, 0.6, G)
    cx = G * rng.uniform(0.35, 0.65)
    half = G * rng.uniform(0.28, 0.42)
    hgt = G * rng.uniform(0.38, 0.55)
    for x in range(G):
        t = abs(x - cx) / half
        if t < 1:
            ytop = int(G - hgt * (1 - t))
            cv[ytop:, x] = hsv(0.03, 0.45, 0.16 + 0.05 * rng.random())
    crater = int(G - hgt)
    cv[crater:crater + 2, max(0, int(cx - half * 0.25)):min(G, int(cx + half * 0.25))] = hsv(0.06, 0.95, 0.95)
    for k in range(int(rng.integers(2, 5))):
        x = int(cx + rng.integers(-2, 3))
        y = int(G - hgt + k * G * 0.12)
        if 0 <= y < G:
            cv[y, max(0, x):min(G, x + 2)] = hsv(0.05, 0.95, rng.uniform(0.7, 1.0))
    sy = int(G - hgt - rng.integers(1, 3))
    blob(cv, cx, sy - 2, G * 0.16, hsv(0.0, 0.0, 0.30), squash=0.8, soft=0.6)
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_bunga(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    gr = hsv(0.30 + rng.uniform(-0.06, 0.06), 0.55, 0.40)
    fy = int(G * rng.uniform(0.45, 0.6))
    cv[fy:] = gr
    n = int(G * rng.uniform(0.8, 1.6))
    fcols = [hsv(0.98, 0.75, 0.95), hsv(0.13, 0.85, 0.95), hsv(0.75, 0.65, 0.9), hsv(0.55, 0.7, 0.95), hsv(0.05, 0.85, 0.9)]
    for _ in range(n):
        x, y = int(rng.integers(0, G)), int(rng.integers(fy, G))
        cv[y, x] = fcols[int(rng.integers(len(fcols)))]
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_terjun(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    cliff = hsv(0.40, 0.35, 0.30)
    lw = int(G * rng.uniform(0.18, 0.3))
    lx = int(rng.integers(1, max(2, G - lw - 1)))
    skyfull = np.broadcast_to(vgrad(top, bot, G), (G, G, 3)).copy()
    cv[:, :] = cliff
    cv[:, :lx] = skyfull[:, :lx]
    cv[:, lx + lw:] = skyfull[:, lx + lw:]
    wf = hsv(0.55, 0.15, 0.90)
    cv[:, lx:lx + lw] = wf
    for x in range(lx, lx + lw):
        for y in range(0, G, 3):
            if rng.random() < 0.5:
                cv[y:y + 2, x] = wf * rng.uniform(0.75, 1.0)
    pool = int(G * 0.78)
    cv[pool:, lx:max(lx + lw, lx + 2)] = hsv(0.52, 0.45, 0.60)
    cv[pool - 1:pool + 1, max(0, lx - 1):min(G, lx + lw + 1)] = hsv(0.55, 0.08, 0.97)
    ctx["water"] = True
    ctx["water_y"] = pool
    ctx["ground"] = np.full(G, G - 1)
    return ctx

def sc_salju(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    cv[:] = vgrad(hsv(0.58, 0.25, 0.72), hsv(0.57, 0.15, 0.92), G)
    base = G * rng.uniform(0.55, 0.7)
    ycol = (base + G * 0.10 * sines(rng, G)).astype(int)
    fill_below(cv, np.clip(ycol, 0, G - 1), hsv(0.58, 0.05, 0.95))
    n = int(rng.integers(2, 6))
    for _ in range(n):
        tx = int(rng.integers(1, G - 1))
        ty = int(np.clip(ycol[tx] - rng.integers(2, 5), 0, G - 1))
        th = int(G * rng.uniform(0.10, 0.2))
        for r in range(th):
            yy = ty - r
            if 0 <= yy < G:
                ww = max(int((r / th) * 1.6) + 1, 1)
                cv[yy, max(0, tx - ww):min(G, tx + ww + 1)] = hsv(0.55, 0.30, 0.22)
    ctx["ground"] = np.clip(ycol, 0, G - 1)
    return ctx

def sc_awan(cv, G, rng, sk, orn, ctx):
    top, bot, key = sk
    if key in ("mimpi", "pagi", "siang"):
        top = top * hsv(0.0, 1.25, 0.78)  # perkuat langit biar awan kontras
    cv[:] = vgrad(top, bot, G)
    stars(cv, rng, 1.0 if key == "malam" else 0, G)
    seay = int(G * rng.uniform(0.55, 0.7))
    for i in range(int(rng.integers(6, 11))):
        cx, cy = rng.uniform(0, G), rng.uniform(seay - G * 0.12, G + G * 0.05)
        base = hsv(0.58, 0.06, 0.97) if key != "malam" else hsv(0.63, 0.22, 0.55)
        shade = base * 0.72
        blob(cv, cx, cy + G * 0.05, rng.uniform(G * 0.10, G * 0.22), shade, squash=0.4)
        blob(cv, cx, cy, rng.uniform(G * 0.10, G * 0.22), base, squash=0.45)
    if rng.random() < 0.6:
        or_matahari(cv, G, rng, ctx, None)
    return ctx

SCENE_FN = {
    "gunung": sc_gunung, "laut": sc_laut, "hutan": sc_hutan, "kota": sc_kota,
    "gurun": sc_gurun, "angkasa": sc_angkasa, "aurora": sc_aurora, "pantai": sc_pantai,
    "danau": sc_danau, "sawah": sc_sawah, "kanjon": sc_kanjon, "volkano": sc_volkano,
    "bunga": sc_bunga, "terjun": sc_terjun, "salju": sc_salju, "awan": sc_awan,
}

# ---------- ornaments ----------

def or_bintang(cv, G, rng, ctx, mood):
    stars(cv, rng, 2.2, G)

def or_bulan(cv, G, rng, ctx, mood):
    x, y, r = int(rng.integers(3, G - 3)), int(rng.integers(1, G // 2)), rng.uniform(1.6, 2.6)
    blob(cv, x, y, r, hsv(0.13, 0.15, 0.95))
    if rng.random() < 0.6:
        blob(cv, x + r * 0.7, y - r * 0.25, r * 0.85, cv[int(np.clip(y - r * 1.2, 0, G - 1)), 0] * 0.9 + hsv(0.66, 0.5, 0.12) * 0.1)

def or_matahari(cv, G, rng, ctx, mood):
    x, y = int(rng.integers(2, G - 2)), int(rng.integers(1, max(2, int(G * 0.5))))
    r = rng.uniform(1.8, 3.2)
    blob(cv, x, y, r + 1.2, hsv(0.10, 0.55, 0.85), soft=0.8)
    blob(cv, x, y, r, hsv(0.12, 0.75, 1.0))

def or_awan(cv, G, rng, ctx, mood):
    cc = hsv(0.58, 0.06, 0.97) if mood not in ("malam",) else hsv(0.63, 0.25, 0.45)
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(0, G), rng.uniform(G * 0.08, G * 0.55)
        for k in range(3):
            blob(cv, cx + k * G * 0.08 - G * 0.08, cy + (k % 2) * G * 0.03, rng.uniform(G * 0.07, G * 0.14), cc, squash=0.55)

def or_burung(cv, G, rng, ctx, mood):
    c = hsv(0.62, 0.4, 0.10)
    for _ in range(int(rng.integers(2, 6))):
        x, y = int(rng.integers(1, G - 3)), int(rng.integers(1, int(G * 0.5)))
        cv[y, x] = c
        if x + 2 < G:
            cv[max(0, y - 1), x + 1] = c
            cv[y, x + 2] = c

def or_perahu(cv, G, rng, ctx, mood):
    if not ctx.get("water"):
        return
    wy = ctx["water_y"]
    x = int(rng.integers(1, max(2, G - 5)))
    y = int(min(wy + rng.integers(1, 3), G - 2))
    cv[y, x:x + 3] = hsv(0.05, 0.5, 0.15)
    if y - 1 >= 0:
        cv[y - 1, x + 1:min(G, x + 3)] = hsv(0.10, 0.05, 0.95)
    if y - 2 >= 0:
        cv[y - 2, x + 1] = hsv(0.10, 0.05, 0.95)

def or_balon(cv, G, rng, ctx, mood):
    x, y = int(rng.integers(2, G - 2)), int(rng.integers(2, max(3, int(G * 0.45))))
    r = rng.uniform(1.5, 2.5)
    blob(cv, x, y, r, [hsv(0.0, 0.8, 0.9), hsv(0.12, 0.85, 0.95), hsv(0.55, 0.8, 0.9)][int(rng.integers(3))])
    if 0 <= y + int(r) + 1 < G:
        cv[y + int(r) + 1, x] = hsv(0.08, 0.6, 0.3)

def or_kupu(cv, G, rng, ctx, mood):
    if ctx.get("scene") in ("angkasa", "awan"):
        return
    x, y = int(rng.integers(1, G - 2)), int(rng.integers(G // 3, G - 1))
    c1, c2 = hsv(rng.uniform(0, 1), 0.8, 0.95), hsv(rng.uniform(0, 1), 0.8, 0.85)
    cv[y, x] = c1
    cv[y, x + 1] = c2
    if rng.random() < 0.6 and y + 1 < G:
        cv[y + 1, x] = c2
        cv[y + 1, x + 1] = c1

def or_bunga(cv, G, rng, ctx, mood):
    if ctx.get("scene") in ("angkasa", "awan", "kota"):
        return
    gy = ctx.get("ground")
    for _ in range(int(rng.integers(3, 8))):
        x = int(rng.integers(0, G))
        y = int(rng.integers(G // 2, G)) if gy is None else int(np.clip(gy[x] + rng.integers(0, max(1, G - gy[x])), 0, G - 1))
        cv[y, x] = [hsv(0.98, 0.75, 0.95), hsv(0.13, 0.85, 0.95), hsv(0.75, 0.65, 0.9)][int(rng.integers(3))]

def or_petir(cv, G, rng, ctx, mood):
    x = int(rng.integers(1, G - 1))
    y = 0
    while y < G - 1:
        cv[y, x] = hsv(0.13, 0.10, 1.0)
        cv[y, min(G - 1, x + 1)] = hsv(0.60, 0.15, 0.95)
        x = int(np.clip(x + rng.integers(-1, 2), 0, G - 1))
        y += rng.integers(1, 3)
    cv[:] = cv * 0.88 + hsv(0.62, 0.1, 0.9) * 0.12

def or_pelangi(cv, G, rng, ctx, mood):
    if ctx.get("scene") == "angkasa":
        return
    yy, xx = np.mgrid[0:G, 0:G]
    cx, cy = G / 2, G * rng.uniform(1.1, 1.6)
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    cols = [hsv(0.00, 0.7, 0.9), hsv(0.09, 0.8, 0.95), hsv(0.14, 0.8, 0.95), hsv(0.35, 0.7, 0.85),
            hsv(0.52, 0.7, 0.9), hsv(0.65, 0.65, 0.9), hsv(0.78, 0.6, 0.9)]
    R = G * rng.uniform(0.55, 0.8)
    for i, c in enumerate(cols):
        rr = R - i * 0.9
        m = (rad <= rr) & (rad > rr - 0.9) & (yy <= cy)
        cv[m] = c

def or_meteor(cv, G, rng, ctx, mood):
    x, y = int(rng.integers(2, G - 3)), int(rng.integers(0, max(1, G // 2)))
    ln = int(rng.integers(3, 6))
    for k in range(ln):
        xx, yy = x - k, y + k
        if 0 <= xx < G and 0 <= yy < G:
            cv[yy, xx] = hsv(0.52, 0.30, 1.0) if k < 2 else hsv(0.55, 0.2, 0.8)

def or_salju(cv, G, rng, ctx, mood):
    for _ in range(int(rng.integers(5, 16))):
        x, y = int(rng.integers(0, G)), int(rng.integers(0, G))
        cv[y, x] = hsv(0.58, 0.03, 1.0)

def or_pohon(cv, G, rng, ctx, mood):
    if ctx.get("scene") in ("angkasa", "awan", "laut", "kanjon"):
        return
    gy = ctx.get("ground")
    for _ in range(int(rng.integers(1, 4))):
        tx = int(rng.integers(1, G - 1))
        base = int(gy[tx]) if gy is not None else G - 1
        th = int(rng.integers(3, max(4, G // 3)))
        c = hsv(0.36, 0.55, 0.14)
        for r in range(th):
            yy = base - r
            if 0 <= yy < G:
                ww = max(int((r / th) * 1.8) + 1, 1)
                cv[yy, max(0, tx - ww):min(G, tx + ww + 1)] = c

ORN_FN = {k: v for k, v in list(globals().items()) if k.startswith("or_")}
ORN_FN = {k[3:]: v for k, v in ORN_FN.items()}

# ---------- color grading + quantize ----------

def rgb_to_hsv_arr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, -1); mn = np.min(rgb, -1); d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    s = np.where(mx > 1e-6, d / (mx + 1e-9), 0)
    return h, s, mx

def hsv_to_rgb_arr(h, s, v):
    i = np.floor(h * 6); f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    i = i.astype(int) % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)

def grade(cv, color_tag):
    for c in TG.COLORS:
        if c[0] == color_tag:
            dh, sm, vm, vb = c[2], c[3], c[4], c[5]
            break
    h, s, v = rgb_to_hsv_arr(np.clip(cv, 0, 1))
    h = (h + dh) % 1.0
    s = np.clip(s * sm, 0, 1)
    v = np.clip(v * vm + vb, 0, 1)
    return hsv_to_rgb_arr(h, s, v)

def quantize(cv):
    pal = PA.PAL_ARR
    if pal is None:
        PA.PAL_ARR = pal = np.array(PA.PALETTE, dtype=np.float32) / 255.0
    d = ((cv[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8)

def render(cond, G, rng):
    """cond dict -> (G,G) uint8 palette indices"""
    cv = np.zeros((G, G, 3), dtype=float)
    sk = sky_for(cond.get("mood"), rng)
    ctx = {"scene": cond["scene"], "water": False}
    SCENE_FN[cond["scene"]](cv, G, rng, sk, cond.get("orn", []), ctx)
    for o in cond.get("orn", []):
        if o in ORN_FN:
            ORN_FN[o](cv, G, rng, ctx, cond.get("mood"))
    mood = cond.get("mood")
    spec = next((m[2] for m in TG.MOODS if m[0] == mood), None)
    cv = grade(cv, cond["color"])
    if spec:
        fogc = hsv(0.58, 0.05, 0.92) if mood == "kabut" else sk[1]
        if spec["fog"] > 0:
            t = np.linspace(0, 1, G)[:, None]
            a = spec["fog"] * (0.35 + 0.65 * t)
            cv = cv * (1 - a[..., None]) + fogc * a[..., None]
        if spec["glow"] > 0:
            cv = np.clip(cv + spec["glow"], 0, 1)
    return quantize(np.clip(cv, 0, 1))

def render_batch(conds, G, seed0):
    """-> (B, G*G) uint8 + rng per sampel deterministik"""
    out = np.zeros((len(conds), G * G), dtype=np.uint8)
    for i, c in enumerate(conds):
        rng = np.random.default_rng(seed0 + i)
        out[i] = render(c, G, rng).reshape(-1)
    return out
