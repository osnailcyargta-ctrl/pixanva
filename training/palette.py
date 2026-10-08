"""Pixanva palette — 96 warna fixed, deterministik, dipakai Python (training) & JS (render).
Index 0 SENGAJA warna gelap kebiruan, BUKAN putih — anti bug "blank putih".
"""
import colorsys


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))


def build_palette():
    P = []
    # 0-13: grayscale ramp (gelap -> terang), sedikit kebiruan biar cinematic
    for i in range(14):
        t = i / 13.0
        v = 0.028 + t * 0.955
        P.append(_hsv(0.62, 0.10 * (1 - t) * 0.6, v))
    # 14-19: navy / indigo malam
    for i, v in enumerate((0.10, 0.16, 0.23, 0.31, 0.42, 0.55)):
        P.append(_hsv(0.655, 0.62 - i * 0.05, v))
    # 20-25: biru siang
    for i, v in enumerate((0.42, 0.52, 0.62, 0.72, 0.84, 0.96)):
        P.append(_hsv(0.585, 0.75 - i * 0.07, v))
    # 26-30: cyan
    for i, v in enumerate((0.30, 0.44, 0.58, 0.74, 0.92)):
        P.append(_hsv(0.53, 0.80 - i * 0.10, v))
    # 31-35: teal
    for i, v in enumerate((0.22, 0.34, 0.48, 0.64, 0.82)):
        P.append(_hsv(0.465, 0.72 - i * 0.08, v))
    # 36-41: hijau hutan
    for i, v in enumerate((0.12, 0.20, 0.30, 0.42, 0.56, 0.72)):
        P.append(_hsv(0.375, 0.68 - i * 0.06, v))
    # 42-46: lime / sawah
    for i, v in enumerate((0.28, 0.42, 0.56, 0.72, 0.88)):
        P.append(_hsv(0.245, 0.72 - i * 0.08, v))
    # 47-52: gold / sand
    for i, v in enumerate((0.50, 0.60, 0.70, 0.80, 0.90, 0.98)):
        P.append(_hsv(0.125, 0.62 - i * 0.08, v))
    # 53-58: orange (senja, lava)
    for i, v in enumerate((0.35, 0.48, 0.60, 0.72, 0.85, 0.97)):
        P.append(_hsv(0.072, 0.85 - i * 0.09, v))
    # 59-63: red / ember
    for i, v in enumerate((0.22, 0.35, 0.50, 0.66, 0.84)):
        P.append(_hsv(0.015, 0.82 - i * 0.08, v))
    # 64-69: magenta / pink
    for i, v in enumerate((0.25, 0.38, 0.52, 0.66, 0.80, 0.94)):
        P.append(_hsv(0.905, 0.72 - i * 0.08, v))
    # 70-75: purple (aurora, senja tua)
    for i, v in enumerate((0.16, 0.26, 0.38, 0.52, 0.68, 0.86)):
        P.append(_hsv(0.765, 0.66 - i * 0.07, v))
    # 76-80: brown / earth
    for i, v in enumerate((0.16, 0.23, 0.31, 0.40, 0.50)):
        P.append(_hsv(0.075, 0.55, v))
    # 81-84: pastel
    P.append(_hsv(0.95, 0.35, 0.92))   # pastel pink
    P.append(_hsv(0.72, 0.28, 0.93))   # lavender
    P.append(_hsv(0.42, 0.25, 0.92))   # mint
    P.append(_hsv(0.13, 0.30, 0.95))   # butter
    # 85-87: snow (putih dingin)
    P.append(_hsv(0.58, 0.10, 0.82))
    P.append(_hsv(0.58, 0.06, 0.92))
    P.append(_hsv(0.58, 0.03, 0.985))
    # 88-91: neon
    P.append(_hsv(0.40, 0.85, 1.00))   # neon green
    P.append(_hsv(0.52, 0.90, 1.00))   # neon cyan
    P.append(_hsv(0.87, 0.95, 1.00))   # hot magenta
    P.append(_hsv(0.71, 0.80, 1.00))   # electric violet
    # 92-95: extra
    P.append(_hsv(0.99, 0.70, 0.28))   # maroon tua
    P.append(_hsv(0.19, 0.55, 0.38))   # olive
    P.append(_hsv(0.60, 0.22, 0.44))   # slate biru-abu
    P.append(_hsv(0.11, 0.28, 0.96))   # cream
    assert len(P) == 96, len(P)
    return P


PALETTE = build_palette()
N_COLORS = 96
PAL_ARR = None  # lazy numpy (V,3) uint8, diisi oleh imgutil


def palette_hex():
    return ['#%02x%02x%02x' % c for c in PALETTE]
