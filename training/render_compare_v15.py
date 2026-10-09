"""Render perbandingan dark12 vs dark15 dari e2e_v15_out.json (grid 16x16, seed sama).
Output: preview/compare_dark_1.2_vs_1.5.png
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
import palette as PA

PAL = np.array(PA.PALETTE, dtype=np.uint8)

with open(os.path.join(BASE, "e2e_v15_out.json")) as f:
    data = json.load(f)

by = {}
for d in data:
    by.setdefault(d["model"], []).append(d)

old, new, title = "dark12", "dark15", "Dark 1.2 vs Dark 1.5 (parameter naik 2.7M \u2192 4.9M)"
olds, news = by[old], by[new]
n = len(olds)
scale = 5
cell = 16 * 2 * scale  # 160 px
pad, label_h, head_h = 10, 16, 30
W = pad * 3 + cell * 2
H = head_h + n * (cell + label_h + pad) + pad
img = Image.new("RGB", (W, H), (12, 12, 14))
dr = ImageDraw.Draw(img)

def draw_grid(d, x, y):
    G = d["G"]
    toks = np.array(d["tokens"], dtype=np.int64)
    g = toks.reshape(G, G)
    tile = PAL[g]  # G,G,3
    im = Image.fromarray(tile, "RGB").resize((cell, cell), Image.NEAREST)
    img.paste(im, (x, y))

dr.text((pad, 8), title, fill=(240, 240, 240))
y = head_h
for i in range(n):
    draw_grid(olds[i], pad, y)
    draw_grid(news[i], pad * 2 + cell, y)
    dr.text((pad, y + cell + 2), f"{old} \u2014 {olds[i]['prompt']}", fill=(160, 200, 255))
    dr.text((pad * 2 + cell, y + cell + 2), f"{new} \u2014 {news[i]['prompt']}", fill=(255, 200, 160))
    y += cell + label_h + pad

out = os.path.join(ROOT, "preview", "compare_dark_1.2_vs_1.5.png")
img.save(out)
print("OK ->", out)
