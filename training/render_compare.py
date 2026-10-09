"""Render perbandingan model lama vs baru dari e2e_qw_out.json (grid 16x16, seed sama).
Output: preview/compare_dark_1.0_vs_1.2.png & preview/compare_heavy_QQ_vs_QW.png
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
import palette as PA

PAL = np.array(PA.PALETTE, dtype=np.uint8)

with open(os.path.join(BASE, "e2e_qw_out.json")) as f:
    data = json.load(f)

by = {}
for d in data:
    by.setdefault(d["model"], []).append(d)

PAIRS = [("dark", "dark12", "Dark 1.0 vs Dark 1.2"),
         ("heavy", "heavyqw", "Heavy QQ vs Heavy QW")]

for old, new, title in PAIRS:
    olds, news = by[old], by[new]
    n = len(olds)
    scale = 5  # 16*2*5 = 160 px per sel
    cell = 16 * 2 * scale
    pad, label_h, head_h = 10, 16, 30
    W = pad * 3 + cell * 2
    H = head_h + n * (cell + label_h + pad) + pad
    im = Image.new("RGB", (W, H), (14, 16, 20))
    dr = ImageDraw.Draw(im)
    dr.text((pad, 8), f"Pixanva {title} — prompt & seed identik (kiri: lama, kanan: baru)",
            fill=(160, 200, 255))
    for i in range(n):
        y = head_h + i * (cell + label_h + pad)
        for j, (dd, name) in enumerate([(olds[i], old), (news[i], new)]):
            toks = np.array(dd["tokens"], dtype=np.int64).reshape(16, 16)
            img = np.repeat(np.repeat(PAL[toks], 2, 0), 2, 1)
            x = pad + j * (cell + pad)
            im.paste(Image.fromarray(img).resize((cell, cell), Image.NEAREST), (x, y))
            uniq = len(set(dd["tokens"]))
            dr.text((x, y + cell + 2),
                    f"{name} | {dd['prompt']} | {uniq} warna",
                    fill=(210, 214, 224))
    out = os.path.join(ROOT, "preview", f"compare_{old}_vs_{new}.png")
    im.save(out)
    print("saved", out)
