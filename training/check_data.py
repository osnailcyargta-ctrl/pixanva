"""Validasi DATA: render contact sheet semua scene x beberapa warna, sebelum training.
PENTING: dipake buat mastiin tag->gambar aligned & gak ada blank.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
import palette as PA
import tags as TG
import render as RD


def up2(grid):  # (G,G) idx -> (2G,2G,3) uint8
    G = grid.shape[0]
    pal = np.array(PA.PALETTE, dtype=np.uint8)
    img = pal[grid]
    return np.repeat(np.repeat(img, 2, 0), 2, 1)


def sheet(path, cells, cols=8, cell=128, labels=None):
    rows = (len(cells) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell, rows * (cell + 14)), (16, 18, 22))
    dr = ImageDraw.Draw(im)
    for i, g in enumerate(cells):
        x, y = (i % cols) * cell, (i // cols) * (cell + 14)
        im.paste(Image.fromarray(up2(g)).resize((cell, cell), Image.NEAREST), (x, y))
        if labels:
            dr.text((x + 4, y + cell + 1), labels[i], fill=(220, 224, 230))
    im.save(path)
    print("saved", path)


if __name__ == "__main__":
    rng = np.random.default_rng(777)
    cells, labels = [], []
    # 1) semua scene, warna default, mood beda-beda
    for i, (sid, label, _) in enumerate(TG.SCENES):
        c = {"scene": sid, "color": TG.COLORS[i % 12][0], "mood": TG.MOODS[i % 8][0],
             "orn": [TG.ORNAMENTS[(i * 3) % 14][0]] if i % 2 else []}
        cells.append(RD.render(c, 32, rng))
        labels.append(f"{label}|{c['color']}|{c['mood']}")
    sheet(os.path.join(os.path.dirname(__file__), "..", "preview", "data_scenes.png"), cells, cols=4, labels=labels)
    # 2) variasi warna untuk 1 scene
    cells, labels = [], []
    for ctag in TG.COLORS:
        for rep in range(2):
            c = {"scene": "gunung", "color": ctag[0], "mood": "senja", "orn": []}
            cells.append(RD.render(c, 24, rng))
            labels.append(ctag[0])
    sheet(os.path.join(os.path.dirname(__file__), "..", "preview", "data_colors.png"), cells, cols=6, labels=labels)
    # 3) variasi seed utk prompt SAMA (harus beda-beda)
    cells, labels = [], []
    for k in range(12):
        c = {"scene": "laut", "color": "hangat", "mood": "senja", "orn": ["perahu"]}
        cells.append(RD.render(c, 24, np.random.default_rng(1000 + k)))
        labels.append(f"seed{k}")
    sheet(os.path.join(os.path.dirname(__file__), "..", "preview", "data_seeds.png"), cells, cols=6, labels=labels)
