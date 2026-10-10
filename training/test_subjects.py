"""Test visual renderer subjek — grid PNG: tiap baris = subjek, tiap kolom = variasi kondisi."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
import render as RD

CASES = [
    # (scene, color, mood, orn, subj{id, attrs, col})
    ("volkano", "hangat", "senja", [], {"id": "ikan", "attrs": ["terbang"], "col": None}),
    ("volkano", "hangat", "malam", [], {"id": "naga", "attrs": ["api", "raksasa"], "col": None}),
    ("kota", "neon", "malam", ["petir"], {"id": "kucing", "attrs": ["neon"], "col": None}),
    ("pantai", "cerah", "siang", ["perahu"], {"id": "burung", "attrs": [], "col": "emas"}),
    ("salju", "es", "pagi", [], {"id": "robot", "attrs": ["es", "raksasa"], "col": None}),
    ("laut", "dingin", "senja", [], {"id": "ikan", "attrs": [], "col": "oranye"}),
    ("hutan", "smaragd", "kabut", [], {"id": "rumah", "attrs": ["kecil"], "col": None}),
    ("angkasa", "gelap", "malam", ["meteor"], {"id": "gajah", "attrs": ["terbang"], "col": "ungu"}),
    ("gurun", "vintage", "siang", [], {"id": "kapal", "attrs": [], "col": None}),
    ("bunga", "pastel", "pagi", ["kupu"], {"id": "kupu", "attrs": ["raksasa"], "col": None}),
    ("gunung", "cerah", "pagi", ["awan"], {"id": "balon", "attrs": [], "col": None}),
    ("danau", "dingin", "mimpi", ["bulan"], {"id": "pohon", "attrs": [], "col": None}),
    ("kanjon", "bumi", "siang", [], {"id": "gajah", "attrs": [], "col": None}),
    ("sawah", "tropis", "pagi", [], {"id": "rumah", "attrs": [], "col": "biru"}),
    ("terjun", "tropis", "siang", ["pelangi"], {"id": "naga", "attrs": ["terbang"], "col": "hijau"}),
    ("awan", "pastel", "pagi", ["matahari"], {"id": "kucing", "attrs": ["terbang"], "col": "putih"}),
    ("laut", "es", "malam", ["bintang"], {"id": "kapal", "attrs": ["es"], "col": None}),
    ("kota", "monokrom", "badai", ["petir"], {"id": "robot", "attrs": ["raksasa", "neon"], "col": None}),
    ("aurora", "dingin", "malam", [], {"id": "burung", "attrs": ["raksasa"], "col": None}),
    ("volkano", "gelap", "mystic", ["petir"], {"id": "kucing", "attrs": ["api", "kecil"], "col": "hitam"}),
]

def main():
    G, UP, COLS = 32, 12, 5
    rows = (len(CASES) + COLS - 1) // COLS
    cell = G * UP
    W, H = COLS * cell, rows * (cell + 22)
    img = Image.new("RGB", (W, H), (12, 13, 17))
    dr = ImageDraw.Draw(img)
    pal = [(int(PA_HEX[i][1:3], 16), int(PA_HEX[i][3:5], 16), int(PA_HEX[i][5:7], 16))
           for i in range(len(PA_HEX))]
    for i, (scene, color, mood, orn, subj) in enumerate(CASES):
        rng = np.random.default_rng(1000 + i * 77)
        cond = {"scene": scene, "color": color, "mood": mood, "orn": orn, "subj": subj}
        grid = RD.render(cond, G, rng)
        tile = Image.new("RGB", (G, G))
        tile.putdata([pal[int(v)] for v in grid.reshape(-1)])
        tile = tile.resize((cell, cell), Image.NEAREST)
        r, c = divmod(i, COLS)
        img.paste(tile, (c * cell, r * (cell + 22)))
        desc = f"{subj['id']}{'+' + ','.join(subj['attrs']) if subj['attrs'] else ''}"
        dr.text((c * cell + 4, r * (cell + 22) + cell + 4),
                f"{desc} @ {scene}/{mood}", fill=(220, 220, 220))
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state", "preview_subjects.png")
    img.save(out)
    print("OK ->", out, img.size)

if __name__ == "__main__":
    from palette import palette_hex
    PA_HEX = palette_hex()
    main()
