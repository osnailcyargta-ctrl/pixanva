"""Render e2e_imajin_out.json → PNG grid + caption (buat eyeball kualitas stage)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw
from palette import palette_hex

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "e2e_imajin_out.json")
    with open(src) as f:
        data = json.load(f)
    G = data["G"]
    pal = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in palette_hex()]
    UP, COLS = 10, 3
    cell = G * UP
    rows = (len(data["cases"]) + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * cell, rows * (cell + 20)), (12, 13, 17))
    dr = ImageDraw.Draw(img)
    for i, case in enumerate(data["cases"]):
        tile = Image.new("RGB", (G, G))
        tile.putdata([pal[int(v) % 96] for v in case["tokens"]])
        tile = tile.resize((cell, cell), Image.NEAREST)
        r, c = divmod(i, COLS)
        img.paste(tile, (c * cell, r * (cell + 20)))
        dr.text((c * cell + 4, r * (cell + 20) + cell + 3), case["text"][:44], fill=(230, 230, 230))
    out = src.replace(".json", ".png")
    img.save(out)
    print("OK ->", out, img.size)


if __name__ == "__main__":
    main()
