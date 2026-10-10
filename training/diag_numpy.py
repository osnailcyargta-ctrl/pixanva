"""Diagnostik sampling imajin — numpy murni, tanpa CFG, beberapa strategi."""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
import model as M
import train_imajin as T
import imajin_data as ID
import tags as TG
from palette import palette_hex

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="5m")
ap.add_argument("--G", type=int, default=16)
args = ap.parse_args()

ITOS = T.load_vocab()
STOI = {w: i for i, w in enumerate(ITOS) if w}
arch = T.ARCHS[args.size]
CFG = {"V": len(ITOS), "d": arch["d"], "L": arch["L"], "H": arch["H"], "mlp": arch["mlp"], "theta": 1000.0}
with open(os.path.join(T.ROOT, "state", f"imajin{args.size}.pkl"), "rb") as f:
    st = pickle.load(f)
P = st["P"]

PROMPTS = ["ikan terbang di volkano", "kucing neon di kota malam ada petir", "pantai senja ada perahu"]
pal = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in palette_hex()]
G = args.G
UP, COLS = 10, len(PROMPTS)
cell = G * UP
img = Image.new("RGB", (COLS * cell, 2 * (cell + 18)), (12, 13, 17))
dr = ImageDraw.Draw(img)

for col, text in enumerate(PROMPTS):
    prefix = ID.encode(text, STOI, G)
    rope_full = M.rope_tables(len(prefix), G, CFG)

    def rope_L(L):
        return tuple((r[:L] if isinstance(r, np.ndarray) else r) for r in rope_full)

    for row, (mode, temp, topk) in enumerate([("argmax", 0.0, 1), ("t.7k16", 0.7, 16)]):
        rng = np.random.default_rng(99)
        toks = list(prefix)
        out = []
        recent = []
        # prefill: full forward, ambil logits posisi terakhir
        ids = np.array([toks], dtype=np.int64)
        logits = M.forward(P, CFG, ids, rope_L(len(toks)))[0, -1]
        for k in range(G * G):
            if mode == "argmax":
                tk = int(np.argmax(logits))
            else:
                z = logits - logits.max()
                p = np.exp(z)
                cand = np.argsort(p)[::-1][:topk]
                pc = p[cand] / p[cand].sum()
                tk = int(rng.choice(cand, p=pc))
            out.append(tk - TG.COLOR_OFFSET)
            recent.append(tk)
            if len(recent) > 12:
                recent.pop(0)
            toks.append(tk)
            if k < G * G - 1:
                ids = np.array([toks], dtype=np.int64)
                logits = M.forward(P, CFG, ids, rope_L(len(toks)))[0, -1]
        uniq = len(set(out))
        print(f"{text!r} [{mode}] uniq={uniq}")
        tile = Image.new("RGB", (G, G))
        tile.putdata([pal[max(0, min(95, int(v)))] for v in out])
        tile = tile.resize((cell, cell), Image.NEAREST)
        img.paste(tile, (col * cell, row * (cell + 18)))
        dr.text((col * cell + 3, row * (cell + 18) + cell + 2), f"{text[:18]}|{mode}", fill=(220, 220, 220))

out_path = os.path.join(os.path.dirname(__file__), f"diag_numpy_{args.size}.png")
img.save(out_path)
print("OK ->", out_path)
