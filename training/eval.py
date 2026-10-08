"""Sampling evaluasi dari checkpoint — contact sheet prompt tetap, dgn CFG.
Dipake buat validasi visual sebelum push."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw
import palette as PA
import tags as TG
import model as M
import optim as OP
from train import get_cfg

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")


def sample(P, cfg, cond, G, seed, temp=0.85, topk=24, guidance=1.8, rng=None):
    """Autoregressive sampling + CFG. Return (G,G) idx."""
    rope_cache = {}
    toks, _ = TG.cond_tokens(cond, G)
    uToks, _ = TG.cond_tokens({"uncond": True}, G)
    useCFG = guidance > 1.001
    nC, nU = len(toks), len(uToks)
    T = nC + G * G

    def rope_for(ncond, Tcur):
        key = ncond
        if key not in rope_cache:
            cax, sax, cay, say, bx = M.rope_tables(ncond, G, cfg)
            rope_cache[key] = (cax, sax, cay, say, bx)
        cax, sax, cay, say, bx = rope_cache[key]
        return (cax[:Tcur], sax[:Tcur], cay[:Tcur], say[:Tcur], bx)

    rng = rng or np.random.default_rng(seed)

    def prefill(tl, ncond):
        ids = np.array([tl], dtype=np.int64)
        logits = M.forward(P, cfg, ids, rope_for(ncond, len(tl)))
        return logits[0, -1]

    logitsC = prefill(toks, nC)
    logitsU = prefill(uToks, nU) if useCFG else None
    out = np.zeros(G * G, dtype=np.int64)
    recent = []

    for k in range(G * G):
        if useCFG:
            logits = logitsU + guidance * (logitsC - logitsU)
        else:
            logits = logitsC
        z = logits / temp
        # repetition penalty ringan
        for t in set(recent):
            z[t] -= 0.12 * (1 + recent.count(t))
        top = np.argpartition(-z, topk)[:topk]
        e = np.exp(z[top] - z[top].max())
        p = e / e.sum()
        tok = int(top[rng.choice(len(p), p=p)])
        out[k] = tok - TG.COLOR_OFFSET
        recent.append(tok)
        if len(recent) > 12:
            recent.pop(0)
        if k < G * G - 1:
            ids = np.concatenate([toks, out[:k + 1] + TG.COLOR_OFFSET])[None]
            logitsC = M.forward(P, cfg, ids, rope_for(nC, ids.shape[1]))[0, -1]
            if useCFG:
                idsU = np.concatenate([uToks, out[:k + 1] + TG.COLOR_OFFSET])[None]
                logitsU = M.forward(P, cfg, idsU, rope_for(nU, idsU.shape[1]))[0, -1]
    return out.reshape(G, G)


def up2(grid):
    pal = np.array(PA.PALETTE, dtype=np.uint8)
    return np.repeat(np.repeat(pal[grid], 2, 0), 2, 1)


PROMPTS = [
    {"scene": "gunung", "color": "hangat", "mood": "senja", "orn": ["burung"]},
    {"scene": "gunung", "color": "es", "mood": "pagi", "orn": []},
    {"scene": "laut", "color": "dingin", "mood": "senja", "orn": ["perahu", "bulan"]},
    {"scene": "kota", "color": "neon", "mood": "malam", "orn": ["hujan" if False else "awan"]},
    {"scene": "hutan", "color": "smaragd", "mood": "kabut", "orn": []},
    {"scene": "angkasa", "color": "gelap", "mood": None, "orn": ["bintang", "meteor"]},
    {"scene": "aurora", "color": "dingin", "mood": "malam", "orn": []},
    {"scene": "gurun", "color": "hangat", "mood": "siang", "orn": ["matahari"]},
    {"scene": "pantai", "color": "tropis", "mood": "siang", "orn": ["pelangi"]},
    {"scene": "volkano", "color": "gelap", "mood": "malam", "orn": ["petir"]},
    {"scene": "bunga", "color": "pastel", "mood": "pagi", "orn": ["kupu"]},
    {"scene": "salju", "color": "dingin", "mood": "mimpi", "orn": ["salju"]},
]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "light"
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    guidance = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
    cfg = get_cfg(name)
    with open(os.path.join(ROOT, "state", f"{name}.pkl"), "rb") as f:
        st = __import__("pickle").load(f)
    P = st["P"]
    print(f"eval {name} step {st['step']} G={G} cfg={guidance}", flush=True)
    cells, labels = [], []
    t0 = time.time()
    for i, c in enumerate(PROMPTS):
        g = sample(P, cfg, c, G, seed=1000 + i, guidance=guidance)
        cells.append(g)
        labels.append(f"{c['scene']}|{c['color']}|{c['mood'] or '-'}")
        print(f"  {i + 1}/{len(PROMPTS)} {labels[-1]} ({time.time() - t0:.0f}s)", flush=True)
    # sheet
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    cell = G * 2 * 3
    im = Image.new("RGB", (cols * cell, rows * (cell + 16)), (14, 16, 20))
    dr = ImageDraw.Draw(im)
    for i, g in enumerate(cells):
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        im.paste(Image.fromarray(up2(g)).resize((cell, cell), Image.NEAREST), (x, y))
        dr.text((x + 4, y + cell + 2), labels[i], fill=(210, 214, 224))
    out = os.path.join(ROOT, "preview", f"eval_{name}_step{st['step']}_G{G}.png")
    im.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
