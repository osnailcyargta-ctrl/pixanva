#!/usr/bin/env python3
"""Rebuild state/imajin12m.pkl dari models/imajin12m.bin (rollback survivor).
Adam momen mulai nol — warm-restart cosine, butuh ~10-20 step buat konvergen lagi."""
import sys, os, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
import numpy as np
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(ROOT, "models", "imajin12m.bin")
with open(path, "rb") as f:
    (mlen,) = np.frombuffer(f.read(4), dtype="<u4")
    meta = json.loads(f.read(mlen).decode())
    blob = f.read()

cfg = meta["cfg"]
order = list(M.param_shapes(cfg).keys())
assert order == meta["order"], "urutan param beda!"
P, off = {}, 0
for k in order:
    shape = M.param_shapes(cfg)[k]
    n = int(np.prod(shape))
    P[k] = np.frombuffer(blob[off*2:(off+n)*2], dtype=np.float16).astype(np.float32).reshape(shape)
    off += n
print(f"imajin12m: {off:,} params, step {meta['step']}, val {meta.get('val_loss')}")

st = {
    "P": P,
    "opt": {"m": {k: np.zeros_like(v) for k, v in P.items()},
            "v": {k: np.zeros_like(v) for k, v in P.items()},
            "t": 0},
    "step": meta["step"],
    "data_rng": np.random.default_rng(4242),
    "history": {"loss": [], "val": [(meta["step"], meta.get("val_loss"))]},
}
out = os.path.join(ROOT, "state", "imajin12m.pkl")
with open(out, "wb") as f:
    pickle.dump(st, f, protocol=4)
print(f"  -> {out}")
