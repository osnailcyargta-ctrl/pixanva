"""Rebuild training checkpoint (.pkl) dari model web (.bin fp16).

Dipakai buat warm-start rilis baru (mis. dark 1.2 lanjutan dark 1.0) dari
bobot yang udah di-export ke repo — sandbox ilang tapi bobot aman di git.

Adam moment (m/v) mulai dari nol — butuh ~10-20 step buat konvergen lagi,
gak masalah karena warm-restart cosine emang naikin LR lagi.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
from train import get_cfg

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")


def bin_to_pkl(name_bin, name_out):
    """models/{name_bin}.bin -> state/{name_out}.pkl (P fp32, step & val dari meta bin)."""
    cfg = get_cfg(name_bin)
    path = os.path.join(ROOT, "models", f"{name_bin}.bin")
    with open(path, "rb") as f:
        (mlen,) = np.frombuffer(f.read(4), dtype="<u4")
        meta = json.loads(f.read(mlen).decode())
        blob = f.read()
    assert meta["cfg"] == cfg, f"cfg mismatch: {meta['cfg']} vs {cfg}"

    order = list(M.param_shapes(cfg).keys())
    assert order == meta["order"], "urutan param beda!"
    P = {}
    off = 0
    for k in order:
        shape = M.param_shapes(cfg)[k]
        n = int(np.prod(shape))
        arr = np.frombuffer(blob[off * 2:(off + n) * 2], dtype=np.float16)
        P[k] = arr.astype(np.float32).reshape(shape)
        off += n
    print(f"[{name_bin}] {off:,} params dibaca, step {meta['step']}, val {meta.get('val_loss')}")

    st = {
        "P": P,
        "opt": {"m": {k: np.zeros_like(v) for k, v in P.items()},
                "v": {k: np.zeros_like(v) for k, v in P.items()},
                "t": 0},
        "step": meta["step"],
        "data_rng": np.random.default_rng(4242),
        "history": {"loss": [], "val": [(meta["step"], meta.get("val_loss"))]},
    }
    out = os.path.join(ROOT, "state", f"{name_out}.pkl")
    with open(out, "wb") as f:
        pickle.dump(st, f, protocol=4)
    print(f"  -> {out}")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "state"), exist_ok=True)
    bin_to_pkl("dark", "dark12")
    bin_to_pkl("heavy", "heavyqw")
