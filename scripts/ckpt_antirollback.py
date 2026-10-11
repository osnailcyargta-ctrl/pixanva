#!/usr/bin/env python3
"""Dump/restore checkpoint training jadi npz fp16 kecil (buat di-push ke branch ckpt).
Usage:
  ckpt_antirollback.py dump  <tag>   # state/<tag>.pkl -> state/ckpt_<tag>.npz
  ckpt_antirollback.py restore <tag> # state/ckpt_<tag>.npz -> state/<tag>.pkl (momen nol)
"""
import sys, os, pickle, json
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "pixanva")
ROOT = os.path.abspath(ROOT)

def main():
    mode, tag = sys.argv[1], sys.argv[2]
    pkl = os.path.join(ROOT, "state", f"{tag}.pkl")
    npz = os.path.join(ROOT, "state", f"ckpt_{tag}.npz")
    if mode == "dump":
        st = pickle.load(open(pkl, "rb"))
        P = st["P"]
        arrs = {k: v.astype(np.float16) for k, v in P.items()}
        np.savez_compressed(npz, __step__=np.int64(st["step"]),
                            __keys__=json.dumps(list(P.keys())), **arrs)
        print(f"dump {tag}: step {st['step']} -> {os.path.getsize(npz)/1e6:.1f} MB")
    elif mode == "restore":
        if not os.path.exists(npz):
            print(f"restore {tag}: gak ada {npz} — skip (dari nol)")
            return
        z = np.load(npz, allow_pickle=False)
        keys = json.loads(str(z["__keys__"]))
        P = {k: z[k].astype(np.float32) for k in keys}
        step = int(z["__step__"])
        st = {
            "P": P,
            "opt": {"m": {k: np.zeros_like(v) for k, v in P.items()},
                    "v": {k: np.zeros_like(v) for k, v in P.items()},
                    "t": 0},
            "step": step,
            "data_rng": np.random.default_rng(4242),
            "history": {"loss": [], "val": [(step, None)]},
        }
        pickle.dump(st, open(pkl, "wb"), protocol=4)
        print(f"restore {tag}: step {step} -> {pkl}")

if __name__ == "__main__":
    main()
