"""Export Pixanva Imajin v4 → models/imajin{5,12,17}m.bin + update meta.json.
Model v4 = teks bebas MASUK langsung (kata + slot laten internal + char fallback).
Vocab v4 (models/imajin_vocab.json) di-build oleh train_imajin.ensure_vocab().
Jalankan: python3 export_imajin.py [5m|12m|17m|all]
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import train_imajin as T

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

INFO = {
    "5m": dict(label="Pixanva Imajin 5M", version="0.5",
               desc="AI custom-prompt beneran — ketik apa aja (ikan terbang di volkano?), AI nggambar langsung dari teks lo",
               ri=20, released="2026-10-10"),
    "12m": dict(label="Pixanva Imajin 12M", version="1.2",
                desc="Fase 2 — makin paham prompt bebas, subjek & atribut makin taat",
                ri=21, released="2026-10-10"),
    "17m": dict(label="Pixanva Imajin 17M", version="1.7",
                desc="Fase final — komposisi paling tajam, paling ngerti prompt ngawur",
                ri=22, released="2026-10-10"),
}


def export_bin(tag, cfg, ckpt_rel, out_name):
    ck = os.path.join(ROOT, "state", ckpt_rel)
    with open(ck, "rb") as f:
        st = pickle.load(f)
    P, step, hist = st["P"], st["step"], st["history"]
    val = None
    if hist.get("val"):
        val = hist["val"][-1][1]
    order = list(M.param_shapes(cfg).keys())
    meta = {"name": tag, "cfg": cfg, "step": step, "val_loss": val,
            "order": order, "n_params": int(sum(P[k].size for k in order))}
    blob = b""
    for k in order:
        blob += P[k].astype(np.float16).tobytes()
    out = os.path.join(ROOT, "models", f"{out_name}.bin")
    with open(out, "wb") as f:
        f.write(len(mj := json.dumps(meta)).to_bytes(4, "little"))
        f.write(mj.encode())
        f.write(blob)
    print(f"[{tag}] step {step} val {val} -> {out} ({os.path.getsize(out)/1e6:.1f} MB, {meta['n_params']:,} params)")
    return meta


def update_meta(entry):
    mp = os.path.join(ROOT, "models", "meta.json")
    with open(mp) as f:
        mj = json.load(f)
    mj["models"] = [m for m in mj["models"] if m.get("id") != entry["id"]]
    mj["models"].append(entry)
    with open(mp, "w") as f:
        json.dump(mj, f, indent=1)
    print(f"meta.json: +{entry['id']} (kind {entry.get('kind')})")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "5m"
    sizes = ["5m", "12m", "17m"] if which == "all" else [which]
    vj = T.ensure_vocab()

    for size in sizes:
        arch = T.ARCHS[size]
        cfg = {"V": len(vj["itos"]), "d": arch["d"], "L": arch["L"], "H": arch["H"],
               "mlp": arch["mlp"], "theta": 1000.0}
        meta = export_bin(f"imajin{size}", cfg, f"imajin{size}.pkl", f"imajin{size}")
        info = INFO[size]
        update_meta({
            "id": "imajin", "label": info["label"], "kind": "t2i", "hidden": True,
            "version": info["version"],
            "d": cfg["d"], "L": cfg["L"], "H": cfg["H"],
            "params": meta["n_params"], "step": meta["step"],
            "g48": meta["step"] > arch["steps"],
            "val_loss": meta["val_loss"], "file": f"models/imajin{size}.bin",
            "vocab": "models/imajin_vocab.json",
            "desc": info["desc"], "isNew": True,
            "ri": info["ri"], "released": info["released"], "main": False,
        })


if __name__ == "__main__":
    main()
