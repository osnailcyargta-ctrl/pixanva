"""Export Prompter → models/prompter.bin (+ vocab) + entry meta.json.
Format bin sama dgn model lain: [u32 metaLen][metaJSON][fp16 weights].
"""
import sys, os, json, pickle, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import train_prompter as TP
import prompt_data as PD

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(TP.ARCHS))
    args = ap.parse_args()
    size = args.size

    # rebuild CFG + vocab persis seperti training (deterministik)
    TP.ARCH = TP.ARCHS[size]
    itos, stoi, _val = TP.load_all()
    CFG = {"V": len(itos), "d": TP.ARCH["d"], "L": TP.ARCH["L"], "H": TP.ARCH["H"],
           "mlp": TP.ARCH["mlp"], "theta": 1000.0}

    ck = os.path.join(ROOT, "state", f"prompter{size}.pkl")
    with open(ck, "rb") as f:
        st = pickle.load(f)
    P, step = st["P"], st["step"]
    hist = st["history"]
    acc = hist["val"][-1][1] if hist["val"] else None

    order = list(M.param_shapes(CFG).keys())
    meta = {
        "name": "prompter", "kind": "prompter", "stage": size,
        "cfg": CFG, "step": step, "slot_acc": acc,
        "order": order, "n_params": int(sum(P[k].size for k in order)),
    }
    blob = b""
    for k in order:
        blob += P[k].astype(np.float16).tobytes()
    out = os.path.join(ROOT, "models", "prompter.bin")
    with open(out, "wb") as f:
        mj = json.dumps(meta)
        f.write(len(mj).to_bytes(4, "little"))
        f.write(mj.encode())
        f.write(blob)

    with open(os.path.join(ROOT, "models", "prompt_vocab.json"), "w") as f:
        json.dump({"itos": itos,
                   "specials": {"PAD": PD.PAD, "BOS": PD.BOS, "A": PD.A,
                                "EOS": PD.EOS, "NONE": PD.NONE}},
                  f, ensure_ascii=False)

    # entry meta.json — hidden dari pemilih model gambar
    mp = os.path.join(ROOT, "models", "meta.json")
    with open(mp) as f:
        meta_all = json.load(f)
    meta_all["models"] = [m for m in meta_all["models"] if m["id"] != "prompter"]
    meta_all["models"].append({
        "id": "prompter", "label": f"Pixanva Prompter {size.upper()}", "kind": "prompter",
        "hidden": True, "version": size, "stage": size,
        "d": CFG["d"], "L": CFG["L"], "H": CFG["H"],
        "params": meta["n_params"], "step": step,
        "g48": False, "val_loss": acc,
        "file": "models/prompter.bin",
        "vocab": "models/prompt_vocab.json",
        "desc": "AI custom-prompt — terjemahin teks bebas jadi tag buat generator gambar",
        "isNew": True, "ri": 100, "released": "2026-10-10", "main": False,
    })
    with open(mp, "w") as f:
        json.dump(meta_all, f, indent=1)
    mb = os.path.getsize(out) / 1e6
    print(f"[prompter{size}] step {step} acc {acc} -> {out} ({mb:.1f} MB, {meta['n_params']:,} params)")


if __name__ == "__main__":
    main()
