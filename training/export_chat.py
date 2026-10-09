"""Export Pixanva Assistant → models/assistant.bin (+ vocab) + entry di meta.json.
Format bin sama persis dgn model gambar: [u32 metaLen][metaJSON][fp16 weights].
UI harus skip model dgn kind:"chat" di pemilih model gambar.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import train_chat as TC

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")


def main():
    cfg = TC.CFG or {"V": 0, "d": TC.ARCH["d"], "L": TC.ARCH["L"],
                     "H": TC.ARCH["H"], "mlp": TC.ARCH["mlp"], "theta": TC.ARCH["theta"]}
    ck = os.path.join(ROOT, "state", "chat.pkl")
    with open(ck, "rb") as f:
        st = pickle.load(f)
    P, step = st["P"], st["step"]
    hist = st["history"]
    val = hist["val"][-1][1] if hist["val"] else None
    # V harus dari vocab asli, bukan cfg sisa — rebuild dari itos
    with open(os.path.join(BASE, "chat_vocab.json")) as f:
        itos = json.load(f)["itos"]
    cfg = dict(cfg)
    cfg["V"] = len(itos)

    order = list(M.param_shapes(cfg).keys())
    meta = {
        "name": "assistant",
        "kind": "chat",
        "cfg": cfg,
        "step": step,
        "val_loss": val,
        "order": order,
        "n_params": int(sum(P[k].size for k in order)),
    }
    blob = b""
    for k in order:
        blob += P[k].astype(np.float16).tobytes()
    out = os.path.join(ROOT, "models", "assistant.bin")
    with open(out, "wb") as f:
        mj = json.dumps(meta)
        f.write(len(mj).to_bytes(4, "little"))
        f.write(mj.encode())
        f.write(blob)

    # vocab buat tokenizer browser
    with open(os.path.join(ROOT, "models", "chat_vocab.json"), "w") as f:
        json.dump({"itos": itos,
                   "specials": {"PAD": 0, "BOS": 1, "EOS": 2, "U": 3, "A": 4}},
                  f, ensure_ascii=False)

    # entry meta.json — hidden dari pemilih model gambar
    mp = os.path.join(ROOT, "models", "meta.json")
    with open(mp) as f:
        meta_all = json.load(f)
    meta_all["models"] = [m for m in meta_all["models"] if m["id"] != "assistant"]
    meta_all["models"].append({
        "id": "assistant", "label": "Pixanva Assistant", "kind": "chat",
        "hidden": True, "version": "1.0",
        "d": cfg["d"], "L": cfg["L"], "H": cfg["H"],
        "params": meta["n_params"], "step": step,
        "g48": False, "val_loss": val,
        "file": "models/assistant.bin",
        "vocab": "models/chat_vocab.json",
        "desc": "LLM chat dari nol — bikin ide prompt di tombol asisten",
        "isNew": True, "ri": 99, "released": "2026-10-09", "main": False,
    })
    with open(mp, "w") as f:
        json.dump(meta_all, f, indent=1)
    mb = os.path.getsize(out) / 1e6
    print(f"[assistant] step {step} val {val} -> {out} ({mb:.1f} MB, {meta['n_params']:,} params)")
    print("models/chat_vocab.json + meta.json diupdate")


if __name__ == "__main__":
    main()
