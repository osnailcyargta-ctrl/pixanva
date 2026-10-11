#!/usr/bin/env python3
"""Tambah/replace entri heavyqr di models/meta.json — AMAN (gak nyentuh entri lain)."""
import json, struct, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
mp = os.path.join(ROOT, "models", "meta.json")

with open(os.path.join(ROOT, "models", "heavyqr.bin"), "rb") as f:
    n = struct.unpack("<I", f.read(4))[0]
    hdr = json.loads(f.read(n))

entry = {
    "id": "heavyqr", "label": "Pixanva Heavy QR", "version": "qr",
    "d": hdr["cfg"]["d"], "L": hdr["cfg"]["L"], "H": hdr["cfg"]["H"],
    "params": hdr["n_params"], "step": hdr["step"],
    "g48": hdr["step"] > 700,
    "val_loss": hdr.get("val_loss"),
    "file": "models/heavyqr.bin",
    "desc": "Update QW→QR — parameter naik beneran (6.4M → 10M, 8 → 10 lapis) — paling detail sepanjang masa",
    "isNew": True, "ri": 8, "released": "2026-10-10", "main": True,
}

mj = json.load(open(mp))
mj["models"] = [m for m in mj["models"] if m.get("id") != "heavyqr"]
ri_order = {"light": 1, "dark": 2, "heavy": 3, "dark12": 4, "heavyqw": 5,
            "dark15": 6, "light13": 7, "heavyqr": 8,
            "assistant": 98, "prompter": 99, "imajin": 100}
models = mj["models"]
idx = len(models)
for i, m in enumerate(models):
    if ri_order.get(m.get("id"), 50) > 8:
        idx = i
        break
models.insert(idx, entry)
# heavy QW turun dari kartu utama (digantikan QR sebagai wakil keluarga heavy)
for m in models:
    if m.get("id") == "heavyqw":
        m["main"] = False
json.dump(mj, open(mp, "w"), indent=1, ensure_ascii=False)
print(f"meta.json: heavyqr posisi {idx+1}/{len(models)} (step {hdr['step']}, {hdr['n_params']:,} params); heavyqw -> popup")
