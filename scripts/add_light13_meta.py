#!/usr/bin/env python3
"""Tambah/replace entri light13 di models/meta.json — AMAN (gak nyentuh entri lain).
Jalanin SETELAH export_model('light13') biar params bisa dibaca dari bin."""
import json, struct, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
mp = os.path.join(ROOT, "models", "meta.json")

# baca header bin light13 buat step/val/n_params yang akurat
with open(os.path.join(ROOT, "models", "light13.bin"), "rb") as f:
    n = struct.unpack("<I", f.read(4))[0]
    hdr = json.loads(f.read(n))

entry = {
    "id": "light13", "label": "Pixanva Light 1.3", "version": "1.3",
    "d": hdr["cfg"]["d"], "L": hdr["cfg"]["L"], "H": hdr["cfg"]["H"],
    "params": hdr["n_params"], "step": hdr["step"],
    "g48": hdr["step"] > 800,
    "val_loss": hdr.get("val_loss"),
    "file": "models/light13.bin",
    "desc": "Update 0.3 — parameter naik beneran (813k → 1.9M, 4 → 6 lapis) — tetap ringan tapi makin tajam",
    "isNew": True, "ri": 7, "released": "2026-10-10", "main": True,
}

mj = json.load(open(mp))
mj["models"] = [m for m in mj["models"] if m.get("id") != "light13"]
# sisip sesuai urutan rilis: sebelum assistant/prompter/imajin (ri 7 > ri 6 dark15)
ri_order = {"light": 1, "dark": 2, "heavy": 3, "dark12": 4, "heavyqw": 5,
            "dark15": 6, "light13": 7, "assistant": 98, "prompter": 99, "imajin": 100}
models = mj["models"]
idx = len(models)
for i, m in enumerate(models):
    if ri_order.get(m.get("id"), 50) > 7:
        idx = i
        break
models.insert(idx, entry)
json.dump(mj, open(mp, "w"), indent=1, ensure_ascii=False)
print(f"meta.json: light13 masuk di posisi {idx+1}/{len(models)} (step {hdr['step']}, {hdr['n_params']:,} params)")
