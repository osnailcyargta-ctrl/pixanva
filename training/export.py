"""Export bobot training -> format web (.bin fp16 + meta JSON + js/palette.js + js/tags.js).
Format .bin: [uint32 meta_len][meta_json utf8][fp16 x N sesuai urutan PARAM_ORDER]
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import palette as PA
import tags as TG
import model as M
import imajin_data as ID
from train import get_cfg, MODELS

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

# versi pixanva — heavy pake skema qwerty (q..m, lanjut w...), tiap huruf = +0.2
VERSIONS = {
    "light": "1.0", "light13": "1.3",
    "dark": "1.0",
    "dark12": "1.2",
    "dark15": "1.5",
    "heavy": "qq", "heavyqw": "qw", "heavyqr": "qr",
}

# urutan + tanggal rilis — dipakai sort "Terbaru" di popup More models
RELEASES = {
    "light": (1, "2026-10-07"), "dark": (2, "2026-10-07"), "heavy": (3, "2026-10-07"),
    "dark12": (4, "2026-10-09"), "heavyqw": (5, "2026-10-09"), "dark15": (6, "2026-10-09"),
    "light13": (7, "2026-10-10"), "heavyqr": (8, "2026-10-10"),
}

# 3 kartu utama di layar depan = versi terbaru tiap keluarga; sisanya masuk popup
MAIN_MODELS = {"light13", "heavyqr", "dark15"}

# urutan meta.json = lama -> baru (ri = release index)
MODEL_INFO = [
    dict(id="light",  label="Pixanva Light 1.0",  desc="Cepat & ringan — paling sederhana",
         g48_thr=800),
    dict(id="dark",   label="Pixanva Dark 1.0",   desc="Seimbang — kualitas & kecepatan",
         g48_thr=800),
    dict(id="heavy",  label="Pixanva Heavy QQ",   desc="Paling kuat generasi awal — paling detail, paling lambat",
         g48_thr=1400),
    dict(id="dark12", label="Pixanva Dark 1.2",   desc="Versi 1.2 — dilatih lebih lama, 96/128px makin rapi, lebih taat tag",
         g48_thr=1230),
    dict(id="heavyqw", label="Pixanva Heavy QW",  desc="Update 0.2 dari QQ — makin taat tag & makin detail",
         g48_thr=1880),
    dict(id="dark15", label="Pixanva Dark 1.5",   desc="Update 0.3 — parameter naik beneran (2.7M → 4.9M, 6 → 8 lapis)",
         g48_thr=1400, new=True),
    dict(id="light13", label="Pixanva Light 1.3", desc="Update 0.3 — parameter naik beneran (813k → 1.9M, 4 → 6 lapis) — tetap ringan tapi makin tajam",
         g48_thr=800, new=True),
    dict(id="heavyqr", label="Pixanva Heavy QR",  desc="Update QW→QR — parameter naik beneran (6.4M → 10M, 8 → 10 lapis) — paling detail sepanjang masa",
         g48_thr=700, new=True),
]


def qwerty_version(i):
    QW = "qwertyuiopasdfghjklzxcvbnm"
    return QW[i // 26] + QW[i % 26]


def export_model(name):
    cfg = get_cfg(name)
    ck = os.path.join(ROOT, "state", f"{name}.pkl")
    with open(ck, "rb") as f:
        st = pickle.load(f)
    P, step = st["P"], st["step"]
    hist = st["history"]
    val = hist["val"][-1][1] if hist["val"] else None
    order = list(M.param_shapes(cfg).keys())
    meta = {
        "name": name,
        "cfg": cfg,
        "step": step,
        "val_loss": val,
        "order": order,
        "n_params": int(sum(P[k].size for k in order)),
    }
    blob = b""
    for k in order:
        blob += P[k].astype(np.float16).tobytes()
    out = os.path.join(ROOT, "models", f"{name}.bin")
    with open(out, "wb") as f:
        f.write(len(meta_json := json.dumps(meta)).to_bytes(4, "little"))
        f.write(meta_json.encode())
        f.write(blob)
    mb = os.path.getsize(out) / 1e6
    print(f"[{name}] step {step} val {val} -> {out} ({mb:.1f} MB, {meta['n_params']:,} params)")
    return meta


def export_webdata():
    """js/palette.js + js/tags.js — sumber kebenaran dari Python biar sinkron."""
    pal = PA.palette_hex()
    tags = {
        "scenes": [{"id": s[0], "label": s[1]} for s in TG.SCENES],
        "colors": [{"id": c[0], "label": c[1]} for c in TG.COLORS],
        "moods": [{"id": m[0], "label": m[1]} for m in TG.MOODS],
        "ornaments": [{"id": o[0], "label": o[1]} for o in TG.ORNAMENTS],
    }
    ids = {
        "COLOR_OFFSET": TG.COLOR_OFFSET,
        "SPECIALS": {n: i for i, n in enumerate(TG.SPECIALS)},
        "SCENE_IDS": TG.SCENE_IDS, "COLOR_IDS": TG.COLOR_IDS,
        "MOOD_IDS": TG.MOOD_IDS, "ORN_IDS": TG.ORN_IDS,
        "GRID_TOKEN": {str(k): v for k, v in TG.GRID_TOKEN.items()},
        "VOCAB": TG.VOCAB,
        # kelas imajin (model gambar slot laten + parser) — gak dipakai model tag lama
        "IMJ": {
            "SEC_SUBJ": TG.SEC_SUBJ, "SEC_ATTR": TG.SEC_ATTR, "SEC_SCOL": TG.SEC_SCOL,
            "NONE": TG.NONE_IMJ,
            "SUBJ_IDS": ID.SUBJ_IDS, "ATTR_IDS": ID.ATTR_IDS, "SCOL_IDS": ID.SCOL_IDS,
        },
    }
    js = (
        "// AUTO-GENERATED dari training/palette.py & tags.py — jangan edit manual\n"
        "export const PALETTE = " + json.dumps(pal) + ";\n"
        "export const TAGS = " + json.dumps(tags, ensure_ascii=False) + ";\n"
        "export const IDS = " + json.dumps(ids) + ";\n"
    )
    with open(os.path.join(ROOT, "js", "data.js"), "w") as f:
        f.write(js)
    print("js/data.js ditulis")


def export_meta_all():
    infos = []
    for info in MODEL_INFO:
        name = info["id"]
        ck = os.path.join(ROOT, "state", f"{name}.pkl")
        if not os.path.exists(ck):
            continue
        with open(ck, "rb") as f:
            st = pickle.load(f)
        cfg = get_cfg(name)
        n = sum(int(np.prod(s)) for s in M.param_shapes(cfg).values())
        ri, rel = RELEASES.get(name, (99, ""))
        infos.append({
            "id": name, "label": info["label"],
            "version": VERSIONS[name],
            "d": cfg["d"], "L": cfg["L"], "H": cfg["H"],
            "params": n, "step": st["step"],
            # g48: model pernah di-finetune grid 48x48 (utk output 96/128px)
            "g48": st["step"] > info["g48_thr"],
            "val_loss": st["history"]["val"][-1][1] if st["history"]["val"] else None,
            "file": f"models/{name}.bin",
            "desc": info["desc"],
            "isNew": bool(info.get("new", False)),
            "ri": ri, "released": rel,
            "main": name in MAIN_MODELS,
        })
    # PENTING: pertahankan model non-gambar (assistant, prompter) yang di-append
    # script export lain — jangan sampai kehapus tiap re-export meta
    keep_ids = {i["id"] for i in infos}
    mp = os.path.join(ROOT, "models", "meta.json")
    if os.path.exists(mp):
        try:
            with open(mp) as f:
                prev = json.load(f)
            for m in prev.get("models", []):
                if m.get("id") not in keep_ids and m.get("kind"):
                    infos.append(m)
        except Exception:
            pass
    with open(mp, "w") as f:
        json.dump({"models": infos, "palette": PA.palette_hex()}, f, indent=1)
    print("models/meta.json ditulis")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "webdata"):
        export_webdata()
    names = [i["id"] for i in MODEL_INFO]
    for name in names:
        if which in ("all", name) and os.path.exists(os.path.join(ROOT, "state", f"{name}.pkl")):
            export_model(name)
    export_meta_all()
