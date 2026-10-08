"""Pixanva tag system — vocab & spesifikasi tiap tag (dipakai renderer + UI).
Vocab layout:
  0..12   special tokens
  13..108 96 warna palet (image token)
  109+    tag kondisioning
"""
import palette as PA

# ---- special tokens ----
PAD, BOS, SEP, UNCOND = 0, 1, 2, 3
SEC_SCENE, SEC_COLOR, SEC_MOOD, SEC_ORN = 4, 5, 6, 7
GRID16, GRID24, GRID32, GRID48, GRID64 = 8, 9, 10, 11, 12
COLOR_OFFSET = 13           # token warna ke-i = COLOR_OFFSET + idx_palet

# ---- scenes: (id_internal, label_id, weight) ----
SCENES = [
    ("gunung",   "Gunung",        1.2),
    ("laut",     "Laut",          1.2),
    ("hutan",    "Hutan",         1.1),
    ("kota",     "Kota",          1.1),
    ("gurun",    "Gurun",         1.0),
    ("angkasa",  "Angkasa",       1.0),
    ("aurora",   "Aurora",        0.8),
    ("pantai",   "Pantai",        1.0),
    ("danau",    "Danau",         0.9),
    ("sawah",    "Sawah",         0.9),
    ("kanjon",   "Kanjon",        0.8),
    ("volkano",  "Volkano",       0.8),
    ("bunga",    "Padang Bunga",  0.9),
    ("terjun",   "Air Terjun",    0.8),
    ("salju",    "Tundra Salju",  0.8),
    ("awan",     "Lautan Awan",   0.8),
]

# ---- color tags: (id, label, hue_shift, sat_mult, val_mult, val_bias) ----
COLORS = [
    ("hangat",   "Hangat",   +0.045, 1.10, 1.00, +0.02),
    ("dingin",   "Dingin",   -0.075, 0.95, 0.98, -0.02),
    ("neon",     "Neon",     +0.02,  1.55, 1.08, +0.05),
    ("pastel",   "Pastel",   +0.01,  0.45, 1.05, +0.16),
    ("monokrom", "Monokrom", 0.0,    0.06, 1.00, 0.0),
    ("bumi",     "Bumi",     +0.02,  0.72, 0.92, -0.02),
    ("tropis",   "Tropis",   +0.03,  1.30, 1.04, +0.03),
    ("gelap",    "Gelap",    -0.01,  1.00, 0.62, -0.10),
    ("cerah",    "Cerah",    +0.005, 1.05, 1.12, +0.08),
    ("vintage",  "Vintage",  +0.075, 0.60, 0.94, +0.04),
    ("es",       "Es",       -0.13,  0.60, 1.06, +0.06),
    ("smaragd",  "Smaragd",  -0.04,  1.15, 0.96, 0.0),
]

# ---- moods: (id, label, spec) spec dipakai renderer ----
MOODS = [
    ("pagi",    "Pagi",     {"sky": "pagi",   "fog": 0.10, "stars": 0.0, "rain": 0, "glow": 0.06}),
    ("siang",   "Siang",    {"sky": "siang",  "fog": 0.02, "stars": 0.0, "rain": 0, "glow": 0.0}),
    ("senja",   "Senja",    {"sky": "senja",  "fog": 0.05, "stars": 0.15, "rain": 0, "glow": 0.10}),
    ("malam",   "Malam",    {"sky": "malam",  "fog": 0.03, "stars": 1.0, "rain": 0, "glow": 0.0}),
    ("kabut",   "Berkabut", {"sky": "siang",  "fog": 0.55, "stars": 0.0, "rain": 0, "glow": 0.0}),
    ("badai",   "Badai",    {"sky": "badai",  "fog": 0.15, "stars": 0.0, "rain": 1, "glow": 0.0}),
    ("mystic",  "Mystic",   {"sky": "mystic", "fog": 0.30, "stars": 0.45, "rain": 0, "glow": 0.14}),
    ("mimpi",   "Mimpi",    {"sky": "mimpi",  "fog": 0.22, "stars": 0.25, "rain": 0, "glow": 0.20}),
]

# ---- ornaments ----
ORNAMENTS = [
    ("bintang", "Bintang"), ("bulan", "Bulan"), ("matahari", "Matahari"),
    ("awan", "Awan"), ("burung", "Burung"), ("perahu", "Perahu"),
    ("balon", "Balon Udara"), ("kupu", "Kupu-kupu"), ("bunga", "Bunga"),
    ("petir", "Petir"), ("pelangi", "Pelangi"), ("meteor", "Meteor"),
    ("salju", "Salju"), ("pohon", "Pohon"),
]

# ---- id assignment (stabil!) ----
SPECIALS = ["PAD", "BOS", "SEP", "UNCOND", "SEC_SCENE", "SEC_COLOR", "SEC_MOOD",
            "SEC_ORN", "G16", "G24", "G32", "G48", "G64"]

SCENE_IDS = {s[0]: 109 + i for i, s in enumerate(SCENES)}
COLOR_IDS = {c[0]: 125 + i for i, c in enumerate(COLORS)}
MOOD_IDS = {m[0]: 137 + i for i, m in enumerate(MOODS)}
ORN_IDS = {o[0]: 145 + i for i, o in enumerate(ORNAMENTS)}
VOCAB = 145 + len(ORNAMENTS)  # 159

GRID_TOKEN = {16: GRID16, 24: GRID24, 32: GRID32, 48: GRID48, 64: GRID64}

# prob tag dibuang saat training (untuk CFG)
UNCOND_P = 0.12


def sample_cond(rng, allow_missing=True):
    """Sample kombinasi tag acak. Return dict."""
    import numpy as np
    scene = SCENES[int(rng.integers(len(SCENES)))][0]
    color = COLORS[int(rng.integers(len(COLORS)))][0]
    mood = None
    if (not allow_missing) or rng.random() < 0.85:
        mood = MOODS[int(rng.integers(len(MOODS)))][0]
    orn = []
    n = rng.integers(0, 4)
    if n:
        idx = rng.choice(len(ORNAMENTS), size=n, replace=False)
        orn = [ORNAMENTS[i][0] for i in idx]
    return {"scene": scene, "color": color, "mood": mood, "orn": orn}


def cond_tokens(cond, G):
    """Encode kondisi jadi list token. Return (tokens list, cond dict final)."""
    toks = [BOS]
    if cond.get("uncond"):
        toks += [UNCOND, GRID_TOKEN[G], SEP]
        return toks, cond
    toks += [SEC_SCENE, SCENE_IDS[cond["scene"]]]
    toks += [SEC_COLOR, COLOR_IDS[cond["color"]]]
    if cond.get("mood"):
        toks += [SEC_MOOD, MOOD_IDS[cond["mood"]]]
    for o in cond.get("orn", []):
        toks += [SEC_ORN, ORN_IDS[o]]
    toks += [GRID_TOKEN[G], SEP]
    return toks, cond


def decode_prompt(tok_ids):
    """Debug: token ids -> string."""
    names = dict(enumerate(SPECIALS))
    for k, v in {**SCENE_IDS, **COLOR_IDS, **MOOD_IDS, **ORN_IDS}.items():
        names[v] = k
    out = []
    for t in tok_ids:
        if COLOR_OFFSET <= t < COLOR_OFFSET + 96:
            out.append(f"c{t - COLOR_OFFSET}")
        else:
            out.append(names.get(t, "?"))
    return " ".join(out)
