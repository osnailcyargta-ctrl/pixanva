"""Pixanva Imajin v2 — data on-the-fly: teks bebas KOMPOSISIONAL → gambar.
SATU model, dua tugas dalam satu sequence:
  [BOS] w1..wk [A] slot1..slot9 [GRID_G] [SEP] c1..c(G*G)
  slot = scene, color, mood, orn1, orn2, subj, attr1, attr2, scol
Model PREDIKSI slot-nya sendiri dari kata (kayak prompter), lalu slot yang
mengondisikan gambar (mekanisme tag model terbukti nempel di piksel).
Slot = LATEN internal — user gak pernah lihat chip tag.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import tags as TG
import subjects as SJ
import prompt_data as PD  # sinonim scene/color/mood/orn + fillers — di-reuse

PAD, BOS, SEP, UNCOND = 0, 1, 2, 3
WORD_BASE = 193          # 0..12 specials, 13..108 palet, 109..158 tag lama, 159..192 kelas imajin
MAX_WORDS = 16
N_SLOTS = 9
N_WORDS_VOCAB = 2400     # 193..2592 kata
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def char_base(n_words=N_WORDS_VOCAB):
    return WORD_BASE + n_words


def char_ids(n_words=N_WORDS_VOCAB):
    """Kata yang GAK ada di vocab → dirinci jadi token karakter (open vocab!)."""
    b = char_base(n_words)
    return {c: b + i for i, c in enumerate(CHARS)}


def charw_id(n_words=N_WORDS_VOCAB):
    return char_base(n_words) + len(CHARS)   # marker awal kata-fallback

# ---- token kelas imajin: SEC markers + NONE + subj (12) + attr (8) + scol (10) ----
A_TOK = 159              # dipakai lama v2 — v3: SEC_SUBJ
SEC_SUBJ, SEC_ATTR, SEC_SCOL, NONE_TOK = 159, 160, 161, 162
SUBJ_IDS = {s: 163 + i for i, s in enumerate(SJ.SUBJECTS)}
ATTR_IDS = {a: 175 + i for i, a in enumerate(SJ.ATTRS)}


def slot_tokens(cond):
    """cond → 9 token slot fix: scene, color, mood, orn1, orn2, subj, attr1, attr2, scol."""
    subj = cond.get("subj")
    attrs = list(subj["attrs"])[:2] if subj else []
    while len(attrs) < 2:
        attrs.append(None)
    return [
        TG.SCENE_IDS[cond["scene"]],
        TG.COLOR_IDS[cond["color"]],
        TG.MOOD_IDS[cond["mood"]] if cond.get("mood") else NONE_TOK,
        TG.ORN_IDS[cond["orn"][0]] if len(cond.get("orn", [])) > 0 else NONE_TOK,
        TG.ORN_IDS[cond["orn"][1]] if len(cond.get("orn", [])) > 1 else NONE_TOK,
        SUBJ_IDS[subj["id"]] if subj else NONE_TOK,
        ATTR_IDS[attrs[0]] if attrs[0] else NONE_TOK,
        ATTR_IDS[attrs[1]] if attrs[1] else NONE_TOK,
        SCOL_IDS[subj["col"]] if subj and subj.get("col") else NONE_TOK,
    ]

# ---------------- sinonim subjek & atribut ----------------
SUBJ_SYN = {
    "ikan":   ["ikan", "fish", "ikan hias", "ikan besar", "kakap", "salmon", "ikan nemo", "ikan mas"],
    "burung": ["burung", "bird", "elang", "perkutut", "burung hantu", "garuda", "kawanan burung", "pipit"],
    "naga":   ["naga", "dragon", "wyvern", "naga china", "draco", "lorong naga"],
    "kucing": ["kucing", "cat", "meong", "kucing oren", "neko", "anjing", "dog", "kitten"],
    "robot":  ["robot", "mecha", "droid", "robot jadul", "andro", "mechanical", "robot mainan"],
    "kapal":  ["kapal", "perahu", "boat", "ship", "sampan", "yacht", "jukung", "feri", "canoe"],
    "kupu":   ["kupu kupu", "butterfly", "rama rama", "kupu2", "kupu", "moth"],
    "pohon":  ["pohon", "tree", "cemara", "pinus", "bonsai", "pohon beringin", "palem"],
    "bunga":  ["bunga", "flower", "mawar", "sakura", "tulip", "sunflower", "melati", "anggrek"],
    "balon":  ["balon udara", "hot air balloon", "balon", "zeppelin", "dirigibel", "airship"],
    "rumah":  ["rumah", "house", "kabin", "cabin", "vila", "gubuk", "pondok", "rumah pohon", "cottage"],
    "gajah":  ["gajah", "elephant", "gajah afrika", "mamut", "gajah sumatra"],
}
ATTR_SYN = {
    "terbang":  ["terbang", "flying", "melayang", "ngambang", "di udara", "soaring", "hovering"],
    "raksasa":  ["raksasa", "giant", "gede banget", "super besar", "colossal", "titanic", "mega", "jumbo"],
    "kecil":    ["kecil", "tiny", "mini", "mungil", "imut", "small", "pygmy"],
    "neon":     ["neon", "glowing", "bercahaya", "cyber", "led", "glow in the dark"],
    "emas":     ["emas", "gold", "golden", "keemasan"],
    "api":      ["api", "berapi", "on fire", "terbakar", "burning", "flaming", "magma", "membakar"],
    "es":       ["es", "ice", "frozen", "beku", "icy", "membeku"],
    "kristal":  ["kristal", "crystal", "crystalline", "berlian", "diamond", "permata"],
}
SUBJ_COLOR_SYN = {
    "merah": ["merah", "red"], "biru": ["biru", "blue"], "hijau": ["hijau", "green"],
    "hitam": ["hitam", "black"], "putih": ["putih", "white"], "ungu": ["ungu", "purple"],
    "pink": ["pink", "merah muda"], "oranye": ["oranye", "orange"], "kuning": ["kuning", "yellow"],
    "perak": ["perak", "silver"],
}
SCOL_IDS = {c: 183 + i for i, c in enumerate(SUBJ_COLOR_SYN)}
ATTR_PAIRS_OK = {  # kombinasi atribut yang boleh bareng
    ("raksasa", "kecil"), ("terbang", "raksasa"), ("terbang", "kecil"),
    ("neon", "raksasa"), ("neon", "kecil"), ("neon", "terbang"), ("neon", "api"),
    ("emas", "raksasa"), ("emas", "terbang"), ("emas", "kecil"),
    ("api", "raksasa"), ("api", "terbang"), ("api", "kecil"),
    ("es", "raksasa"), ("es", "terbang"), ("es", "kecil"),
    ("kristal", "raksasa"), ("kristal", "terbang"), ("kristal", "kecil"),
    ("emas", "api"), ("es", "kristal"),
}
# subjek yang nyambung sama "api" (biar data gak absurd mulu)
FIRE_OK = {"naga", "kucing", "robot", "rumah", "kapal", "pohon", "gajah", "ikan"}

SCENE_PREP = ["di", "di", "di", "di atas", "keliling", "depan", "pinggir"]


def _pick(rng, d):
    return d[int(rng.integers(len(d)))] if hasattr(rng, "integers") else d[rng.randint(0, len(d) - 1)]


def sample_cond(rng):
    """Cond komposisional: scene/color/mood/orn + subjek (70%) + atribut."""
    cond = {"scene": _pick(rng, [s[0] for s in TG.SCENES]),
            "color": _pick(rng, [c[0] for c in TG.COLORS]),
            "mood": None, "orn": [], "subj": None}
    if rng.random() < 0.75:
        cond["mood"] = _pick(rng, [m[0] for m in TG.MOODS])
        if rng.random() < 0.45:
            cond["color"] = _pick(rng, PD.MOOD_COLOR_AFFINITY[cond["mood"]])
    n_orn = int(rng.integers(0, 4))
    if rng.random() < 0.55 and n_orn:
        idx = rng.choice(len(TG.ORNAMENTS), size=min(n_orn, 2), replace=False)
        cond["orn"] = [TG.ORNAMENTS[i][0] for i in np.atleast_1d(idx)]

    if rng.random() < 0.70:
        sid = _pick(rng, SJ.SUBJECTS)
        attrs = []
        r = rng.random()
        n_attr = 0 if r < 0.25 else (1 if r < 0.72 else 2)
        for _ in range(n_attr):
            for _try in range(8):
                a = _pick(rng, SJ.ATTRS)
                if a == "api" and sid not in FIRE_OK:
                    continue
                if attrs and (a, attrs[0]) not in ATTR_PAIRS_OK:
                    continue
                attrs.append(a)
                break
        col = None
        if rng.random() < 0.32 and "neon" not in attrs and "kristal" not in attrs and "emas" not in attrs and "es" not in attrs:
            col = _pick(rng, list(SUBJ_COLOR_SYN.keys()))
        cond["subj"] = {"id": sid, "attrs": attrs, "col": col}
    return cond


def build_text(rng, cond):
    """Caption bebas gaya manusia dari cond — variasi urutan + slang + english mix."""
    subj = cond.get("subj")
    s_w = _pick(rng, PD.SCENE_SYN[cond["scene"]])
    c_w = _pick(rng, PD.COLOR_SYN[cond["color"]]) if rng.random() < 0.82 else None
    m_w = _pick(rng, PD.MOOD_SYN[cond["mood"]]) if cond.get("mood") else None
    o_ws = [_pick(rng, PD.ORN_SYN[o]) for o in cond.get("orn", [])]

    sp = None
    if subj:
        j_w = _pick(rng, SUBJ_SYN[subj["id"]])
        a_ws = [_pick(rng, ATTR_SYN[a]) for a in subj["attrs"]]
        k_w = _pick(rng, SUBJ_COLOR_SYN[subj["col"]]) if subj.get("col") else None
        # urutan frasa subjek: "ikan terbang", "naga api raksasa", "kucing neon warna ungu"
        core = [j_w] + a_ws
        if k_w and rng.random() < 0.75:
            core.insert(int(rng.integers(1, len(core) + 1)), k_w)
        rng.shuffle(core) if hasattr(rng, "shuffle") else None
        sp = " ".join(core)

    prep = _pick(rng, SCENE_PREP)
    style = rng.random() if hasattr(rng, "random") else rng.rand()
    parts = []
    if subj is None:
        # landscape polos — gaya prompt_data lama
        base = PD.build_text(rng, {**cond, "orn": cond.get("orn", [])})
        return base
    if style < 0.30:
        # subjek dulu: "ikan terbang di volkano"
        parts.append(sp)
        parts.append(f"{prep} {s_w}")
        if m_w and rng.random() < 0.55:
            parts.append(m_w)
        if c_w and rng.random() < 0.4:
            parts.append(c_w)
        parts.extend([f"ada {o}" if rng.random() < 0.5 else o for o in o_ws])
    elif style < 0.55:
        # scene dulu: "di volkano ada ikan terbang"
        parts.append(f"{prep} {s_w}")
        parts.append(f"ada {sp}")
        if m_w and rng.random() < 0.5:
            parts.append(m_w)
        parts.extend([f"plus {o}" if rng.random() < 0.4 else o for o in o_ws])
    elif style < 0.78:
        # request: "gw mau naga api raksasa di kota malam dong"
        head = _pick(rng, PD.OPENERS) + sp
        parts.append(head + f" {prep} {s_w}")
        if m_w:
            parts.append(m_w)
        if c_w and rng.random() < 0.45:
            parts.append(f"warnanya {c_w}")
        parts.extend(o_ws)
    elif style < 0.92:
        # english mix: "giant golden dragon in volcano night vibes"
        t = sp + " in " + s_w
        if m_w:
            t += f" {m_w}"
        parts.append(t + _pick(rng, [" vibes", " aesthetic", ""]))
        if c_w:
            parts.append(c_w)
        parts.extend(o_ws)
    else:
        # super pendek: "naga api" / "ikan terbang volkano"
        parts.append(sp)
        if rng.random() < 0.6:
            parts.append(s_w)
        if m_w and rng.random() < 0.3:
            parts.append(m_w)

    txt = " ".join(p for p in parts if p)
    if rng.random() < 0.30:
        txt += _pick(rng, PD.CLOSERS)
    if rng.random() < 0.15:
        txt = _pick(rng, ["eh ", "oi ", "bro ", ""]) + txt
    if rng.random() < 0.18:
        txt += ", " + _pick(rng, PD.FILLERS)
    return " ".join(txt.split())


# 5% prompt kosong/vague — latihan robustness (user ketik ngawur)
def _vague_text(rng):
    t = _pick(rng, PD.OPENERS) + _pick(rng, ["gambar keren", "yang bagus", "yang estetik",
                                             "sesuatu yang indah", "satu aja", "bebas",
                                             "apa aja deh", "surprise me", "yang paling keren"])
    if rng.random() < 0.5:
        t += _pick(rng, PD.CLOSERS)
    return t.strip()


def make_sample(rng):
    """Return (text, cond). cond.subj bisa None."""
    for _ in range(20):
        cond = sample_cond(rng)
        if cond["subj"] is None and rng.random() < 0.35:
            return _vague_text(rng), cond
        txt = build_text(rng, cond)
        nw = len(tok(txt))
        if 1 <= nw <= MAX_WORDS:
            return txt, cond
    return "ikan terbang di volkano", {"scene": "volkano", "color": "hangat", "mood": "senja",
                                       "orn": [], "subj": {"id": "ikan", "attrs": ["terbang"], "col": None}}


# ---------------- tokenizer & vocab ----------------
def tok(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def domain_words():
    """Semua kata domain WAJIB masuk vocab (gak boleh kepotong frekuensi)."""
    ws = set()
    for syns in SUBJ_SYN.values():
        ws.update(" ".join(syns).split())
    for syns in ATTR_SYN.values():
        ws.update(" ".join(syns).split())
    for syns in SUBJ_COLOR_SYN.values():
        ws.update(" ".join(syns).split())
    for d in (PD.SCENE_SYN, PD.COLOR_SYN, PD.MOOD_SYN, PD.ORN_SYN):
        for syns in d.values():
            ws.update(" ".join(syns).split())
    for f in PD.FILLERS + PD.OPENERS + PD.CLOSERS:
        ws.update(f.split())
    ws.update(SCENE_PREP)
    ws.update(["ada", "plus", "warnanya", "langitnya", "warna", "tone", "pas", "yang",
               "vibes", "mood", "aesthetic", "in", "with", "gw", "aku", "mau", "bikin",
               "gambar", "ikan", "langit", "dan", "sama", "the", "a"])
    return sorted(ws)


def build_vocab(n_words=N_WORDS_VOCAB, seed=321):
    """itos: 0..12 specials (None di list — specials lewat map), 13..108 None (palet),
    109..158 None (tag lama), 159..192 kelas imajin, 193+ kata, sisanya blok char
    (None — char lewat map chars). Return (itos, n_words_actual)."""
    from collections import Counter
    rng = np.random.default_rng(seed)
    cnt = Counter()
    for _ in range(26000):
        txt, _c = make_sample(rng)
        cnt.update(tok(txt))
    must = domain_words()
    words = list(must)
    for w, _c in cnt.most_common(n_words * 2):
        if w not in must:
            words.append(w)
        if len(words) >= n_words:
            break
    words += [None] * (n_words - len(words))   # pad — char base selalu stabil di 193+n_words
    itos = [None] * WORD_BASE + words
    # nama kelas di slot-nya (debug + export)
    itos[A_TOK] = "<a>"
    itos[NONE_TOK] = "<none>"
    for s, i in SUBJ_IDS.items():
        itos[i] = "@" + s
    for a, i in ATTR_IDS.items():
        itos[i] = "&" + a
    for c, i in SCOL_IDS.items():
        itos[i] = "$" + c
    itos += [None] * (len(CHARS) + 1)   # blok char + CHARW — gak masuk itos
    return itos, sum(1 for w in words if w)


def _mangle(w, rng):
    """Typo beneran: huruf ilang / dobel / ketukar — biarin model belajar char-level."""
    if len(w) < 3:
        return w
    r = rng.random()
    i = int(rng.integers(len(w)))
    if r < 0.62:                      # drop 1 huruf
        return w[:i] + w[i + 1:]
    if r < 0.80:                      # dobel 1 huruf
        return w[:i] + w[i] + w[i:]
    if r < 0.92 and len(w) >= 4:      # tukar 2 huruf sebelahan
        j = min(i, len(w) - 2)
        return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    # vokal diganti vokal lain (biar beneran beda dari aslinya)
    vw = "aeiou"
    idx = [k for k, ch in enumerate(w) if ch in vw]
    if not idx:
        return w[:i] + w[i] + w[i:]
    k = idx[int(rng.integers(len(idx)))]
    return w[:k] + vw[int(rng.integers(len(vw)))] + w[k + 1:]


def encode(text, stoi, G, rng=None, typo_p=0.0, max_txt=64):
    """teks → [BOS] kata.. [A] (slot + GRID + SEP di-append pemanggil).
    OPEN VOCAB: kata yang gak ada di vocab (atau kena typo-aug) → [CHARW] +
    token karakter. Jadi kata NGAWUR pun tetap ngaruh ke gambar — gak dibuang
    diem-diem kayak versi lama (itu yang bikin 'prompt jadi tags')."""
    ids = [BOS]
    cids = char_ids()
    cw = charw_id()
    for w in tok(text)[:MAX_WORDS]:
        ww = w
        if rng is not None and typo_p and len(ww) >= 3 and rng.random() < typo_p:
            ww = _mangle(ww, rng)
        i = stoi.get(ww)
        if i is not None and i >= WORD_BASE:
            ids.append(i)
        else:
            ids.append(cw)
            for ch in ww:
                cid = cids.get(ch)
                if cid is not None:
                    ids.append(cid)
        if len(ids) >= max_txt:
            break
    ids.append(A_TOK)
    return ids


def uncond_tokens(G):
    return [BOS, UNCOND, TG.GRID_TOKEN[G], SEP]


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for i in range(8):
        txt, cond = make_sample(rng)
        s = cond.get("subj")
        print(f"{txt!r}\n   scene={cond['scene']} mood={cond.get('mood')} "
              f"subj={s['id'] if s else None}+{s['attrs'] if s else []} col={s['col'] if s else None}\n")
    itos, nw = build_vocab()
    print("vocab:", len(itos), f"({nw} kata) | contoh:", [w for w in itos[193:200]])
    stoi = {w: i for i, w in enumerate(itos) if w}
    demo = ["ikan terbang di volkano", "ikan terbang di volcano", "kucing ninja naik roket"]
    for d in demo:
        ids = encode(d, stoi, 16)
        dec = [itos[t] if itos[t] else (f"char:{CHARS[t - char_base(nw)]}" if char_base(nw) <= t < char_base(nw) + 36 else f"#{t}") for t in ids]
        print(f"  {d!r} -> {dec}")
