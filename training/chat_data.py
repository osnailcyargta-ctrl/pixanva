"""Pixanva Assistant — corpus + word-level tokenizer (LLM chat dari nol, ±7M params).
Dialog sintetis bahasa gaul: user minta ide prompt → asisten jawab ack + 3 set prompt.
Format jawaban STRICT biar UI bisa parse:
  {ack}
  {topik: "kepikiran {label} ya..."}
  {neg: "oh iya, X gw ilangin"}       (kalau ada)
  nih 3 pilihannya :
  1. {style} = {Label} • {Label} • ...
  2. ...
  3. ...
  {closing}
"""
import sys, os, json, re, random
sys.path.insert(0, os.path.dirname(__file__))
import tags as TG

PAD, BOS, EOS, U, A = 0, 1, 2, 3, 4
NSPEC = 5
SPECIALS = ["<pad>", "<bos>", "<eos>", "<u>", "<a>"]

SCENE_LBL = {s[0]: s[1] for s in TG.SCENES}
COLOR_LBL = {c[0]: c[1] for c in TG.COLORS}
MOOD_LBL = {m[0]: m[1] for m in TG.MOODS}
ORN_LBL = {o[0]: o[1] for o in TG.ORNAMENTS}

# slang user → id tag (subset ringkas; asisten selalu jawab pakai label resmi)
SCENE_SYN = {
    "gunung": ["gunung", "pegunungan", "puncak", "bukit", "mountain"],
    "laut": ["laut", "lautan", "samudra", "ocean"],
    "hutan": ["hutan", "rimba", "jungle", "forest"],
    "kota": ["kota", "urban", "city", "gedung"],
    "gurun": ["gurun", "desert", "pasir gurun"],
    "angkasa": ["angkasa", "galaksi", "luar angkasa", "antariksa"],
    "aurora": ["aurora"],
    "pantai": ["pantai", "beach", "pesisir"],
    "danau": ["danau", "telaga"],
    "sawah": ["sawah", "ladang"],
    "kanjon": ["kanjon", "canyon", "ngarai"],
    "volkano": ["volkano", "gunung api", "vulkan", "lava"],
    "bunga": ["bunga", "ladang bunga", "sakura"],
    "terjun": ["air terjun", "terjun", "curug"],
    "salju": ["salju", "tundra", "snow"],
    "awan": ["lautan awan", "atas awan"],
}
COLOR_SYN = {
    "hangat": ["hangat"], "dingin": ["dingin"], "neon": ["neon", "cyberpunk"],
    "pastel": ["pastel"], "monokrom": ["monokrom", "hitam putih"], "bumi": ["bumi", "earth tone"],
    "tropis": ["tropis"], "gelap": ["gelap", "dark"], "cerah": ["cerah", "terang"],
    "vintage": ["vintage", "retro"], "es": ["es", "ice"], "smaragd": ["smaragd", "emerald"],
}
MOOD_SYN = {
    "pagi": ["pagi", "fajar"], "siang": ["siang"], "senja": ["senja", "sore", "sunset"],
    "malam": ["malam", "midnight"], "kabut": ["kabut", "fog"], "badai": ["badai", "storm"],
    "mystic": ["mystic", "mistis", "epik"], "mimpi": ["mimpi", "dreamy"],
}
ORN_SYN = {
    "bintang": ["bintang"], "bulan": ["bulan"], "matahari": ["matahari"], "awan": ["awan"],
    "burung": ["burung"], "perahu": ["perahu", "kapal"], "balon": ["balon"],
    "kupu": ["kupu"], "petir": ["petir", "kilat"], "pelangi": ["pelangi"],
    "meteor": ["meteor"], "salju": ["salju"], "pohon": ["pohon"], "bunga": ["bunga"],
}

STYLES = [
    ("aman enak",    ["senja", "pagi", "siang", "mimpi"],   ["awan", "burung", "perahu", "pohon", "matahari"]),
    ("dramatis",     ["malam", "badai", "kabut", "mystic"], ["petir", "bulan", "meteor", "bintang"]),
    ("dreamy",       ["mimpi", "mystic", "kabut", "pagi"],  ["pelangi", "kupu", "balon", "bintang"]),
    ("epik",         ["mystic", "badai", "malam"],          ["petir", "meteor", "bulan"]),
    ("kalem",        ["pagi", "kabut", "siang"],            ["perahu", "pohon", "burung"]),
    ("vibrant",      ["siang", "senja"],                    ["pelangi", "balon", "burung"]),
    ("misterius",    ["malam", "kabut", "mystic"],          ["bulan", "bintang", "meteor"]),
    ("hangat klasik",["senja", "pagi"],                     ["matahari", "awan", "burung"]),
    ("cerah segar",  ["siang", "pagi"],                     ["awan", "burung", "pelangi"]),
    ("kelam pekat",  ["malam", "badai"],                    ["petir", "meteor", "bintang"]),
]

ACK = [
    "oke bro, gw racik dulu", "sip, ide bagus, bentar ya", "noted, gw susun 3 opsi",
    "oke, gw gali ide dulu", "bentar, gw racikin", "siap, gw racik 3 varian",
]
TOPIC = [
    "kepikiran {s} ya, gw racik 3 varian",
    "{s} nih, subjek enak, ini 3 varian",
    "sip {s}, gw padu mood dan ornamen buat 3 varian",
]
NEG_LINE = [
    "oh iya, {n} gw ilangin dari semua opsi",
    "siap, tanpa {n} semuanya",
    "oke, {n} gw buang dari opsi",
]
DELIVER = ["nih 3 pilihannya :", "ini 3 set siap pakai :", "3 opsi siap, tinggal pilih :"]
CLOSE = [
    "klik salah satu buat dipasang, kurang sreg bilang lagi",
    "pilih yang paling sreg, mau versi lain tinggal bilang lagi",
    "tinggal pilih aja, gak pas bilang lagi nanti gw racik ulang",
]
NOTOPIC = [
    "bro kurang spesifik nih, tapi nih 3 opsi seru dulu",
    "gw tebak tebak ya, ini 3 opsi acak enak",
]
HELLO = [
    "halo bro, gw asisten prompt pixanva, mau gambar apa nih",
    "hai bro, gw bisa bantu bikin ide prompt, gas apa bro",
]
THX_U = ["oke sip makasih bro", "mantap nih", "oke gw coba dulu", "sip dah"]
THX_A = [
    "sip, semoga hasilnya keren. panggil aja kalau butuh ide lagi",
    "gas bro, kalau mau ide lain tinggal chat lagi",
]
REASK_U = ["lagi dong versi lain", "kurang sreg, ganti yang lain", "racik ulang dong", "versi lain deh"]
NEGADD_U = ["jangan {n}", "oh iya jangan ada {n}", "tanpa {n} ya"]
ASK_TPL = [
    "gw mau bikin {s} enaknya gimana tag-nya?",
    "bro kasih ide buat {s} dong",
    "mau gambar {s}, tag enak apa?",
    "ide prompt {s} dong bro",
    "{s} enaknya gimana bro?",
    "bikin {s} dong bro",
]
ASK_M = ["kasih {s} yang {m} dong", "{s} {m} gimana tag-nya?", "gw mau {s} yang {m}, ide prompt?"]
ASK_C = ["{s} warna {c} enaknya gimana?", "ide prompt {s} {c} dong"]
SEP = " sep "


def tok(s):
    return re.findall(r"[a-z0-9]+|[•=.,:?!-]", s.lower())


def pick_syn(rng, syn):
    wid = rng.choice(list(syn.keys()))
    return wid, rng.choice(syn[wid])


def join_labels(parts):
    return " • ".join(parts)


def make_sets(rng, scene, color, mood, negs, n=3):
    """3 set gaya beda → list of (style, [labels])."""
    neg = set(negs)
    used = []
    idxs = list(range(len(STYLES)))
    rng.shuffle(idxs)
    out = []
    for si in idxs:
        if len(out) >= n:
            break
        name, mp, op = STYLES[si]
        if name in used:
            continue
        sc = scene if scene else rng.choice(TG.SCENES)[0]
        labels = [SCENE_LBL[sc]]
        co = color if (color and not neg & {color}) else rng.choice(
            [c for c in TG.COLORS if c[0] not in neg])[0]
        labels.append(COLOR_LBL[co])
        mo = None
        if mood and mood not in neg:
            mo = mood
        else:
            cands = [m for m in mp if m not in neg]
            mo = rng.choice(cands) if cands else rng.choice(
                [m for m in TG.MOODS if m[0] not in neg])[0]
        labels.append(MOOD_LBL[mo])
        orn_pool = [o for o in (list(op) + [o[0] for o in TG.ORNAMENTS])
                    if o not in neg and o != sc]
        rng.shuffle(orn_pool)
        n_orn = rng.choice([0, 1, 1, 2])
        for o in orn_pool[:n_orn]:
            labels.append(ORN_LBL[o])
        out.append((name, labels))
        used.append(name)
    return out


def make_reply(rng, scene=None, color=None, mood=None, negs=None, notopic=False,
               hello=False, thx=False):
    negs = negs or []
    lines = []
    if hello:
        return rng.choice(HELLO)
    if thx:
        return rng.choice(THX_A)
    lines.append(rng.choice(ACK))
    if scene:
        lines.append(rng.choice(TOPIC).format(s=SCENE_LBL[scene].lower()))
    if negs:
        lines.append(rng.choice(NEG_LINE).format(n=", ".join(
            (SCENE_LBL.get(x) or COLOR_LBL.get(x) or MOOD_LBL.get(x) or ORN_LBL.get(x, x)).lower()
            for x in negs)))
    if notopic:
        lines.append(rng.choice(NOTOPIC))
    lines.append(rng.choice(DELIVER))
    sets = make_sets(rng, scene, color, mood, negs)
    for i, (name, labels) in enumerate(sets):
        lines.append(f"{i + 1}. {name} = {join_labels(labels)}")
    lines.append(rng.choice(CLOSE))
    return "\n".join(lines)


def gen_corpus(n=46000, seed=7):
    rng = random.Random(seed)
    out, seen = [], set()

    def add(u, a):
        key = hash((u, a))
        if key in seen:
            return False
        seen.add(key)
        out.append({"u": u, "a": a})
        return True

    while len(out) < n:
        r = rng.random()
        sid, slang = pick_syn(rng, SCENE_SYN)
        mid, mslang = pick_syn(rng, MOOD_SYN)
        cid, cslang = pick_syn(rng, COLOR_SYN)
        nid, nslang = pick_syn(rng, rng.choice([ORN_SYN, MOOD_SYN, ORN_SYN]))
        if r < 0.40:  # ide murni
            u = rng.choice(ASK_TPL).format(s=slang)
            add(u, make_reply(rng, scene=sid))
        elif r < 0.52:  # ide + mood
            u = rng.choice(ASK_M).format(s=slang, m=mslang)
            add(u, make_reply(rng, scene=sid, mood=mid))
        elif r < 0.62:  # ide + warna
            u = rng.choice(ASK_C).format(s=slang, c=cslang)
            add(u, make_reply(rng, scene=sid, color=cid))
        elif r < 0.74:  # ide + negative
            u = rng.choice(ASK_TPL).format(s=slang) + " jangan " + nslang
            add(u, make_reply(rng, scene=sid, negs=[nid]))
        elif r < 0.84:  # reask (konteks 2 turn)
            prev = rng.choice(ASK_TPL).format(s=slang)
            u = prev + SEP + rng.choice(REASK_U)
            add(u, make_reply(rng, scene=sid))
        elif r < 0.90:  # neg modify (konteks 2 turn)
            prev = rng.choice(ASK_TPL).format(s=slang)
            u = prev + SEP + rng.choice(NEGADD_U).format(n=nslang)
            add(u, make_reply(rng, scene=sid, negs=[nid]))
        elif r < 0.95:  # sapaan / help
            u = rng.choice(["halo", "hai bro", "halo bro", "bisa apa aja?", "halo, bisa bantu?"])
            add(u, make_reply(rng, hello=True))
        elif r < 0.98:  # makasih
            add(rng.choice(THX_U), make_reply(rng, thx=True))
        else:  # tanpa topik
            u = rng.choice(["gw bingung mau gambar apa", "kasih ide random dong bro",
                            "kasih apa aja deh yang enak"])
            add(u, make_reply(rng, notopic=True))
    return out


def build_vocab(samples, min_freq=2, cap=3000):
    from collections import Counter
    cnt = Counter()
    for s in samples:
        cnt.update(tok(s["u"]))
        cnt.update(tok(s["a"]))
    words = [w for w, c in cnt.most_common(cap) if c >= min_freq]
    itos = SPECIALS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode(u, a, stoi):
    us, as_ = tok(u), tok(a)
    ids = [BOS, U] + [stoi.get(w, PAD) for w in us] + [A] + \
          [stoi.get(w, PAD) for w in as_] + [EOS]
    sup_from = 3 + len(us)  # index token asisten pertama
    return ids, sup_from


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, "..")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 46000
    samples = gen_corpus(n)
    itos, stoi = build_vocab(samples)
    lens = []
    for s in samples:
        ids, _ = encode(s["u"], s["a"], stoi)
        lens.append(len(ids))
    lens.sort()
    print(f"samples={len(samples)} vocab={len(itos)} "
          f"len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")
    with open(os.path.join(base, "chat_samples.json"), "w") as f:
        json.dump(samples, f, ensure_ascii=False)
    with open(os.path.join(base, "chat_vocab.json"), "w") as f:
        json.dump({"itos": itos, "specials": SPECIALS}, f, ensure_ascii=False)
    print("chat_samples.json + chat_vocab.json ditulis")
    for s in samples[:3]:
        print("--- USER:", s["u"])
        print(s["a"])
