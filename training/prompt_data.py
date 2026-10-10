"""Pixanva Prompter — data on-the-fly: teks bebas gaya manusia → tag slot.
Model custom-prompt: user ketik apa aja ("pantai senja ada perahu dong bro"),
prompter output urutan slot fix: scene → color → mood → orn1 → orn2 → EOS.
Data digenerate prosedural (infinite) — gak perlu file korpus gede.

Format encode (mirror worker.js nanti):
  ids   = [BOS] + kata + [A]
  slots = [SCENE_i, COLOR_i, MOOD_i|NONE, ORN1_i|NONE, ORN2_i|NONE, EOS]
  supervise posisi dari (len(ids)-1) — posisi A memprediksi slot pertama.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
import tags as TG

PAD, BOS, A, EOS, NONE = 0, 1, 2, 3, 4
NSPEC = 5
SPECIALS = ["<pad>", "<bos>", "<a>", "<eos>", "<none>"]

# token kelas: "@" + id tag (mis. "@gunung", "@senja", "@none")
CLASS_TOKENS = (
    ["@" + s[0] for s in TG.SCENES]
    + ["@" + c[0] for c in TG.COLORS]
    + ["@" + m[0] for m in TG.MOODS]
    + ["@" + o[0] for o in TG.ORNAMENTS]
)
SCENE_TOK = {s[0]: "@" + s[0] for s in TG.SCENES}
COLOR_TOK = {c[0]: "@" + c[0] for c in TG.COLORS}
MOOD_TOK = {m[0]: "@" + m[0] for m in TG.MOODS}
ORN_TOK = {o[0]: "@" + o[0] for o in TG.ORNAMENTS}
NONE_TOK = "<none>"

# ---------------- sinonim (subset ringkas + ragam, sejalan dgn SYN di app.js) ----------------
SCENE_SYN = {
    "gunung": ["gunung", "pegunungan", "puncak", "bukit", "mountain", "gunung tinggi", "rinjani", "everest", "highland", "gunung es"],
    "laut": ["laut", "lautan", "samudra", "ocean", "laut dalam", "segara", "deep sea", "palung"],
    "hutan": ["hutan", "rimba", "jungle", "forest", "hutan belantara", "pepohonan", "hutan tropis", "taiga"],
    "kota": ["kota", "urban", "city", "gedung", "downtown", "skyline", "metropolis", "kota malam", "jalan kota"],
    "gurun": ["gurun", "desert", "sahara", "pasiran", "duna", "gurun pasir", "padang pasir"],
    "angkasa": ["angkasa", "galaksi", "galaxy", "planet", "nebula", "antariksa", "luar angkasa", "orbit", "bintang-bintang space"],
    "aurora": ["aurora", "northern lights", "kutub utara", "polar", "aurora borealis"],
    "pantai": ["pantai", "beach", "pesisir", "tebing laut", "tepian laut", "seashore", "tanjung"],
    "danau": ["danau", "telaga", "lake", "situ", "danau gunung"],
    "sawah": ["sawah", "ladang", "rice field", "petak sawah", "perkebunan", "ternak sawah", "rice terrace", "terasering"],
    "kanjon": ["kanjon", "canyon", "ngarai", "jurang", "antelope", "ravine"],
    "volkano": ["volkano", "volcano", "gunung api", "vulkan", "lava", "gunung berapi", "krakatau", "erupsi"],
    "bunga": ["bunga", "ladang bunga", "flower field", "sakura", "tulip", "mawar", "sunflower", "padang bunga", "lavender", "garden"],
    "terjun": ["air terjun", "waterfall", "curug", "coban", "terjun", "grojogan"],
    "salju": ["salju", "snow", "tundra", "es kutub", "arctic", "gunung salju", "bingkai es", "snow mountain"],
    "awan": ["lautan awan", "di atas awan", "cloud sea", "atas awan", "lautan awan pagi", "above the clouds"],
}
COLOR_SYN = {
    "hangat": ["hangat", "warm", "oranye", "keemasan", "golden", "jingga", "amber"],
    "dingin": ["dingin", "cool", "cold", "biru dingin", "sejuk"],
    "neon": ["neon", "cyberpunk", "cyber", "futuristik", "glow", "synthwave", "vaporwave", "bercahaya"],
    "pastel": ["pastel", "lembut", "soft", "candy", "pucat"],
    "monokrom": ["monokrom", "monochrome", "hitam putih", "abu abu", "grayscale", "b w", "noir"],
    "bumi": ["bumi", "earth tone", "coklat", "tanah", "natural", "kayu", "sepia"],
    "tropis": ["tropis", "tropical", "jungle vibe", "bali", "hawaii"],
    "gelap": ["gelap", "dark", "remang", "gothic", "galap", "gelap gelapan"],
    "cerah": ["cerah", "bright", "terang", "vivid", "colorful", "berwarna", "fun"],
    "vintage": ["vintage", "retro", "jadul", "old school", "film look", "analog", "90an"],
    "es": ["es", "ice", "beku", "gletser", "glacier", "es biru", "frozen"],
    "smaragd": ["smaragd", "emerald", "zamrud", "hijau zamrud", "hijau"],
}
MOOD_SYN = {
    "pagi": ["pagi", "morning", "fajar", "sunrise", "subuh", "early morning", "buka hari"],
    "siang": ["siang", "noon", "midday", "terik", "ciang", "tengah hari"],
    "senja": ["senja", "sore", "sunset", "golden hour", "maghrib", "dusk", "petang"],
    "malam": ["malam", "night", "tengah malam", "midnight", "dini hari", "late night", "bermalam"],
    "kabut": ["kabut", "fog", "mist", "berkabut", "misty", "foggy", "berembun"],
    "badai": ["badai", "storm", "hujan", "angin kencang", "stormy", "gerimis", "menghadang badai", "tornado"],
    "mystic": ["mystic", "mistis", "magis", "magic", "fantasy", "gaib", "epik", "epic", "legendaris", "angker"],
    "mimpi": ["mimpi", "dreamy", "dream", "surreal", "halus", "soft mood", "bermimpi"],
}
ORN_SYN = {
    "bintang": ["bintang", "star", "berbintang", "starry", "starry night"],
    "bulan": ["bulan", "moon", "purnama", "bulan sabit", "crescent", "full moon"],
    "matahari": ["matahari", "sun", "matahari terbenam", "sunset sun"],
    "awan": ["awan", "cloud", "mendung", "berawan", "cloudy"],
    "burung": ["burung", "bird", "kawanan burung", "flock", "burung laut"],
    "perahu": ["perahu", "boat", "kapal", "sampan", "perahu layar", "sailboat", "jukung"],
    "balon": ["balon", "balon udara", "hot air balloon", "zeppelin", "balon warna"],
    "kupu": ["kupu", "kupu kupu", "butterfly", "kupu2", "kupu-kupu"],
    "bunga": ["bunga", "flower", "kelopak", "petal"],
    "petir": ["petir", "lightning", "kilat", "halilintar", "thunder", "samber"],
    "pelangi": ["pelangi", "rainbow", "palangi"],
    "meteor": ["meteor", "meteor shower", "hujan meteor", "komet", "comet", "bintang jatuh"],
    "salju": ["hujan salju", "snowing", "salju turun", "snow fall", "bersalju"],
    "pohon": ["pohon", "tree", "cemara", "pine", "pohon cemara", "pinus"],
}

# ---------------- kata sambung / noise gaya ngobrol ----------------
FILLERS = ["dong", "bro", "bang", "yah", "sih", "deh", "banget", "pls", "plis",
           "kayak", "gitu", "aja", "nih", "tuh", "yang keren", "yang estetik",
           "aesthetic", "vibes", "enak", "enaknya", "keren", "bagus", "cantik",
           "indah banget", "gimana", "gmn", "buat wallpaper", "buat pfp", "hd",
           "buat wallpaper hp", "buat dp", "buat profil", "4k", "kayak film",
           "kayak lukisan", "gloomy", "cozy", "tenang", "sepi", "ramai", "hidup",
           "viral", "panorama", "pemandangan", "view", "suasana", "nuansa",
           "atmosphere", "landscape", "scenery", "wajib coba", "auto keren",
           "mantap", "mantul", "gokil", "sange akut", "hh", "hehe", "wkwk",
           "serius", "real", "realistis", "fantasi", "mimpi banget", "deep",
           "sadge", "melankolis", "romantis", "sentimental", "nostalgia"]
OPENERS = ["", "", "", "", "gw mau ", "aku mau ", "bikin ", "bikinin ", "buatkan ",
           "kasih ", "ide prompt ", "prompt ", "gw pengen ", "coba bikin ",
           "make a ", "i want ", "can i get ", "kasih gw ", "request "]
CLOSERS = ["", "", "", "", " dong", " ya", " bro", " deh", " banget", " plis",
           " dong bro", " yang bagus", " yang keren", " yang estetik"]


def _w2(rng, syn):
    """1-2 kata sinonim (kadang frasa 2 kata)."""
    return rng.choice(syn)


def build_text(rng, cond):
    """Bikin kalimat bebas dari kombinasi tag — gaya chat beneran."""
    s_w = _w2(rng, SCENE_SYN[cond["scene"]])
    c_w = _w2(rng, COLOR_SYN[cond["color"]])
    m_w = _w2(rng, MOOD_SYN[cond["mood"]]) if cond.get("mood") else None
    o_ws = [_w2(rng, ORN_SYN[o]) for o in cond.get("orn", [])]

    style = rng.random()
    parts = []
    if style < 0.28:
        # langsung: "pantai senja ada perahu"
        parts.append(s_w)
        if m_w and rng.random() < 0.8:
            parts.append(m_w)
        parts.append(c_w)
        parts.extend(o_ws)
    elif style < 0.52:
        # request: "gw mau pantai yang senja warnanya hangat"
        head = rng.choice(OPENERS) + s_w
        if m_w:
            head += f" yang {m_w}"
        parts.append(head)
        if rng.random() < 0.85:
            parts.append(f"warnanya {c_w}")
        for o in o_ws:
            parts.append(rng.choice([f"ada {o}", f"plus {o}", f"with {o}", o]))
    elif style < 0.72:
        # suasana di depan: "pantai pas senja langitnya hangat"
        t = s_w + (f" pas {m_w}" if m_w else "")
        parts.append(t)
        parts.append(rng.choice([f"langitnya {c_w}", f"warna {c_w}", f"tone {c_w}", c_w]))
        parts.extend([rng.choice([f"ada {o}", o]) for o in o_ws])
    elif style < 0.88:
        # english mix: "pantai senja vibes hangat"
        t = s_w
        if m_w:
            t += f" {m_w}"
        parts.append(t + rng.choice([" vibes", " mood", " aesthetic", ""]))
        parts.append(c_w)
        parts.extend(o_ws)
    else:
        # bertanya: "pantai enaknya gimana? mau yang senja"
        parts.append(f"{s_w} enaknya gimana")
        if m_w:
            parts.append(f"mau yang {m_w}")
        if rng.random() < 0.7:
            parts.append(c_w)
        parts.extend(o_ws)

    txt = " ".join(p for p in parts if p)
    if rng.random() < 0.45:
        txt += rng.choice(CLOSERS)
    if rng.random() < 0.25:
        txt = rng.choice(["eh ", "oi ", "halo, ", "bro ", ""]) + txt
    if rng.random() < 0.2:
        txt += ", " + rng.choice(FILLERS)
    elif rng.random() < 0.3:
        txt += " " + rng.choice(FILLERS)
    return " ".join(txt.split())


def sample_cond(rng):
    cond = TG.sample_cond(rng)
    # mood lebih sering ada (70%) biar model jago nagkep suasana
    if rng.random() < 0.30:
        cond["mood"] = None
    n = rng.integers(0, 4) if hasattr(rng, "integers") else rng.randint(0, 3)
    if not cond["orn"] and n and rng.random() < 0.6:
        import numpy as np
        idx = rng.choice(len(TG.ORNAMENTS), size=int(n), replace=False)
        cond["orn"] = [TG.ORNAMENTS[i][0] for i in idx]
    return cond


def make_sample(rng):
    """Return (text, cond)."""
    for _ in range(20):
        cond = sample_cond(rng)
        # orn max 2 buat slot fix
        cond["orn"] = list(cond["orn"])[:2]
        txt = build_text(rng, cond)
        if 2 <= len(txt.split()) <= 24:
            return txt, cond
    return "pantai senja", {"scene": "pantai", "color": "hangat", "mood": "senja", "orn": []}


# ---------------- tokenizer & vocab ----------------
def tok(text):
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab():
    itos = list(SPECIALS) + list(CLASS_TOKENS)
    return itos  # kata ditambahkan saat training (dari data), tapi kata di corpus
    # tidak perlu vocab statis — train_prompter yang ngumpulin kata


def encode(text, cond, stoi):
    """Return (ids, slot_targets). slot_targets = list 6 token target."""
    ids = [BOS] + [stoi[w] for w in tok(text) if w in stoi] + [A]
    slots = [
        stoi[SCENE_TOK[cond["scene"]]],
        stoi[COLOR_TOK[cond["color"]]],
        stoi[MOOD_TOK[cond["mood"]]] if cond.get("mood") else stoi[NONE_TOK],
        stoi[ORN_TOK[cond["orn"][0]]] if len(cond["orn"]) > 0 else stoi[NONE_TOK],
        stoi[ORN_TOK[cond["orn"][1]]] if len(cond["orn"]) > 1 else stoi[NONE_TOK],
        EOS,
    ]
    return ids, slots


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(7)
    for _ in range(12):
        txt, cond = make_sample(rng)
        print(f"{txt!r}\n   -> {cond}\n")
