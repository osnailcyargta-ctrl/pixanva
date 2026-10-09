"""Tes sampling cepat assistant (mirror worker.js chatGenerate) dari state/chat.pkl."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import model as M
import chat_data as CD

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

st = pickle.load(open(os.path.join(ROOT, "state", "chat.pkl"), "rb"))
P = st["P"]
print(f"[tes] step {st['step']}, history val: {[v for v in st['history']['val'][-4:]]}")

vj = json.load(open(os.path.join(BASE, "chat_vocab.json")))
itos, stoi, S = vj["itos"], {w: i for i, w in enumerate(vj["itos"])}, vj["specials"]
cfg = dict(P["cfg"]) if "cfg" in P else None

# — rebuild model wrapper dari bobot —
import train_chat as TC
cfg = TC.CFG
cfg["V"] = len(itos)

def sample(prompt, temp=0.82, topk=24, seed=7):
    rng = np.random.default_rng(seed)
    uw = CD.tok(prompt)
    ids = [S["BOS"], S["U"]] + [stoi[w] for w in uw if w in stoi] + [S["A"]]
    T = len(ids) + 130
    rope = M.rope_tables(T, 0, cfg)
    stt = M.new_state(cfg, T)
    pos = 0
    for t in ids:
        logits = M.step(P, stt, t, pos, rope)
        pos += 1
    out = []
    recent = []
    for i in range(130):
        tok = M.sample(logits, temp, topk, rng, recent, 1.12)
        if tok == S["EOS"]:
            break
        recent.append(tok)
        w = itos[tok]
        if w:
            out.append(w)
        logits = M.step(P, stt, tok, pos, rope)
        pos += 1
    return " ".join(out)

for q in [
    "gw mau bikin gunung enaknya gimana?",
    "pantai senja dong yang enak",
    "kota malam neon yang keren",
    "bebas aja kasih yang bagus",
]:
    print("\n=== USER:", q)
    print(sample(q, seed=11))
