"""Debug slot loss: cek tgt/smask di posisi slot + prediksi top-5 model."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import train_imajin as T

vj = T.ensure_vocab()
T.load_stoi(vj)
T.CFG = {"V": len(vj["itos"]), "d": 224, "L": 8, "H": 7, "mlp": 896, "theta": 1000.0}
itos = vj["itos"]

rng = np.random.default_rng(99)
ids, tgt, smask = T.make_batch(rng, 16, 2, 123)

B, Tx = ids.shape
G = 16
rope = M.rope_tables(Tx - G * G, G, T.CFG)
logits = M.forward(T.CFG and st_P, T.CFG, ids, rope) if False else None

st = pickle.load(open(os.path.join(T.ROOT, "state", "imajin5m.pkl"), "rb"))
P = st["P"]
logits = M.forward(P, T.CFG, ids, rope)

# softmax per posisi
z = logits - logits.max(-1, keepdims=True)
sm = np.exp(z) / np.exp(z).sum(-1, keepdims=True)

def nm(t):
    if t < 0: return "<skip>"
    if t < 13: return f"<spec{t}>"
    if itos[t]: return itos[t]
    if 13 <= t < 109: return f"pal{t-13}"
    return f"#{t}"

for b in range(B):
    row = ids[b]
    k = int(np.where(row == ID_A)[0][0]) if (ID_A := 159) else 0
    print(f"--- baris {b} (L={Tx - G*G}) ---")
    print("ids:", " ".join(nm(int(t)) for t in row[:Tx - G * G]))
    print("tgt:", " ".join(nm(int(t)) for t in tgt[b][:Tx - G * G]))
    print("smask:", "".join("S" if smask[b, p] else "." for p in range(Tx - G * G)))
    for p in range(k, min(k + 12, Tx - G * G)):
        top = np.argsort(-sm[b, p])[:5]
        got = int(tgt[b, p])
        hit = "✓" if top[0] == got else "✗"
        tops = " ".join(f"{nm(int(t))}:{sm[b, p, t]:.2f}" for t in top)
        print(f"  pos {p} ({nm(int(row[p]))}) want {nm(got)} {hit} | {tops}")
