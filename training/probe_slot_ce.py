"""Probe CE per posisi slot pada checkpoint aktif — diagnosis presisi."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import train_imajin as T
import imajin_data as ID

vj = T.ensure_vocab()
T.load_stoi(vj)
T.CFG = {"V": len(vj["itos"]), "d": 224, "L": 8, "H": 7, "mlp": 896, "theta": 1000.0}
itos = vj["itos"]

st = pickle.load(open(os.path.join(T.ROOT, "state", "imajin5m.pkl"), "rb"))
P = st["P"]
print("ckpt step:", st["step"])

rng = np.random.default_rng(4242)
N = 48
ce = np.zeros((ID.N_SLOTS + 2,))   # s1..s9, GRID, SEP
cnt = 0
rank_hit = np.zeros((ID.N_SLOTS,))
SLOT_NAMES = ["scene", "color", "mood", "orn1", "orn2", "subj", "attr1", "attr2", "scol", "GRID", "SEP"]
for _ in range(N):
    txt, cond = ID.make_sample(rng)
    pre, slots = T.prefix_tokens(txt, cond, 16, rng=None)
    ids = np.array(pre)[None]
    rope = M.rope_tables(len(pre), 0, T.CFG)
    logits = M.forward(P, T.CFG, ids, rope)
    z = logits[0] - logits[0].max(-1, keepdims=True)
    k = len(pre) - ID.N_SLOTS - 3     # index [A] (A, s1..s9, GRID, SEP)
    for j in range(ID.N_SLOTS + 2):
        t = pre[k + 1 + j]
        zz = z[k + j]
        lz = zz[t] - np.log(np.exp(zz).sum())
        ce[j] -= lz
    # rank scene
    order = np.argsort(-z[k])
    rank_hit[0] += int(np.where(order == pre[k + 1])[0][0] < 3)
    cnt += 1

ce /= cnt
print("CE per posisi slot (randinkatan ~ln(86)≈4.45 kalau ngawur):")
for j, nm_ in enumerate(SLOT_NAMES):
    print(f"  {nm_:6s} {ce[j]:.3f}")
print("scene top-3 hit:", rank_hit[0] / cnt)
