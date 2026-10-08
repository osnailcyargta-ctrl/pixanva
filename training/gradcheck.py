"""Gradcheck numeric utk transformer from-scratch — WAJIB 100% pass sebelum training."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M

rng = np.random.default_rng(0)
cfg = {"V": 24, "d": 16, "H": 2, "L": 2, "mlp": 32, "theta": 100.0}
P = M.init_params(cfg, rng, dtype=np.float64)
n_cond, G = 3, 2
B = 2
rope = M.rope_tables(n_cond, G, cfg)
T = n_cond + G * G
ids = rng.integers(0, cfg["V"], size=(B, T))
targets = rng.integers(0, cfg["V"], size=(B, T))
targets[:, :n_cond] = -1  # cond gak dihitung loss
targets[1, -2:] = -1


def compute_loss(P):
    logits = M.forward(P, cfg, ids, rope)
    loss, _ = M.loss_and_dlogits(logits, targets)
    return loss


logits, cache = M.forward(P, cfg, ids, rope, record=True)
loss, dlogits = M.loss_and_dlogits(logits, targets)
G_ = M.backward(P, cfg, cache, dlogits)
M.count_grads_match(P, G_)

# pilih sampel param dari SEMUA key
keys = sorted(P.keys())
idxs = []
for k in keys:
    flat = P[k].size
    take = min(4, flat)
    idxs += [(k, int(i)) for i in rng.choice(flat, size=take, replace=False)]

h = 1e-5
bad = 0
for k, i in idxs:
    a = G_[k].reshape(-1)[i]
    orig = P[k].reshape(-1)[i]
    P[k].reshape(-1)[i] = orig + h
    lp = compute_loss(P)
    P[k].reshape(-1)[i] = orig - h
    lm = compute_loss(P)
    P[k].reshape(-1)[i] = orig
    n = (lp - lm) / (2 * h)
    # kriteria standar: atol + rtol (gradient kecil banget didominasi noise float)
    ok = abs(a - n) <= 1e-7 + 1e-4 * max(abs(a), abs(n))
    if not ok:
        bad += 1
        print(f"FAIL {k}[{i}] analytic={a:.8f} numeric={n:.8f} rel={abs(a-n)/max(abs(a),abs(n),1e-8):.2e}")
print(f"{len(idxs) - bad}/{len(idxs)} pass")
assert bad == 0, "GRADCHECK GAGAL"
print("GRADCHECK OK")
