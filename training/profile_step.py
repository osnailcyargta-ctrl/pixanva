"""Profile 1 training step — cari hot spot."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import tags as TG, model as M, optim as OP
from train import get_cfg, make_batch

cfg = get_cfg("light")
rng = np.random.default_rng(1)
P = M.init_params(cfg, rng)
opt = OP.AdamW(P, lr=1e-4)

ids, tgt = make_batch(rng, 16, 32, 0)
rope = M.rope_tables(ids.shape[1] - 256, 16, cfg)
print("T =", ids.shape[1])

def tick(label, fn, n=3):
    t = time.time()
    for _ in range(n):
        r = fn()
    dt = (time.time() - t) / n
    print(f"{label}: {dt*1000:.0f} ms")
    return r

logits, cache0 = M.forward(P, cfg, ids, rope, record=True)
loss, dlogits = M.loss_and_dlogits(logits, tgt)
tick("forward", lambda: M.forward(P, cfg, ids, rope, record=True))
def fresh_bwd():
    _, c = M.forward(P, cfg, ids, rope, record=True)
    return M.backward(P, cfg, c, dlogits)

G_ = tick("backward", fresh_bwd)
tick("clip", lambda: OP.clip_grads(G_, 1.0))
tick("opt.step", lambda: opt.step(P, G_))
tick("render+batch", lambda: make_batch(rng, 16, 32, 0))
