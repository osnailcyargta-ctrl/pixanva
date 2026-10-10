"""Profile waktu per-step train_imajin per arsitektur & bucket G — kalibrasi jumlah langkah."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import optim as OP
import train_imajin as T

ITOS = T.load_vocab()
T.STOI = {w: i for i, w in enumerate(ITOS) if w}

CASES = [
    ("5m", dict(d=224, L=8, H=7, mlp=896), [(16, 10), (32, 3), (48, 1)]),
    ("12m", dict(d=320, L=9, H=8, mlp=1280), [(16, 6), (32, 2), (48, 1)]),
    ("17m", dict(d=352, L=10, H=8, mlp=1600), [(16, 5), (32, 2), (48, 1)]),
]

for name, cfg, buckets in CASES:
    cfg = {"V": len(ITOS), **cfg, "theta": 1000.0}
    P = M.init_params(cfg, np.random.default_rng(0))
    opt = OP.AdamW(P, lr=2.5e-4)
    rng = np.random.default_rng(7)
    for G, B in buckets:
        ids, tgt = T.make_batch(rng, G, B, 123)
        rope = M.rope_tables(ids.shape[1] - G * G, G, cfg)
        # warmup 1
        logits, cache = M.forward(P, cfg, ids, rope, record=True)
        loss, dl = M.loss_and_dlogits(logits, tgt)
        dl *= T.GSCALE
        G_ = M.backward(P, cfg, cache, dl)
        opt.step(P, G_, 1.0)
        # timed 2
        t0 = time.time()
        for _ in range(2):
            logits, cache = M.forward(P, cfg, ids, rope, record=True)
            loss, dl = M.loss_and_dlogits(logits, tgt)
            dl *= T.GSCALE
            G_ = M.backward(P, cfg, cache, dl)
            opt.step(P, G_, 1.0)
        dt = (time.time() - t0) / 2
        print(f"{name} G{G} B{B}: {dt:.2f}s/step", flush=True)
        del cache, G_, logits
