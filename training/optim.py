"""AdamW + cosine schedule + checkpoint — semua from scratch."""
import os
import pickle
import time
import numpy as np


class AdamW:
    def __init__(self, params, lr, wd=0.01, b1=0.9, b2=0.95):
        self.lr = lr
        self.wd = wd
        self.b1, self.b2 = b1, b2
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, P, G_, lr_scale=1.0):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        lr = self.lr * lr_scale
        for k, p in P.items():
            g = G_[k]
            if p.ndim > 1 and k != "emb":  # weight decay matrics only
                g = g + self.wd * p
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / bc1
            vh = self.v[k] / bc2
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load(self, st):
        self.m, self.v, self.t = st["m"], st["v"], st["t"]


def cosine_scale(step, total, warmup=60, floor=0.08):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    p = min(p, 1.0)
    return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * p))


def clip_grads(G_, clip=1.0):
    tot = 0.0
    for g in G_.values():
        tot += float((g * g).sum())
    norm = np.sqrt(tot)
    if norm > clip:
        s = clip / (norm + 1e-9)
        for k in G_:
            G_[k] *= s
    return norm


def save_ckpt(path, P, opt, step, data_rng, history):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"P": P, "opt": opt.state(), "step": step,
                     "data_rng": data_rng, "history": history}, f, protocol=4)
    os.replace(tmp, path)


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def time_left(budget):
    return budget is None or time.time() < budget
