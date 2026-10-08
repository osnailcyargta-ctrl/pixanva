"""Pixanva transformer — decoder-only GPT, from scratch, numpy murni.
- RoPE 2D faktoris (sumbu-x = baris grid, sumbu-y = kolom) -> 1 model utk semua resolusi
- Kondisioning: tag jadi prefix tokens
- Backward pass diturunin MANUAL (gak ada autograd/framework)
"""
import numpy as np

EPS = 1e-5
SQ2PI = np.sqrt(2 / np.pi)


# ---------------- params ----------------

def param_shapes(cfg):
    V, d, L, m = cfg["V"], cfg["d"], cfg["L"], cfg["mlp"]
    S = {"emb": (V, d), "lnf.g": (d,), "lnf.b": (d,)}
    for l in range(L):
        p = f"l{l}."
        S.update({
            p + "ln1.g": (d,), p + "ln1.b": (d,),
            p + "wqkv": (d, 3 * d), p + "bqkv": (3 * d,),
            p + "wo": (d, d), p + "bo": (d,),
            p + "ln2.g": (d,), p + "ln2.b": (d,),
            p + "w1": (d, m), p + "b1": (m,),
            p + "w2": (m, d), p + "b2": (d,),
        })
    return S


def init_params(cfg, rng, dtype=np.float32):
    S = param_shapes(cfg)
    P = {}
    L = cfg["L"]
    for name, sh in S.items():
        if name == "emb":
            P[name] = (rng.standard_normal(sh) * 0.02).astype(dtype)
        elif name.endswith(("wqkv", "w1")):
            P[name] = (rng.standard_normal(sh) * 0.02).astype(dtype)
        elif name.endswith(("wo", "w2")):
            scale = 0.02 / np.sqrt(2 * L)
            P[name] = (rng.standard_normal(sh) * scale).astype(dtype)
        elif name.endswith(".g"):
            P[name] = np.ones(sh, dtype=dtype)
        else:
            P[name] = np.zeros(sh, dtype=dtype)
    return P


def n_params(cfg):
    return sum(int(np.prod(s)) for s in param_shapes(cfg).values())


# ---------------- rope ----------------

def rope_tables(n_cond, G, cfg):
    """Token i: cond -> (x=i, y=0); image (r,c) -> (x=r, y=c+16).
    Return (cos_x, sin_x, cos_y, sin_y) per blok dimensi head."""
    hd = cfg["d"] // cfg["H"]
    bx = hd // 2
    by = hd - bx
    OFF = 16
    T = n_cond + G * G
    xs = np.empty(T, dtype=np.int64)
    ys = np.empty(T, dtype=np.int64)
    xs[:n_cond] = np.arange(n_cond)
    ys[:n_cond] = 0
    r = np.arange(G * G) // G
    c = np.arange(G * G) % G
    xs[n_cond:] = r
    ys[n_cond:] = c + OFF
    theta = cfg.get("theta", 1000.0)
    fx = theta ** (-2 * np.arange(bx // 2) / bx)
    fy = theta ** (-2 * np.arange(by // 2) / by)
    ax = xs[:, None] * fx[None, :]
    ay = ys[:, None] * fy[None, :]
    return (np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), bx)


def rot1d(v, cos, sin):
    """v: (B,H,T,n); cos/sin: (T, n/2) -> RoPE dalam blok."""
    n2 = cos.shape[-1]
    v1, v2 = v[..., :n2], v[..., n2:]
    cx = cos[None, None, :, :]
    sx = sin[None, None, :, :]
    return np.concatenate([v1 * cx - v2 * sx, v1 * sx + v2 * cx], -1)


def apply_rope(x, rope):
    _, _, _, _, bx = rope
    cax, sax, cay, say = rope[0], rope[1], rope[2], rope[3]
    return np.concatenate([rot1d(x[..., :bx], cax, sax), rot1d(x[..., bx:], cay, say)], -1)


def apply_rope_inv(x, rope):
    _, sax, _, say, bx = rope
    cax, cay = rope[0], rope[2]
    return np.concatenate([rot1d(x[..., :bx], cax, -sax), rot1d(x[..., bx:], cay, -say)], -1)


# ---------------- forward ----------------

def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    sigma = np.sqrt(var + EPS)
    xhat = (x - mu) / sigma
    return xhat * g + b, xhat, sigma


def gelu(x):
    # 1 alokasi, sisanya in-place (x^3 via perkalian — np.power scalar & lambat)
    t = x * x
    t *= x
    t *= 0.044715
    t += x
    t *= SQ2PI
    np.tanh(t, out=t)
    t += 1.0
    t *= x
    t *= 0.5
    return t


def gelu_b(x, dy):
    # dz: dy dimodifikasi in-place (dy cuma dipakai di sini)
    x2 = x * x
    t = x2 * x
    t *= 0.044715
    t += x
    t *= SQ2PI
    np.tanh(t, out=t)          # t = tanh(u)
    a2 = t * t
    a2 *= -1.0
    a2 += 1.0                  # 1 - t^2
    a2 *= x
    a2 *= SQ2PI * (1 + 3 * 0.044715 * x2)
    a2 *= 0.5
    t += 1.0
    t *= 0.5
    t += a2                    # grad faktor
    dy *= t
    return dy


def ln_input_bwd(dxhat, xhat, sigma):
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    dxhat -= m1
    dxhat -= xhat * m2
    dxhat /= sigma
    return dxhat


def forward(P, cfg, ids, rope, record=False):
    B, T = ids.shape
    d, H, L = cfg["d"], cfg["H"], cfg["L"]
    hd = d // H

    x = P["emb"][ids]
    cache = {"ids": ids, "rope": rope, "layers": []}
    mask = np.triu(np.full((T, T), -1e9, dtype=x.dtype), k=1)
    scale = 1.0 / np.sqrt(hd)

    for l in range(L):
        p = f"l{l}."
        x_in = x
        h1, xhat1, sg1 = layernorm(x, P[p + "ln1.g"], P[p + "ln1.b"])
        qkv = h1 @ P[p + "wqkv"] + P[p + "bqkv"]
        q, k, v = np.split(qkv, 3, -1)
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        qr = apply_rope(q, rope)
        kr = apply_rope(k, rope)
        att = qr @ kr.transpose(0, 1, 3, 2)
        att *= scale
        att += mask
        att -= att.max(-1, keepdims=True)
        np.exp(att, out=att)
        att += 1e-24              # anti-denormal (floor cukup tinggi biar produk backward aman)
        att /= att.sum(-1, keepdims=True)
        Pm = att
        o = (Pm @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        del att
        proj = o @ P[p + "wo"] + P[p + "bo"]
        x_mid = x_in + proj
        h2, xhat2, sg2 = layernorm(x_mid, P[p + "ln2.g"], P[p + "ln2.b"])
        z = h2 @ P[p + "w1"] + P[p + "b1"]
        a = gelu(z)
        mlp = a @ P[p + "w2"] + P[p + "b2"]
        x = x_mid + mlp
        if record:
            # Pm (B,H,T,T) raksasa saat G48 — simpan fp16 biar gak swap-thrash (RAM 3GB)
            pm_c = Pm.astype(np.float16) if T > 1024 else Pm
            cache["layers"].append(dict(x_in=x_in, x_mid=x_mid, xhat1=xhat1,
                                        sg1=sg1, qr=qr, kr=kr, v=v, Pm=pm_c, o=o,
                                        xhat2=xhat2, sg2=sg2, z=z, a=a))
            del Pm, pm_c
    hf, xhatf, sgf = layernorm(x, P["lnf.g"], P["lnf.b"])
    logits = hf @ P["emb"].T
    if record:
        cache["hf"] = hf
        cache["xhatf"] = xhatf
        cache["sgf"] = sgf
        cache["x_last"] = x
        return logits, cache
    return logits


# ---------------- loss ----------------

def loss_and_dlogits(logits, targets, _eye_cache={}):
    """targets: (B,T), -1 = skip. Return (loss scalar, dlogits)."""
    B, T, V = logits.shape
    m = targets >= 0
    z = logits - logits.max(-1, keepdims=True)
    el = np.exp(z)
    sm = el / el.sum(-1, keepdims=True)
    sm += 1e-26              # anti-denormal sebelum operasi lanjutan
    tgt = np.clip(targets, 0, V - 1)
    ll = np.take_along_axis(z, tgt[..., None], -1)[..., 0] \
        - np.log(el.sum(-1, keepdims=True))[..., 0]
    n = max(int(m.sum()), 1)
    loss = -ll[m].sum() / n
    key = (V, logits.dtype.type)
    if key not in _eye_cache:
        _eye_cache[key] = np.eye(V, dtype=logits.dtype)
    dlogits = sm
    dlogits[m] -= _eye_cache[key][tgt[m]]
    dlogits[~m] = 0
    dlogits /= n
    dlogits += 1e-30
    return loss, dlogits


# ---------------- backward ----------------

def backward(P, cfg, cache, dlogits):
    B, T = cache["ids"].shape
    d, H, L = cfg["d"], cfg["H"], cfg["L"]
    hd = d // H
    scale = 1.0 / np.sqrt(hd)
    G_ = {}

    hf, xhatf, x_last = cache["hf"], cache["xhatf"], cache["x_last"]
    dhf = dlogits @ P["emb"]  # (B,T,d) — grad lewat tied head
    G_["lnf.g"] = (dhf * xhatf).sum((0, 1))
    G_["lnf.b"] = dhf.sum((0, 1))
    G_emb = dlogits.transpose(2, 0, 1).reshape(cfg["V"], -1) @ hf.reshape(-1, d)
    dx = ln_input_bwd(dhf * P["lnf.g"], xhatf, cache["sgf"])

    for l in reversed(range(L)):
        p = f"l{l}."
        rec = cache["layers"][l]
        # ---- MLP branch (x_out = x_mid + mlp) ----
        dmlp = dx
        G_[p + "b2"] = dmlp.sum((0, 1))
        G_[p + "w2"] = rec["a"].reshape(-1, rec["a"].shape[-1]).T @ dmlp.reshape(-1, d)
        da = dmlp @ P[p + "w2"].T
        dz = gelu_b(rec["z"], da)
        G_[p + "b1"] = dz.sum((0, 1))
        h2 = rec["xhat2"] * P[p + "ln2.g"] + P[p + "ln2.b"]
        G_[p + "w1"] = h2.reshape(-1, d).T @ dz.reshape(-1, cfg["mlp"])
        dxhat2 = dz @ P[p + "w1"].T
        G_[p + "ln2.g"] = (dxhat2 * rec["xhat2"]).sum((0, 1))
        G_[p + "ln2.b"] = dxhat2.sum((0, 1))
        d_xmid_mlp = ln_input_bwd(dxhat2 * P[p + "ln2.g"], rec["xhat2"], rec["sg2"])
        dx_mid = d_xmid_mlp + dmlp  # residual
        # ---- attention branch (x_mid = x_in + proj) ----
        dproj = dx_mid
        G_[p + "bo"] = dproj.sum((0, 1))
        G_[p + "wo"] = rec["o"].reshape(-1, d).T @ dproj.reshape(-1, d)
        do = dproj @ P[p + "wo"].T
        do4 = do.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        Pm = rec["Pm"].astype(np.float32) if rec["Pm"].dtype == np.float16 else rec["Pm"]
        dv = Pm.transpose(0, 1, 3, 2) @ do4
        dPm = do4 @ rec["v"].transpose(0, 1, 3, 2)
        tmp = dPm * Pm
        dPm -= tmp.sum(-1, keepdims=True)
        dPm *= Pm               # dS (in-place)
        dqr = dPm @ rec["kr"] * scale
        dkr = dPm.transpose(0, 1, 3, 2) @ rec["qr"] * scale
        dq = apply_rope_inv(dqr, cache["rope"])
        dk = apply_rope_inv(dkr, cache["rope"])
        dqkv = np.concatenate([
            dq.transpose(0, 2, 1, 3).reshape(B, T, d),
            dk.transpose(0, 2, 1, 3).reshape(B, T, d),
            dv.transpose(0, 2, 1, 3).reshape(B, T, d)], -1)
        G_[p + "bqkv"] = dqkv.sum((0, 1))
        h1 = rec["xhat1"] * P[p + "ln1.g"] + P[p + "ln1.b"]
        G_[p + "wqkv"] = h1.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
        dh1 = dqkv @ P[p + "wqkv"].T
        G_[p + "ln1.g"] = (dh1 * rec["xhat1"]).sum((0, 1))
        G_[p + "ln1.b"] = dh1.sum((0, 1))
        d_xin_attn = ln_input_bwd(dh1 * P[p + "ln1.g"], rec["xhat1"], rec["sg1"])
        dx_total = d_xin_attn + dproj  # residual
        dx = dx_total
        cache["layers"][l] = None

    G_["emb"] = G_emb
    ids = cache["ids"].reshape(-1)
    np.add.at(G_["emb"], ids, dx.reshape(-1, d))
    return G_


def count_grads_match(P, G_):
    assert set(P.keys()) == set(G_.keys()), (set(P) ^ set(G_))
    for k in P:
        assert P[k].shape == G_[k].shape, (k, P[k].shape, G_[k].shape)
