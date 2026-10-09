"""Pixanva training — chunked & resumable (dipanggil berulang dgn --time-budget).
Semua data digenerate on-the-fly dari renderer (infinite data, tag aligned).
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import palette as PA
import tags as TG
import render as RD
import model as M
import optim as OP

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    # nama: (d, L, H, mlp, batch_g16, batch_g32, lr, steps)
    # b48: batch saat finetune G=48 — attention (B,H,T,T) harus muat di RAM 3GB
    # steps utk dark12/heavyqw = TOTAL langkah absolut (warm-restart dari bobot 1.0/QQ)
    "light":  dict(d=128, L=4, H=4, mlp=512, b16=24, b32=8,  lr=4e-4,   steps=800,  theta=1000.0, b48=4),
    "dark":   dict(d=192, L=6, H=6, mlp=768, b16=12, b32=4,  lr=3e-4,   steps=800,  theta=1000.0, b48=1),
    "heavy":  dict(d=256, L=8, H=8, mlp=1024, b16=8, b32=2,  lr=2.6e-4, steps=800,  theta=1000.0, b48=1),
    # rilis 1.2 / QW — arsitektur sama, dilatih lanjutan lebih lama + finetune G48 lebih banyak
    "dark12": dict(d=192, L=6, H=6, mlp=768, b16=12, b32=4,  lr=3e-4,   steps=1230, theta=1000.0, b48=1),
    "heavyqw": dict(d=256, L=8, H=8, mlp=1024, b16=8, b32=2, lr=2.6e-4, steps=1880, theta=1000.0, b48=1),
}

VAL_SEED = 12345
VAL_N = 96
GSCALE = 32768  # loss scaling — jaga gradient jauh dari range denormal float32


def get_cfg(name):
    c = MODELS[name]
    return {"V": TG.VOCAB, "d": c["d"], "L": c["L"], "H": c["H"], "mlp": c["mlp"],
            "theta": c["theta"]}


def make_batch(rng, G, B, model_seed):
    conds = [TG.sample_cond(rng) for _ in range(B)]
    # cond dropout utk CFG
    for c in conds:
        if rng.random() < TG.UNCOND_P:
            c = dict(c)
            c["uncond"] = True
    toks = [TG.cond_tokens(c, G)[0] for c in conds]
    nmax = max(len(t) for t in toks)
    ids = np.zeros((B, nmax + G * G), dtype=np.int64)
    tgt = np.full((B, nmax + G * G), -1, dtype=np.int64)
    for i, t in enumerate(toks):
        ids[i, :len(t)] = t
        img = RD.render(conds[i], G, np.random.default_rng(model_seed + i * 977 + int(rng.integers(1 << 30))))
        img_ids = img.reshape(-1).astype(np.int64) + TG.COLOR_OFFSET
        ids[i, len(t):len(t) + G * G] = img_ids
        # target = geser 1: posisi j memprediksi token j+1
        full = np.concatenate([t, img_ids])
        tgt[i, len(t) - 1:len(t) + G * G - 1] = full[len(t):len(t) + G * G]
    return ids, tgt


def make_val(cfg_name):
    """Val set tetap (deterministik) utk kurva loss."""
    rng = np.random.default_rng(VAL_SEED)
    out = []
    for G in (16, 32):
        conds = []
        for i in range(VAL_N // 2):
            c = TG.sample_cond(rng)
            if rng.random() < 0.1:
                c = dict(c)
                c["uncond"] = True
            conds.append(c)
        B = len(conds)
        toks = [TG.cond_tokens(c, G)[0] for c in conds]
        nmax = max(len(t) for t in toks)
        ids = np.zeros((B, nmax + G * G), dtype=np.int64)
        tgt = np.full((B, nmax + G * G), -1, dtype=np.int64)
        for i, t in enumerate(toks):
            ids[i, :len(t)] = t
            img = RD.render(conds[i], G, np.random.default_rng(VAL_SEED * 31 + G * 1000 + i))
            img_ids = img.reshape(-1).astype(np.int64) + TG.COLOR_OFFSET
            ids[i, len(t):len(t) + G * G] = img_ids
            full = np.concatenate([t, img_ids])
            tgt[i, len(t) - 1:len(t) + G * G - 1] = full[len(t):len(t) + G * G]
        out.append((ids, tgt))
    return out


def eval_val(P, cfg, val, maxB=8):
    tot = 0.0
    cnt = 0
    for ids, tgt in val:
        n = ids.shape[0]
        T = ids.shape[1]
        G = 16 if T < 600 else 32
        ncond = T - G * G
        rope = M.rope_tables(ncond, G, cfg)
        mb = 4 if T > 600 else maxB
        for s in range(0, n, mb):
            e = min(s + mb, n)
            logits = M.forward(P, cfg, ids[s:e], rope)
            l, _ = M.loss_and_dlogits(logits, tgt[s:e])
            tot += float(l) * (e - s)
            cnt += (e - s)
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--budget", type=float, default=500, help="detik per chunk")
    ap.add_argument("--override-steps", type=int, default=0)
    ap.add_argument("--extra-g48", type=int, default=0, help="step tambahan khusus G=48 (finetune)")
    args = ap.parse_args()

    name = args.model
    c = MODELS[name]
    cfg = get_cfg(name)
    total_steps = args.override_steps or c["steps"]
    state_dir = os.path.join(BASE, "..", "state")
    os.makedirs(state_dir, exist_ok=True)
    ckpt_path = os.path.join(state_dir, f"{name}.pkl")
    log_path = os.path.join(state_dir, f"{name}.log")

    if os.path.exists(ckpt_path):
        st = OP.load_ckpt(ckpt_path)
        P, opt, step, drng, hist = st["P"], None, st["step"], st["data_rng"], st["history"]
        opt = OP.AdamW(P, lr=c["lr"])
        opt.load(st["opt"])
        print(f"[{name}] resume dari step {step}", flush=True)
    else:
        rng0 = np.random.default_rng(hash(name) % (2 ** 32))
        P = M.init_params(cfg, rng0)
        opt = OP.AdamW(P, lr=c["lr"])
        step, drng, hist = 0, np.random.default_rng(42), {"loss": [], "val": []}
        print(f"[{name}] mulai baru, params={M.n_params(cfg):,}", flush=True)

    val = make_val(cfg)
    budget_end = time.time() + args.budget
    ema = hist["loss"][-1][2] if hist["loss"] else None
    g48_mode = args.extra_g48 > 0 and step >= total_steps

    last_save = 0
    while step < total_steps or (g48_mode and step < total_steps + args.extra_g48):
        if time.time() > budget_end - 4:
            break
        # pilih bucket: mayoritas G=16, sisanya G=32
        if g48_mode:
            G = 48
            B = c.get("b48", 1)
        else:
            r = drng.random()
            p32 = 0.22 if step > total_steps * 0.25 else 0.08
            G = 32 if r < p32 else 16
            B = c["b32"] if G == 32 else c["b16"]
        lr_scale = OP.cosine_scale(step, total_steps + args.extra_g48)
        ids, tgt = make_batch(drng, G, B, step * 7919)
        rope = M.rope_tables(ids.shape[1] - G * G, G, cfg)
        logits, cache = M.forward(P, cfg, ids, rope, record=True)
        loss, dlogits = M.loss_and_dlogits(logits, tgt)
        dlogits *= GSCALE
        G_ = M.backward(P, cfg, cache, dlogits)
        OP.clip_grads(G_, 1.0 * GSCALE)
        opt.step(P, G_, lr_scale)
        step += 1
        ema = loss if ema is None else 0.95 * ema + 0.05 * loss
        hist["loss"].append((step, float(loss), float(ema)))
        if step % 25 == 0:
            line = f"[{name}] step {step} loss {loss:.4f} ema {ema:.4f} lr_scale {lr_scale:.3f} G{G}"
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if step % 150 == 0 or step == total_steps or (g48_mode and step == total_steps + args.extra_g48):
            vl = eval_val(P, cfg, val)
            hist["val"].append((step, vl))
            line = f"[{name}] step {step} VAL {vl:.4f}"
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if step - last_save >= 25 or time.time() > budget_end - 8:
            OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
            last_save = step
    OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
    print(f"[{name}] chunk selesai di step {step} ema {ema:.4f}", flush=True)


if __name__ == "__main__":
    main()
