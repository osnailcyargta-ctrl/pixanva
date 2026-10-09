"""Pixanva Assistant trainer — text LM decoder-only, from scratch (numpy).
Reuse model.py/optim.py persis (rope text: posisi (i, 0)). Chunked & resumable.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import optim as OP
import chat_data as CD

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

# arsitektur — target ±7M params (vocab ikut corpus, jangan mabok parameter)
ARCH = dict(d=256, L=8, H=8, mlp=1152, lr=3e-4, steps=2600, theta=1000.0)
BATCH = 16
T_CAP = 132          # filter sample lebih panjang dari ini
GSCALE = 8192        # scaling grad (sama strategi dgn train.py image)
VAL_SEED = 123
VAL_N = 320


def load_all():
    with open(os.path.join(BASE, "chat_samples.json")) as f:
        samples = json.load(f)
    with open(os.path.join(BASE, "chat_vocab.json")) as f:
        itos = json.load(f)["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    data = []
    for s in samples:
        ids, sup_from = CD.encode(s["u"], s["a"], stoi)
        if len(ids) <= T_CAP:
            data.append((ids, sup_from))
    return itos, data


def make_batch(rng, data, B):
    idx = rng.integers(0, len(data), size=B)
    chunk = [data[i] for i in idx]
    T = max(len(ids) for ids, _ in chunk)
    ids_a = np.zeros((B, T), dtype=np.int64)
    tgt = np.full((B, T), -1, dtype=np.int64)
    for r, (ids, sf) in enumerate(chunk):
        L_ = len(ids)
        ids_a[r, :L_] = ids
        # target geser 1: posisi j memprediksi token j+1; supervise hanya token asisten
        full = np.array(ids, dtype=np.int64)
        t0, t1 = sf - 1, L_ - 1
        tgt[r, t0:t1] = full[t0 + 1:L_]
    rope = M.rope_tables(T, 0, CFG)
    return ids_a, tgt, rope


def eval_val(P, val, mb=8):
    tot = cnt = 0
    for s in range(0, len(val), mb):
        chunk = val[s:s + mb]
        T = max(len(i) for i, _ in chunk)
        ids_a = np.zeros((len(chunk), T), dtype=np.int64)
        tgt = np.full((len(chunk), T), -1, dtype=np.int64)
        for r, (ids, sf) in enumerate(chunk):
            ids_a[r, :len(ids)] = ids
            full = np.array(ids, dtype=np.int64)
            tgt[r, sf - 1:len(ids) - 1] = full[sf:len(ids)]
        rope = M.rope_tables(T, 0, CFG)
        logits = M.forward(P, CFG, ids_a, rope)
        l, _ = M.loss_and_dlogits(logits, tgt)
        tot += float(l) * len(chunk)
        cnt += len(chunk)
    return tot / max(cnt, 1)


def sample_reply(P, stoi, itos, u_text, max_new=120, temp=0.8, topk=24, seed=1):
    """Sampling cepat buat sanity check lokal."""
    rng = np.random.default_rng(seed)
    us = CD.tok(u_text)
    ids = [CD.BOS, CD.U] + [stoi.get(w, CD.PAD) for w in us] + [CD.A]
    cfg = CFG
    T0 = len(ids)
    st_cap = T0 + max_new + 2
    rope = M.rope_tables(st_cap, 0, cfg)
    # prefill via forward penuh (cukup cepat utk sanity check)
    arr = np.array(ids, dtype=np.int64)[None, :]
    logits = M.forward(P, cfg, arr, rope)[0]
    out = []
    recent = []
    last = ids[-1]
    for t in range(max_new):
        z = logits - logits.max()
        p = np.exp(z) / np.exp(z).sum()
        p[np.argsort(p)[:-topk]] = 0
        p /= p.sum()
        if recent:
            for w in set(recent[-24:]):
                pass
        nid = int(rng.choice(len(p), p=p))
        if nid == CD.EOS:
            break
        out.append(itos[nid])
        recent.append(nid)
        if t + 1 >= max_new:
            break
        # step 1 token: pakai forward kecil (prefill ulang murah di T kecil)
        arr = np.append(arr, [[nid]], axis=1)
        rope = M.rope_tables(arr.shape[1], 0, cfg)
        logits = M.forward(P, cfg, arr, rope)[0, -1]
    return " ".join(out)


CFG = None


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=500, help="detik per chunk")
    ap.add_argument("--override-steps", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="cuma tes sampling dari ckpt")
    args = ap.parse_args()

    itos, data = load_all()
    stoi = {w: i for i, w in enumerate(itos)}
    rng0 = np.random.default_rng(VAL_SEED)
    perm = rng0.permutation(len(data))
    val = [data[i] for i in perm[:VAL_N]]
    train = [data[i] for i in perm[VAL_N:]]

    CFG = {"V": len(itos), "d": ARCH["d"], "L": ARCH["L"], "H": ARCH["H"],
           "mlp": ARCH["mlp"], "theta": ARCH["theta"]}
    n_par = M.n_params(CFG)

    state_dir = os.path.join(ROOT, "state")
    os.makedirs(state_dir, exist_ok=True)
    ckpt_path = os.path.join(state_dir, "chat.pkl")
    log_path = os.path.join(state_dir, "chat.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    if args.sample:
        with open(ckpt_path, "rb") as f:
            st = pickle_load(ckpt_path)
        P = st["P"]
        for u in ["gw mau bikin gunung enaknya gimana tag-nya?",
                  "ide prompt pantai senja dong bro",
                  "bikin kota neon dong bro jangan siang",
                  "halo"]:
            log("PROMPT: " + u)
            log(sample_reply(P, stoi, itos, u, seed=2))
            log("")
        return

    if os.path.exists(ckpt_path):
        st = OP.load_ckpt(ckpt_path)
        P, step, drng, hist = st["P"], st["step"], st["data_rng"], st["history"]
        opt = OP.AdamW(P, lr=ARCH["lr"])
        opt.load(st["opt"])
        log(f"[chat] resume dari step {step}")
    else:
        prng = np.random.default_rng(777)
        P = M.init_params(CFG, prng)
        opt = OP.AdamW(P, lr=ARCH["lr"])
        step, drng, hist = 0, np.random.default_rng(42), {"loss": [], "val": []}
        log(f"[chat] mulai baru — params={n_par:,} vocab={len(itos)} "
            f"train={len(train)} val={len(val)}")

    total_steps = args.override_steps or ARCH["steps"]
    budget_end = time.time() + args.budget
    ema = hist["loss"][-1][2] if hist["loss"] else None
    last_save = 0
    dr = np.random.default_rng(4242)

    while step < total_steps:
        if time.time() > budget_end - 4:
            break
        lr_scale = OP.cosine_scale(step, total_steps, warmup=80)
        ids, tgt, rope = make_batch(dr, train, BATCH)
        logits, cache = M.forward(P, CFG, ids, rope, record=True)
        loss, dlogits = M.loss_and_dlogits(logits, tgt)
        dlogits *= GSCALE
        G_ = M.backward(P, CFG, cache, dlogits)
        OP.clip_grads(G_, 1.0 * GSCALE)
        opt.step(P, G_, lr_scale)
        step += 1
        ema = loss if ema is None else 0.95 * ema + 0.05 * loss
        hist["loss"].append((step, float(loss), float(ema)))
        if step % 25 == 0:
            log(f"[chat] step {step} loss {loss:.4f} ema {ema:.4f} lr_scale {lr_scale:.3f}")
        if step % 200 == 0 or step == total_steps:
            vl = eval_val(P, val)
            hist["val"].append((step, vl))
            log(f"[chat] step {step} VAL {vl:.4f}")
        if step - last_save >= 25 or time.time() > budget_end - 8:
            OP.save_ckpt(ckpt_path, P, opt, step, dr, hist)
            last_save = step
    OP.save_ckpt(ckpt_path, P, opt, step, dr, hist)
    log(f"[chat] chunk selesai di step {step} ema {ema:.4f}")


def pickle_load(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
