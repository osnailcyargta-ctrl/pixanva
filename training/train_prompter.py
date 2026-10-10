"""Pixanva Prompter trainer — teks bebas → slot tag (scene/color/mood/orn×2).
Decoder-only sama persis dgn model.py; posisi teks (i, 0) kayak chat.
Chunked & resumable (--budget detik per chunk). Ukuran: 5m / 12m / 17m.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import model as M
import optim as OP
import prompt_data as PD

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

# ---- arsitektur per tahap (target param ±10%) ----
ARCHS = {
    "5m":  dict(d=256, L=6, H=8, mlp=896,  lr=3e-4,   steps=900, VOCAB_WORDS=2400),
    "12m": dict(d=384, L=8, H=6, mlp=1024, lr=2.6e-4, steps=700, VOCAB_WORDS=2400),
    "17m": dict(d=448, L=8, H=7, mlp=1280, lr=2.4e-4, steps=600, VOCAB_WORDS=2400),
}
BATCH = 24
GSCALE = 8192
VAL_SEED = 321
VAL_N = 384
N_SLOTS = 6  # scene, color, mood, orn1, orn2, eos


def build_vocab(arch):
    """Kumpulin frekuensi kata dari sample deterministik → vocab kata teratas."""
    rng = np.random.default_rng(VAL_SEED)
    from collections import Counter
    cnt = Counter()
    for _ in range(6000):
        txt, _ = PD.make_sample(rng)
        cnt.update(PD.tok(txt))
    words = [w for w, _c in cnt.most_common(arch["VOCAB_WORDS"])]
    itos = list(PD.SPECIALS) + list(PD.CLASS_TOKENS) + words
    return itos


def load_all():
    itos = build_vocab(ARCH)
    stoi = {w: i for i, w in enumerate(itos)}
    rng = np.random.default_rng(VAL_SEED)
    val = [PD.make_sample(rng) for _ in range(VAL_N)]
    return itos, stoi, val


def encode_sample(txt, cond, stoi):
    ids, slots = PD.encode(txt, cond, stoi)
    return ids, slots


def make_batch(rng, stoi, B, train=True):
    data_rng = rng if train else np.random.default_rng(VAL_SEED)
    chunk = [PD.make_sample(data_rng) for _ in range(B)]
    rows, tgts = [], []
    T = 0
    for txt, cond in chunk:
        ids, slots = encode_sample(txt, cond, stoi)
        full = ids + slots
        rows.append(full)
        tgts.append((len(ids) - 1, slots))
        T = max(T, len(full))
    ids_a = np.zeros((B, T), dtype=np.int64)
    tgt = np.full((B, T), -1, dtype=np.int64)
    for r, (full, (a_pos, slots)) in enumerate(zip(rows, tgts)):
        ids_a[r, :len(full)] = full
        for i, s in enumerate(slots):
            tgt[r, a_pos + i] = s
    rope = M.rope_tables(T, 0, CFG)
    return ids_a, tgt, rope


def slot_accuracy(P, val, stoi, itos, mb=12):
    """Akurasi per-slot (argmax) — metrik utama kualitas prompter."""
    hits = np.zeros(N_SLOTS)
    tot = 0
    none_id = stoi["<none>"]
    for s in range(0, len(val), mb):
        chunk = val[s:s + mb]
        T = max(len(PD.encode(t, c, stoi)[0]) + N_SLOTS for t, c in chunk)
        ids_a = np.zeros((len(chunk), T), dtype=np.int64)
        tgt = np.full((len(chunk), T, N_SLOTS), -1, dtype=np.int64)
        for r, (t, c) in enumerate(chunk):
            ids, slots = PD.encode(t, c, stoi)
            ids_a[r, :len(ids) + N_SLOTS] = ids + slots
            for i, s2 in enumerate(slots):
                tgt[r, len(ids) - 1 + i, i] = s2
        rope = M.rope_tables(T, 0, CFG)
        for st in range(0, len(chunk), 4):
            e = min(st + 4, len(chunk))
            logits = M.forward(P, CFG, ids_a[st:e], rope)
            for r in range(e - st):
                for i in range(N_SLOTS):
                    want = tgt[st + r, :, i]
                    pos = int(np.where(want >= 0)[0][0])
                    pred = int(np.argmax(logits[r, pos]))
                    if pred == want[pos]:
                        hits[i] += 1
                tot += 1
    acc = hits / max(tot, 1)
    return acc


CFG = None


def main():
    global CFG, ARCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(ARCHS))
    ap.add_argument("--budget", type=float, default=420)
    ap.add_argument("--override-steps", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    ARCH = ARCHS[args.size]
    itos, stoi, val = load_all()
    CFG = {"V": len(itos), "d": ARCH["d"], "L": ARCH["L"], "H": ARCH["H"],
           "mlp": ARCH["mlp"], "theta": 1000.0}
    n_par = M.n_params(CFG)

    state_dir = os.path.join(ROOT, "state")
    os.makedirs(state_dir, exist_ok=True)
    tag = f"prompter{args.size}"
    ckpt_path = os.path.join(state_dir, f"{tag}.pkl")
    log_path = os.path.join(state_dir, f"{tag}.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    if args.eval_only:
        with open(ckpt_path, "rb") as f:
            st = __import__("pickle").load(f)
        acc = slot_accuracy(st["P"], val, stoi, itos)
        log(f"[{tag}] slot acc scene/color/mood/orn1/orn2/eos = "
            + " ".join(f"{a:.3f}" for a in acc[:5]) + f" mean={acc[:5].mean():.3f}")
        return

    if os.path.exists(ckpt_path):
        st = OP.load_ckpt(ckpt_path)
        P, step, drng, hist = st["P"], st["step"], st["data_rng"], st["history"]
        opt = OP.AdamW(P, lr=ARCH["lr"])
        opt.load(st["opt"])
        log(f"[{tag}] resume dari step {step}")
    else:
        prng = np.random.default_rng(888)
        P = M.init_params(CFG, prng)
        opt = OP.AdamW(P, lr=ARCH["lr"])
        step, drng, hist = 0, np.random.default_rng(4242), {"loss": [], "val": []}
        log(f"[{tag}] mulai baru — params={n_par:,} vocab={len(itos)} (kata {ARCH['VOCAB_WORDS']})")

    total_steps = args.override_steps or ARCH["steps"]
    budget_end = time.time() + args.budget
    ema = hist["loss"][-1][2] if hist["loss"] else None
    last_save = 0

    while step < total_steps:
        if time.time() > budget_end - 4:
            break
        lr_scale = OP.cosine_scale(step, total_steps, warmup=60)
        ids, tgt, rope = make_batch(drng, stoi, BATCH)
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
            log(f"[{tag}] step {step} loss {loss:.4f} ema {ema:.4f} lr_scale {lr_scale:.3f}")
        if step % 150 == 0 or step == total_steps:
            acc = slot_accuracy(P, val[:128], stoi, itos)
            hist["val"].append((step, float(acc.mean())))
            log(f"[{tag}] step {step} SLOT-ACC mean {acc.mean():.3f} "
                f"(scene {acc[0]:.2f} color {acc[1]:.2f} mood {acc[2]:.2f} orn {acc[3]:.2f}/{acc[4]:.2f})")
        if step - last_save >= 25 or time.time() > budget_end - 8:
            OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
            last_save = step
    OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
    log(f"[{tag}] chunk selesai di step {step} ema {ema:.4f}")


if __name__ == "__main__":
    main()
