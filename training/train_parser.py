"""Pixanva Parser — teks bebas → 9 slot laten (scene,color,mood,orn1,orn2,subj,attr1,attr2,scol).
Arsitektur & resep persis train_prompter (terbukti slot-acc 98%): decoder-only,
posisi (i,0), output token = id kelas (scene 109.. / subj 163.. / NONE 162 / dst).
Slot id space = sama dgn model gambar imajin (tags.py kelas imajin) — nyambung 1:1.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import tags as TG
import imajin_data as ID
import model as M
import optim as OP

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

ARCH = dict(d=256, L=6, H=8, mlp=896, lr=3e-4, steps=900, BATCH=24)
GSCALE = 8192
VAL_SEED = 321
VAL_N = 384


def load_vocab():
    vj_path = os.path.join(ROOT, "models", "imajin_vocab.json")
    with open(vj_path) as f:
        return json.load(f)["itos"]


ITOS = None
STOI = None
CFG = None


def encode_sample(txt, cond):
    ids = ID.encode(txt, STOI, 16)   # [BOS] kata.. (+A_TOK dari encode — dibuang)
    ids = ids[:-1]                   # ganti A dgn SEP sebagai penutup teks
    ids.append(2)                    # SEP = 2
    slots = ID.slot_tokens(cond)
    return ids, slots


def make_batch(rng, B, train=True):
    data_rng = rng if train else np.random.default_rng(VAL_SEED)
    rows, tgts = [], []
    T = 0
    for _ in range(B):
        txt, cond = ID.make_sample(data_rng)
        ids, slots = encode_sample(txt, cond)
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


def slot_accuracy(P, val, mb=12):
    hits = np.zeros(ID.N_SLOTS)
    tot = 0
    for s in range(0, len(val), mb):
        chunk = val[s:s + mb]
        T = max(len(encode_sample(t, c)[0]) + ID.N_SLOTS for t, c in chunk)
        ids_a = np.zeros((len(chunk), T), dtype=np.int64)
        tgt = np.full((len(chunk), T, ID.N_SLOTS), -1, dtype=np.int64)
        for r, (t, c) in enumerate(chunk):
            ids, slots = encode_sample(t, c)
            ids_a[r, :len(ids) + ID.N_SLOTS] = ids + slots
            for i, s2 in enumerate(slots):
                tgt[r, len(ids) - 1 + i, i] = s2
        rope = M.rope_tables(T, 0, CFG)
        for st in range(0, len(chunk), 4):
            e = min(st + 4, len(chunk))
            logits = M.forward(P, CFG, ids_a[st:e], rope)
            for r in range(e - st):
                for i in range(ID.N_SLOTS):
                    want = tgt[st + r, :, i]
                    pos = int(np.where(want >= 0)[0][0])
                    if int(np.argmax(logits[r, pos])) == int(want[pos]):
                        hits[i] += 1
                tot += 1
    acc = hits / max(tot, 1)
    return acc


def main():
    global ITOS, STOI, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--override-steps", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    ITOS = load_vocab()
    STOI = {w: i for i, w in enumerate(ITOS) if w}
    CFG = {"V": len(ITOS), "d": ARCH["d"], "L": ARCH["L"], "H": ARCH["H"],
           "mlp": ARCH["mlp"], "theta": 1000.0}
    n_par = M.n_params(CFG)

    tag = "parser"
    state_dir = os.path.join(ROOT, "state")
    os.makedirs(state_dir, exist_ok=True)
    ckpt_path = os.path.join(state_dir, f"{tag}.pkl")
    log_path = os.path.join(state_dir, f"{tag}.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    if args.eval_only:
        with open(ckpt_path, "rb") as f:
            st = __import__("pickle").load(f)
        rng = np.random.default_rng(VAL_SEED)
        val = [ID.make_sample(rng) for _ in range(VAL_N)]
        acc = slot_accuracy(st["P"], val)
        log(f"[{tag}] slot-acc " + " ".join(f"{a:.3f}" for a in acc) + f" mean={acc.mean():.3f}")
        return

    if os.path.exists(ckpt_path):
        st = OP.load_ckpt(ckpt_path)
        P, step, drng, hist = st["P"], st["step"], st["data_rng"], st["history"]
        opt = OP.AdamW(P, lr=ARCH["lr"])
        opt.load(st["opt"])
        log(f"[{tag}] resume dari step {step}")
    else:
        prng = np.random.default_rng(2027)
        P = M.init_params(CFG, prng)
        opt = OP.AdamW(P, lr=ARCH["lr"])
        step, drng, hist = 0, np.random.default_rng(4242), {"loss": [], "slot": []}
        log(f"[{tag}] mulai baru — params={n_par:,} vocab={len(ITOS)}")

    total_steps = args.override_steps or ARCH["steps"]
    rng = np.random.default_rng(VAL_SEED)
    val = [ID.make_sample(rng) for _ in range(128)]
    budget_end = time.time() + args.budget
    ema = hist["loss"][-1][2] if hist["loss"] else None
    last_save = 0

    while step < total_steps:
        if time.time() > budget_end - 4:
            break
        lr_scale = OP.cosine_scale(step, total_steps, warmup=60)
        ids, tgt, rope = make_batch(drng, ARCH["BATCH"])
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
            log(f"[{tag}] step {step} loss {loss:.4f} ema {ema:.4f} lr {lr_scale:.3f}")
        if step % 150 == 0 or step == total_steps:
            acc = slot_accuracy(P, val)
            hist["slot"].append((step, float(acc.mean())))
            log(f"[{tag}] step {step} SLOT-ACC mean {acc.mean():.3f} "
                f"(scn {acc[0]:.2f} col {acc[1]:.2f} mood {acc[2]:.2f} orn {acc[3]:.2f}/{acc[4]:.2f} "
                f"subj {acc[5]:.2f} atr {acc[6]:.2f}/{acc[7]:.2f} scol {acc[8]:.2f})")
        if step - last_save >= 25 or time.time() > budget_end - 8:
            OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
            last_save = step
    OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
    log(f"[{tag}] chunk selesai di step {step} ema {ema:.4f}")


if __name__ == "__main__":
    main()
