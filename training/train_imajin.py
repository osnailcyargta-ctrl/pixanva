"""Pixanva Imajin v4 trainer — TEKS BEBAS MASUK LANGSUNG ke model gambar.
SATU model, satu sequence:
  [BOS] w1..wk [A] slot1..slot9 [GRID_G] [SEP] c1..c(G*G)
- Kata dipakai apa adanya (open vocab): kata gak dikenal → [CHARW] + token
  karakter — GAK ada lagi "prompt diubah jadi tags".
- Slot = LATEN internal yang diprediksi model SENDIRI dari kata (aux task,
  bikin kondisioning nempel di piksel kayak model tag yang terbukti) — user
  gak pernah lihat slot/chip apa pun.
- Loss: slot+GRID di-upweight ×4 (pemahaman teks), sisanya loss gambar.
- CFG: 12% sample uncond = [BOS, UNCOND, GRID, SEP].
Warm-start fase 1 dari imajin6m v3 (backbone warisan heavyqw).
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import tags as TG
import render as RD
import imajin_data as ID
import model as M
import optim as OP

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
SEP = 2

ARCHS = {
    # tangga PM beneran (V=2630): 5.4M -> 11.9M -> 17.2M
    "5m":  dict(d=224, L=8,  H=7, mlp=896,  b16=8,  b32=2, b48=1, lr=2.2e-4, steps=900,  warm=None),
    "12m": dict(d=320, L=9,  H=8, mlp=1280, b16=6,  b32=2, b48=1, lr=2.5e-4, steps=950,  warm=None),
    "17m": dict(d=352, L=10, H=8, mlp=1600, b16=5,  b32=2, b48=1, lr=2.3e-4, steps=850,  warm=None),
}
BATCH_G48 = 1
GSCALE = 32768
SLOT_W = 0.0           # upweight gradient posisi slot+GRID (pemahaman teks) —
                        # 4.0 kelemeteran pas warm-resume (slot-acc mentok 6%)
TYPO_P = 0.12           # prob per kata kena typo → belajar char-fallback beneran
VAL_SEED = 5150
VAL_N = 96
UNCOND_P = 0.12


# ---------------- vocab ----------------

def vocab_path():
    return os.path.join(ROOT, "models", "imajin_vocab.json")


def ensure_vocab(force=False):
    """Build + simpan vocab v4 (itos + specials + chars) sekali — dipakai
    training & inference JS (harus mirror 1:1)."""
    vp = vocab_path()
    if os.path.exists(vp) and not force:
        with open(vp) as f:
            return json.load(f)
    itos, n_words = ID.build_vocab()
    vj = {
        "itos": itos,
        "n_words": n_words,
        "specials": {"BOS": TG.BOS, "SEP": TG.SEP, "UNCOND": TG.UNCOND,
                     "A": ID.A_TOK, "NONE": ID.NONE_TOK, "WORD_BASE": ID.WORD_BASE,
                     "CHARW": ID.charw_id()},
        "grid_token": {str(g): TG.GRID_TOKEN[g] for g in (16, 24, 32, 48, 64)},
        "chars": ID.char_ids(),
    }
    with open(vp, "w") as f:
        json.dump(vj, f)
    print(f"vocab v4: {len(itos)} token ({n_words} kata + {len(ID.CHARS)} char + CHARW)", flush=True)
    return vj


STOI = None


def load_stoi(vj):
    global STOI
    STOI = {w: i for i, w in enumerate(vj["itos"]) if w}
    return STOI


# ---------------- batch ----------------

def prefix_tokens(txt, cond, G, rng=None, typo_p=0.0):
    """(prefix, slots_or_None). uncond → tanpa kata & slot."""
    if rng is not None and rng.random() < UNCOND_P:
        return [TG.BOS, TG.UNCOND, TG.GRID_TOKEN[G], TG.SEP], None
    pre_ids = ID.encode(txt, STOI, G, rng=rng, typo_p=typo_p)   # [BOS]..[A]
    slots = ID.slot_tokens(cond)
    return pre_ids + slots + [TG.GRID_TOKEN[G], TG.SEP], slots


def build_rows(rng, G, B, model_seed, typo_p):
    rows = []
    for _ in range(B):
        txt, cond = ID.make_sample(rng)
        pre, slots = prefix_tokens(txt, cond, G, rng=rng, typo_p=typo_p)
        rows.append((pre, slots, cond))
    return rows


def rows_to_batch(rows, G, model_seed, rng):
    nmax = max(len(p) for p, _, _ in rows)
    B = len(rows)
    T = nmax + G * G
    ids = np.zeros((B, T), dtype=np.int64)
    tgt = np.full((B, T), -1, dtype=np.int64)
    smask = np.zeros((B, T), dtype=bool)
    for i, (pre, slots, cond) in enumerate(rows):
        img = RD.render(cond, G, np.random.default_rng(model_seed + i * 977 + int(rng.integers(1 << 30))))
        img_ids = img.reshape(-1).astype(np.int64) + TG.COLOR_OFFSET
        L = len(pre)
        ids[i, :L] = pre
        ids[i, L:L + G * G] = img_ids
        full = np.concatenate([np.array(pre), img_ids])
        tgt[i, L - 1:L + G * G - 1] = full[L:L + G * G]
        if slots is not None:
            k = L - 1 - ID.N_SLOTS - 2        # index [A] (A, s1..s9, GRID, SEP)
            tgt[i, k:L - 1] = full[k + 1:L]   # [A]→s1, s1→s2, .., s9→GRID, GRID→SEP
            tgt[i, :max(k, 1)] = -1           # buang loss "prediksi kata berikutnya"
            smask[i, k:k + ID.N_SLOTS + 1] = True   # s1..s9 + GRID di-upweight
    return ids, tgt, smask


def make_batch(rng, G, B, model_seed):
    rows = build_rows(rng, G, B, model_seed, TYPO_P)
    return rows_to_batch(rows, G, model_seed, rng)


def make_val():
    rng = np.random.default_rng(VAL_SEED)
    out = []
    for G in (16, 32):
        B = VAL_N // 2
        rows = build_rows(rng, G, B, VAL_SEED * 31 + G * 1000, typo_p=0.0)  # val bersih (no typo)
        out.append(rows_to_batch(rows, G, VAL_SEED * 31 + G * 1000, rng))
    return out


def eval_val(P, val, maxB=8):
    tot, cnt = 0.0, 0
    for ids, tgt, _sm in val:
        n, T = ids.shape
        G = 16 if T < 600 else 32
        ncond = T - G * G
        rope = M.rope_tables(ncond, G, CFG)
        mb = 4 if T > 600 else maxB
        for s in range(0, n, mb):
            e = min(s + mb, n)
            logits = M.forward(P, CFG, ids[s:e], rope)
            l, _ = M.loss_and_dlogits(logits, tgt[s:e])
            tot += float(l) * (e - s)
            cnt += e - s
    return tot / max(cnt, 1)


def slot_accuracy(P, val, n=96):
    """Seberapa tepat model nebak slot laten dari kata (greedy first-slot)."""
    rng = np.random.default_rng(7788)
    hits, tot = 0, 0
    for _ in range(n):
        txt, cond = ID.make_sample(rng)
        pre, slots = prefix_tokens(txt, cond, 16, rng=None)
        want = slots[0]  # scene — paling gampang dibaca
        ids = np.array(pre[:len(pre) - ID.N_SLOTS - 2])   # [BOS] kata.. [A]
        rope = M.rope_tables(len(ids), 0, CFG)
        logits = M.forward(P, CFG, ids[None], rope)
        got = int(np.argmax(logits[0, -1]))
        hits += int(got == want)
        tot += 1
    return hits / max(tot, 1)


def warm_start(ckpt_src, cfg):
    """Copy backbone + emb rows lama (0..192: palet/tag/kelas) — rows kata/char fresh."""
    with open(ckpt_src, "rb") as f:
        st = __import__("pickle").load(f)
    P_old = st["P"]
    P = M.init_params(cfg, np.random.default_rng(2028))
    n_copy = min(P_old["emb"].shape[0], P["emb"].shape[0])
    P["emb"][:n_copy] = P_old["emb"][:n_copy]
    for k in P_old:
        if k == "emb":
            continue
        if k in P and P[k].shape == P_old[k].shape:
            P[k] = P_old[k].copy()
    print(f"warm-start: {n_copy} emb rows + bobot transformer dari {os.path.basename(ckpt_src)}", flush=True)
    return P


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(ARCHS))
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--override-steps", type=int, default=0)
    ap.add_argument("--extra-g48", type=int, default=50)
    ap.add_argument("--rebuild-vocab", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    vj = ensure_vocab(force=args.rebuild_vocab)
    load_stoi(vj)

    arch = ARCHS[args.size]
    CFG = {"V": len(vj["itos"]), "d": arch["d"], "L": arch["L"], "H": arch["H"],
           "mlp": arch["mlp"], "theta": 1000.0}
    n_par = M.n_params(CFG)

    tag = f"imajin{args.size}"
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
        log(f"[{tag}] VAL akhir {eval_val(st['P'], make_val()):.4f} | slot-acc(scene) {slot_accuracy(st['P'], None):.3f}")
        return

    if os.path.exists(ckpt_path):
        st = OP.load_ckpt(ckpt_path)
        P, step, drng, hist = st["P"], st["step"], st["data_rng"], st["history"]
        opt = OP.AdamW(P, lr=arch["lr"])
        opt.load(st["opt"])
        log(f"[{tag}] resume dari step {step}")
    else:
        wsrc = arch.get("warm")
        src = os.path.join(state_dir, f"{wsrc}.pkl") if wsrc else None
        if src and os.path.exists(src):
            P = warm_start(src, CFG)
        elif src:
            log(f"[{tag}] warm {wsrc} gak ada — dari nol")
            P = M.init_params(CFG, np.random.default_rng(hash(tag + "v4") % (2 ** 32)))
        else:
            P = M.init_params(CFG, np.random.default_rng(hash(tag + "v4") % (2 ** 32)))
        opt = OP.AdamW(P, lr=arch["lr"])
        step, drng, hist = 0, np.random.default_rng(42), {"loss": [], "val": []}
        log(f"[{tag}] mulai baru — params={n_par:,} V={CFG['V']} warm={arch.get('warm')}")

    total_steps = args.override_steps or arch["steps"]
    val = make_val()
    budget_end = time.time() + args.budget
    ema = hist["loss"][-1][2] if hist["loss"] else None
    g48_mode = step >= total_steps
    g48_end = total_steps + args.extra_g48
    last_save = 0

    while step < (g48_end if g48_mode else total_steps):
        if time.time() > budget_end - 4:
            break
        if g48_mode:
            G, B = 48, BATCH_G48
        else:
            r = drng.random()
            p32 = 0.22 if step > total_steps * 0.25 else 0.08
            G = 32 if r < p32 else 16
            B = arch["b32"] if G == 32 else arch["b16"]
        lr_scale = OP.cosine_scale(step, g48_end)
        ids, tgt, smask = make_batch(drng, G, B, step * 7919)
        rope = M.rope_tables(ids.shape[1] - G * G, G, CFG)
        logits, cache = M.forward(P, CFG, ids, rope, record=True)
        loss, dlogits = M.loss_and_dlogits(logits, tgt)
        dlogits[smask] *= SLOT_W
        dlogits *= GSCALE
        G_ = M.backward(P, CFG, cache, dlogits)
        OP.clip_grads(G_, 1.0 * GSCALE)
        opt.step(P, G_, lr_scale)
        step += 1
        ema = loss if ema is None else 0.95 * ema + 0.05 * loss
        hist["loss"].append((step, float(loss), float(ema)))
        if step % 25 == 0:
            log(f"[{tag}] step {step} loss {loss:.4f} ema {ema:.4f} lr {lr_scale:.3f} G{G} B{B}")
        if step % 150 == 0 or step == g48_end:
            vl = eval_val(P, val)
            hist["val"].append((step, vl))
            log(f"[{tag}] step {step} VAL {vl:.4f}")
        if step - last_save >= 25 or time.time() > budget_end - 8:
            OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
            last_save = step
    OP.save_ckpt(ckpt_path, P, opt, step, drng, hist)
    log(f"[{tag}] chunk selesai di step {step} ema {ema:.4f}")


if __name__ == "__main__":
    main()
