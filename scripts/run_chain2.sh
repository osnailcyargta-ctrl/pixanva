#!/bin/bash
# Rantai anti-rollback: Heavy QR (dari nol / lanjut dari branch ckpt) -> push rilis
# -> Imajin 12M v2 (lanjut dari bin ke 1800) -> push rilis.
# Progres QR di-push ke branch ckpt tiap ~2 chunk (16 menit) — rollback = loss maks 16 menit.
set -x
cd "$(dirname "$0")/.."
RLOG=state/chain2.log

echo "=== CHAIN2 start $(date) ===" >> $RLOG

# ---------- FASE A: HEAVY QR ----------
scripts/ckpt_git.sh pull state/ckpt_heavyqr.npz >> $RLOG 2>&1 || true
python3 scripts/ckpt_antirollback.py restore heavyqr >> $RLOG 2>&1 || true

QR_DONE=0
for i in $(seq 1 20); do
  python3 training/train.py --model heavyqr --override-steps 800 --budget 480 >> $RLOG 2>&1
  STEP=$(python3 -c "import pickle;print(pickle.load(open('state/heavyqr.pkl','rb'))['step'])")
  echo "=== QR chunk $i step=$STEP $(date) ===" >> $RLOG
  if [ "$STEP" -ge 850 ]; then QR_DONE=1; break; fi
  if [ $((i % 2)) -eq 0 ]; then
    python3 scripts/ckpt_antirollback.py dump heavyqr >> $RLOG 2>&1 \
      && scripts/ckpt_git.sh push state/ckpt_heavyqr.npz "qr-step$STEP" >> $RLOG 2>&1
  fi
done

if [ "$QR_DONE" = "1" ]; then
  python3 -c "import sys; sys.path.insert(0,'training'); from export import export_model; export_model('heavyqr')" >> $RLOG 2>&1 || { echo "QR EXPORT GAGAL" >> $RLOG; exit 1; }
  python3 scripts/add_heavyqr_meta.py >> $RLOG 2>&1 || { echo "QR META GAGAL" >> $RLOG; exit 1; }
  sed -i 's/?v=v29/?v=v30/g' index.html js/app.js js/worker.js
  node --check js/app.js && node --check js/worker.js || { echo "JS GAGAL" >> $RLOG; exit 1; }
  git add -A
  git commit -m "rilis Pixanva Heavy QR — param naik beneran (6.4M → 10M, 8 → 10 lapis), paling detail sepanjang masa; Heavy QW turun ke popup; cache-bust v30" >> $RLOG 2>&1
  git push >> $RLOG 2>&1 && echo "=== PUSH QR OK $(date) ===" >> $RLOG || { echo "=== PUSH QR GAGAL $(date) ===" >> $RLOG; exit 1; }
else
  echo "=== QR gak selesai (rollback mid-flight?) — progres aman di branch ckpt ===" >> $RLOG
  python3 scripts/ckpt_antirollback.py dump heavyqr >> $RLOG 2>&1 \
    && scripts/ckpt_git.sh push state/ckpt_heavyqr.npz "qr-final" >> $RLOG 2>&1
  exit 0
fi

# ---------- FASE B: IMAJIN 12M v2 ----------
python3 scripts/bin2pkl_imajin12m.py >> $RLOG 2>&1 || true
for i in $(seq 1 14); do
  python3 training/train_imajin.py --size 12m --override-steps 1800 --budget 480 >> $RLOG 2>&1
  STEP=$(python3 -c "import pickle;print(pickle.load(open('state/imajin12m.pkl','rb'))['step'])")
  echo "=== 12M chunk $i step=$STEP $(date) ===" >> $RLOG
  if [ "$STEP" -ge 1850 ]; then break; fi
done

python3 training/train_imajin.py --size 12m --eval-only >> $RLOG 2>&1
python3 training/export_imajin.py 12m >> $RLOG 2>&1 || { echo "12M EXPORT GAGAL" >> $RLOG; exit 1; }
sed -i 's/?v=v30/?v=v31/g' index.html js/app.js js/worker.js
node --check js/app.js && node --check js/worker.js || { echo "JS GAGAL" >> $RLOG; exit 1; }
timeout 300 node training/e2e_imajin.mjs 12m >> $RLOG 2>&1 || echo "e2e warning" >> $RLOG
git add -A
git commit -m "imajin 12M v2 — lanjut ke 1800 step (warm-restart dari bin): slot laten & CFG makin mateng; cache-bust v31" >> $RLOG 2>&1
git push >> $RLOG 2>&1 && echo "=== PUSH 12M v2 OK $(date) ===" >> $RLOG || echo "=== PUSH 12M GAGAL $(date) ===" >> $RLOG
echo "=== CHAIN2 done $(date) ===" >> $RLOG
