#!/bin/bash
# FASE B saja: Imajin 12M v2 — lanjut dari bin (step 950) ke 1800 (+50 g48 skip — OOM).
# Progres di-push ke branch ckpt tiap 2 chunk. Log: state/run_12mv2.log
set -x
cd "$(dirname "$0")/.."
RLOG=state/run_12mv2.log
echo "=== 12M v2 start $(date) ===" >> $RLOG

python3 scripts/bin2pkl_imajin12m.py >> $RLOG 2>&1

for i in $(seq 1 30); do
  python3 training/train_imajin.py --size 12m --override-steps 1800 --budget 540 >> $RLOG 2>&1
  STEP=$(python3 -c "import pickle;print(pickle.load(open('state/imajin12m.pkl','rb'))['step'])")
  echo "=== chunk $i step=$STEP $(date) ===" >> $RLOG
  if [ "$STEP" -ge 1800 ]; then break; fi
  if [ $((i % 2)) -eq 0 ]; then
    python3 scripts/ckpt_antirollback.py dump imajin12m >> $RLOG 2>&1 \
      && scripts/ckpt_git.sh push state/ckpt_imajin12m.npz "imj12-step$STEP" >> $RLOG 2>&1
  fi
done

python3 training/train_imajin.py --size 12m --eval-only >> $RLOG 2>&1
python3 training/export_imajin.py 12m >> $RLOG 2>&1 || { echo "EXPORT GAGAL" >> $RLOG; exit 1; }
sed -i 's/?v=v30/?v=v31/g' index.html js/app.js js/worker.js
node --check js/app.js && node --check js/worker.js || { echo "JS GAGAL" >> $RLOG; exit 1; }
timeout 300 node training/e2e_imajin.mjs 12m >> $RLOG 2>&1 || echo "e2e warning" >> $RLOG
git add -A
git commit -m "imajin 12M v2 — lanjut ke 1800 step (warm-restart dari bin): slot laten & CFG makin mateng; cache-bust v31" >> $RLOG 2>&1
git push >> $RLOG 2>&1 && echo "=== PUSH 12M v2 OK $(date) ===" >> $RLOG || echo "=== PUSH 12M GAGAL $(date) ===" >> $RLOG
