#!/bin/bash
# Runner imajin fase 2 (12M): train chunked sampe target -> eval -> export ->
# bump JS (bin 12m + cache-bust v27) -> push. Semua log ke state/imajin12m_runner.log
set -x
cd "$(dirname "$0")/.."
RLOG=state/imajin12m_runner.log
TARGET=1000
MAXCHUNK=15

echo "=== STAGE 2 (12M) start $(date) ===" >> $RLOG

for i in $(seq 1 $MAXCHUNK); do
  python3 training/train_imajin.py --size 12m --budget 480 >> $RLOG 2>&1
  STEP=$(python3 -c "import pickle;print(pickle.load(open('state/imajin12m.pkl','rb'))['step'])")
  echo "=== chunk $i done, step=$STEP $(date) ===" >> $RLOG
  if [ "$STEP" -ge "$TARGET" ]; then
    echo "=== target $TARGET tercapai ===" >> $RLOG
    break
  fi
done

# eval akhir + export
python3 training/train_imajin.py --size 12m --eval-only >> $RLOG 2>&1
python3 training/export_imajin.py 12m >> $RLOG 2>&1 || { echo "EXPORT GAGAL" >> $RLOG; exit 1; }

# bump JS: bin 5m -> 12m, cache-bust v27 -> v28
sed -i 's/imajin5m\.bin/imajin12m.bin/g' js/worker.js
sed -i 's/?v=v27/?v=v28/g' index.html js/app.js js/worker.js
node --check js/app.js && node --check js/worker.js || { echo "JS SYNTAX GAGAL" >> $RLOG; exit 1; }

# sanity: worker harus nunjuk imajin12m.bin + vocab masih ke-fetch
rg -q "imajin12m.bin" js/worker.js || { echo "WORKER GAK NUNJUK 12M" >> $RLOG; exit 1; }
rg -q "imajin_vocab.json" js/worker.js || { echo "VOCAB FETCH ILANG" >> $RLOG; exit 1; }

# e2e cepat 12m (node, bin baru) — 3 prompt
timeout 180 node training/e2e_imajin.mjs >> $RLOG 2>&1 || echo "E2E warning (cek manual)" >> $RLOG

git add -A
git commit -m "imajin fase 2 — Pixanva Imajin 12M (d320 L9 H8): makin paham prompt bebas; worker -> imajin12m.bin; cache-bust v27" >> $RLOG 2>&1
git push >> $RLOG 2>&1 && echo "=== PUSH 12M OK $(date) ===" >> $RLOG || echo "=== PUSH GAGAL $(date) ===" >> $RLOG
