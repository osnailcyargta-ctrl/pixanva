#!/bin/bash
# Runner Light 1.3: train chunked ke >=830 -> export bin -> meta entry ->
# cache-bust v27 -> push -> RESTART runner 12m. Log: state/light13_runner.log
set -x
cd "$(dirname "$0")/.."
RLOG=state/light13_runner.log
TARGET=830

echo "=== LIGHT13 start $(date) ===" >> $RLOG
for i in $(seq 1 8); do
  python3 training/train.py --model light13 --budget 480 >> $RLOG 2>&1
  STEP=$(python3 -c "import pickle;print(pickle.load(open('state/light13.pkl','rb'))['step'])")
  echo "=== chunk $i step=$STEP $(date) ===" >> $RLOG
  if [ "$STEP" -ge "$TARGET" ]; then break; fi
done

python3 -c "import sys; sys.path.insert(0,'training'); from export import export_model; export_model('light13')" >> $RLOG 2>&1 || { echo "EXPORT BIN GAGAL" >> $RLOG; exit 1; }
python3 scripts/add_light13_meta.py >> $RLOG 2>&1 || { echo "META GAGAL" >> $RLOG; exit 1; }

# cache-bust v26 -> v27 + tambah cache-bust di fetch meta.json
sed -i 's/?v=v26/?v=v27/g' index.html js/app.js js/worker.js
sed -i 's|fetch("models/meta.json")|fetch("models/meta.json?v=v27")|' js/app.js
node --check js/app.js && node --check js/worker.js || { echo "JS SYNTAX GAGAL" >> $RLOG; exit 1; }

git add -A
git commit -m "rilis Pixanva Light 1.3 — param naik beneran (813k → 1.9M, 4 → 6 lapis), dilatih sampai langkah final; entri meta + cache-bust v27" >> $RLOG 2>&1
git push >> $RLOG 2>&1 && echo "=== PUSH LIGHT13 OK $(date) ===" >> $RLOG || { echo "=== PUSH GAGAL $(date) ===" >> $RLOG; exit 1; }

# lanjutkan 12m
nohup bash scripts/run_stage12m.sh > /dev/null 2>&1 &
echo "=== runner 12m direstart $(date) ===" >> $RLOG
