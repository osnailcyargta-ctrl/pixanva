#!/bin/bash
# Runner chat LLM — relaunch chunked training sampai target tercapai, lalu export.
cd /home/z/my-project/pixanva/training
TARGET=2600
for i in $(seq 1 40); do
  STEP=$(python3 -c "
import pickle, os
p = '../state/chat.pkl'
print(pickle.load(open(p, 'rb'))['step'] if os.path.exists(p) else 0)
")
  if [ "$STEP" -ge "$TARGET" ]; then
    echo "[runner] training selesai di step $STEP"
    break
  fi
  echo "[runner] chunk $i — lanjut dari step $STEP"
  python3 train_chat.py --budget 540 --override-steps $TARGET
done
echo "[runner] export..."
python3 export_chat.py
echo "[runner] BERESE"
