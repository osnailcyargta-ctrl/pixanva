#!/bin/bash
# Push/pull checkpoint npz ke branch `ckpt` (orphan, kecil, di-force-push) —
# satu-satunya penyelamat kalau sandbox ke-rollback lagi.
# Usage: ckpt_git.sh push <file-npz> <label>   |   ckpt_git.sh pull <file-npz>
set -e
cd "$(dirname "$0")/.."
CMD=$1; FILE=$2; LABEL=$3

if [ "$CMD" = "push" ]; then
  timeout 60 git fetch origin ckpt > /dev/null 2>&1 || true
  PREV=$(git rev-parse origin/ckpt 2>/dev/null || true)
  export GIT_INDEX_FILE=/tmp/ckpt_idx_$$
  git read-tree --empty
  git add -f "$FILE"
  TREE=$(git write-tree)
  unset GIT_INDEX_FILE
  if [ -n "$PREV" ]; then
    COMMIT=$(git commit-tree "$TREE" -p "$PREV" -m "ckpt $LABEL $(date +%H:%M)")
  else
    COMMIT=$(git commit-tree "$TREE" -m "ckpt $LABEL $(date +%H:%M)")
  fi
  timeout 120 git push -f origin "$COMMIT:refs/heads/ckpt" > /dev/null 2>&1 \
    && echo "ckpt pushed ($LABEL)" || echo "ckpt push GAGAL (lanjut aja)"
elif [ "$CMD" = "pull" ]; then
  timeout 60 git fetch origin ckpt > /dev/null 2>&1 || { echo "branch ckpt gak ada"; exit 0; }
  TIP=$(git rev-parse origin/ckpt 2>/dev/null) || { echo "branch ckpt kosong"; exit 0; }
  git cat-file blob "$TIP:$(basename $FILE)" > "$FILE" 2>/dev/null \
    && echo "ckpt pulled -> $FILE" || echo "ckpt gak ketemu di branch"
fi
