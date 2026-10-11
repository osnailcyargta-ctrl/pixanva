#!/bin/bash
# Push/pull checkpoint npz ke branch `ckpt` (orphan, di-force-push).
# Usage: ckpt_git.sh push <file-npz> <label>   |   ckpt_git.sh pull <file-npz>
cd "$(dirname "$0")/.."
CMD=$1; FILE=$2; LABEL=$3

if [ "$CMD" = "push" ]; then
  timeout 60 git fetch origin ckpt > /dev/null 2>&1
  PREV=""
  git rev-parse -q --verify origin/ckpt > /dev/null 2>&1 && PREV=$(git rev-parse origin/ckpt)
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
  if timeout 120 git push -f origin "$COMMIT:refs/heads/ckpt" > /dev/null 2>&1; then
    echo "ckpt pushed ($LABEL)"
  else
    echo "ckpt push GAGAL (lanjut aja)"
  fi
elif [ "$CMD" = "pull" ]; then
  timeout 60 git fetch origin ckpt > /dev/null 2>&1
  TIP=""
  git rev-parse -q --verify origin/ckpt > /dev/null 2>&1 && TIP=$(git rev-parse origin/ckpt)
  if [ -z "$TIP" ]; then echo "branch ckpt gak ada/kosong"; exit 0; fi
  git cat-file blob "$TIP:$(basename $FILE)" > "$FILE" 2>/dev/null \
    && echo "ckpt pulled -> $FILE" || echo "ckpt gak ketemu di branch"
fi
