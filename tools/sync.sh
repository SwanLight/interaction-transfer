#!/usr/bin/env bash
# 把 src/ tools/ plan/ log/ 同步到服务器。服务器无 rsync，走 tar over ssh。
set -euo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
DEST="${IT_DEST:-/workspace/interaction_transfer}"
cd "$(dirname "$0")/.."

COPYFILE_DISABLE=1 tar --no-xattrs --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    -czf - src tools plan log \
  | ssh -o BatchMode=yes "$HOST" "mkdir -p $DEST && tar -xzf - -C $DEST"

echo "已同步到 $HOST:$DEST"
