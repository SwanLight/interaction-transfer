#!/usr/bin/env bash
# 把 src/ tools/ plan/ log/ 同步到服务器。服务器无 rsync，走 tar over ssh。
set -euo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
DEST="${IT_DEST:-/workspace/interaction_transfer}"
cd "$(dirname "$0")/.."

# 记下代码版本随代码一起送过去。服务器上没有 .git，生成脚本自己 git rev-parse
# 只会得到 "unknown"，数据集的 manifest 就没有出处可查——而这批数据要一直用到 S7。
git rev-parse HEAD > .git_sha 2>/dev/null || echo unknown > .git_sha
git diff --quiet && git diff --cached --quiet || echo "-dirty" >> .git_sha
tr -d '\n' < .git_sha > .git_sha.tmp && mv .git_sha.tmp .git_sha

COPYFILE_DISABLE=1 tar --no-xattrs --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    -czf - src tools plan log .git_sha \
  | ssh -o BatchMode=yes "$HOST" "mkdir -p $DEST && tar -xzf - -C $DEST"

rm -f .git_sha
echo "已同步到 $HOST:$DEST"
