#!/usr/bin/env bash
# 在服务器后台跑一条 Isaac Sim 命令并轮询到结束。避免 ssh 前台超时。
set -uo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
DEST="${IT_DEST:-/workspace/interaction_transfer}"
TAG="${2:-run}"
CMD="$1"
ssh -o BatchMode=yes "$HOST" \
  "cd $DEST && rm -f /tmp/$TAG.log /tmp/$TAG.done && setsid nohup bash -c '$CMD; echo \$? > /tmp/$TAG.done' > /tmp/$TAG.log 2>&1 < /dev/null & disown" 2>/dev/null
sleep 5
while ! ssh -o BatchMode=yes "$HOST" "test -f /tmp/$TAG.done" 2>/dev/null; do sleep 10; done
code=$(ssh -o BatchMode=yes "$HOST" "cat /tmp/$TAG.done")
echo "EXIT=$code"
