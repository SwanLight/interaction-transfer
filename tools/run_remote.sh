#!/usr/bin/env bash
# 在服务器后台跑一条 Isaac Sim 命令并轮询到结束。避免 ssh 前台超时。
set -uo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
DEST="${IT_DEST:-/workspace/interaction_transfer}"
TAG="${2:-run}"
GPU="${IT_GPU:-0}"   # P-29：必须 pin 单卡，否则外力提交时 CUDA 非法访存
CMD="$1"
WAIT="${IT_WAIT:-1}"   # IT_WAIT=0 只投递不轮询，用于八张卡并发
ssh -o BatchMode=yes "$HOST" \
  "cd $DEST && rm -f /tmp/$TAG.log /tmp/$TAG.done && setsid nohup bash -c 'export CUDA_VISIBLE_DEVICES=$GPU; $CMD; echo \$? > /tmp/$TAG.done' > /tmp/$TAG.log 2>&1 < /dev/null & disown" 2>/dev/null
if [ "$WAIT" = "0" ]; then
  echo "LAUNCHED tag=$TAG gpu=$GPU （日志 /tmp/$TAG.log，退出码 /tmp/$TAG.done）"
  exit 0
fi
sleep 5
while ! ssh -o BatchMode=yes "$HOST" "test -f /tmp/$TAG.done" 2>/dev/null; do sleep 10; done
code=$(ssh -o BatchMode=yes "$HOST" "cat /tmp/$TAG.done")
# ⚠️ **退出码 0 不等于成功**（P-74）：脚本一旦创建过 Isaac 的 SimulationApp，
# 未捕获异常也会以 0 退出——Kit 的关停流程自己接管了进程收尾。实测最小复现：
# `AppLauncher(headless=True); raise RuntimeError` -> EXIT=0 而日志里有 traceback。
# 所以这里再查一次日志。P-55 说"报告开头不是判据，退出码才是"，
# 这一条要补上："在 Isaac 下退出码也不是唯一判据"。
if ssh -o BatchMode=yes "$HOST" "grep -q 'Traceback (most recent call last)' /tmp/$TAG.log"; then
  echo "EXIT=$code TRACEBACK=yes  <<< 日志里有未捕获异常，按失败处理"
  exit 1
fi
echo "EXIT=$code"
