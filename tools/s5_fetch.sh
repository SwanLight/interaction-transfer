#!/usr/bin/env bash
# 把服务器上 S5 的文本产物取回 out/s5/。
#
# 只取**文本**：报告 txt/json 与 artifact 的 report.json。artifact 本身是 npz，
# 按 D-23 不进版本控制（`.gitignore` 里 `*.npz`）。
#
# ⚠️ P-50：产物先清空再取，逐个显式拷，最后查文件数——通配符很容易把上一轮的
# 旧文件一起带回来，而文件名一样，看不出来。
set -euo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
SRC="${IT_S5_OUT:-/tmp/s5}"
cd "$(dirname "$0")/.."
DEST="out/s5"

rm -rf "$DEST"
mkdir -p "$DEST"
ssh -o BatchMode=yes "$HOST" \
  "cd $SRC && tar -czf - \$(find . -name '*.txt' -o -name '*.json' | sort)" \
  | tar -xzf - -C "$DEST"

echo "取回 $(find "$DEST" -type f | wc -l | tr -d ' ') 个文件："
find "$DEST" -type f | sort | sed 's/^/  /'
