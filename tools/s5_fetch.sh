#!/usr/bin/env bash
# 把服务器上 S5 的文本产物取回 out/s5/。
#
# 取报告 txt/json 与 artifact 的 report.json，**以及 S5 的图**。artifact 本身是 npz，
# 按 D-23 不进版本控制（`.gitignore` 里 `*.npz`）。
#
# ⚠️ png 也要取：清空那一步按"非 .md 一律删"，而它只补回 txt/json，于是每 fetch 一次
# 就把 `fig_*.png` 删掉一次。图是 gitignore 的（能重新生成），所以 git status 看不出来，
# 下一次重建 report.html 才发现图没了——**P-66 的同一个形状，删的粒度比补的粒度大**。
#
# ⚠️ P-50：产物先清空再取，最后查文件数——通配符很容易把上一轮的旧文件一起带回来，
# 而文件名一样，看不出来。
#
# ⚠️ **但清空不能连手写文档一起删（P-66）。** `out/s5/README.md` 是手写的、
# 没有任何脚本能重新生成。第一版这里直接 `rm -rf "$DEST"` 把它删了，
# 下一次 `git add -A` 还把这次删除静默提交了进去。现在只删**脚本产的**那些。
set -euo pipefail
HOST="${IT_HOST:-root@10.0.6.98}"
SRC="${IT_S5_OUT:-/tmp/s5}"
cd "$(dirname "$0")/.."
DEST="out/s5"

mkdir -p "$DEST"
find "$DEST" -type f ! -name '*.md' -delete
find "$DEST" -mindepth 1 -type d -empty -delete
ssh -o BatchMode=yes "$HOST" \
  "cd $SRC && tar -czf - \$(find . \\( -name '*.txt' -o -name '*.json' -o -name '*.png' \\) | sort)" \
  | tar -xzf - -C "$DEST"

echo "取回 $(find "$DEST" -type f | wc -l | tr -d ' ') 个文件："
find "$DEST" -type f | sort | sed 's/^/  /'
