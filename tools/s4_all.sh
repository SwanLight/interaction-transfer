#!/usr/bin/env bash
# S4 全套验收：逐项跑、逐项收退出码、任一项非零则整体失败。
#
# ⚠️ 这个脚本是 P-55 的结构性补救。此前 S4 的四类检查是手敲命令一条条跑的，
# 输出重定向进各自的 txt。第 2b 条抛了未捕获异常之后：
#   * 那份 txt 的**开头四行仍然全是 [PASS]**，尾部才是 traceback；
#   * 后面五条检查根本没跑；
#   * 而我按文件头写了 README，声称"七条通过"。
# 所以判据不能是"报告里有没有 PASS"，只能是**退出码**，而且必须有一处把
# 所有退出码汇总起来。
#
# 用法（服务器上）::
#
#     bash tools/s4_all.sh                 # 只跑验收（verify / leak / probe）
#     bash tools/s4_all.sh --extract       # 连提取一起重跑（16 640 条，约 20 min）
#     bash tools/s4_all.sh --only region_probe   # 只重跑某一类（三个数据集都跑）
#
# 提取默认**不跑**：它是幂等的重活，而验收才是每次改完提取器都要重来的那一半。
# `--only` 是给"只改了报告措辞、要让落盘产物跟上代码"这类情形用的；
# 它**不放宽判据**——被选中的那一类仍然逐项收退出码，任一非零则整体非零。
set -uo pipefail

OUT="${IT_S4_OUT:-/tmp/s4_reports}"
PY="${IT_PY:-/isaac-sim/python.sh}"
JOBS="${IT_JOBS:-24}"
DO_EXTRACT=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --extract) DO_EXTRACT=1; shift ;;
    --only)    ONLY="${2:?--only 后面要跟 extract|verify|leak|region_probe}"; shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done

# 任务名  S3 源目录  S4 记录目录
TASKS="drawer:/tmp/s3_drawer_v3:/tmp/s4_drawer
wipe:/tmp/s3_wipe_v5:/tmp/s4_wipe
knob:/tmp/s3_knob_v4:/tmp/s4_knob"

mkdir -p "$OUT"
FAILED=""
RAN=0

run() {   # run <报告文件名> <说明> <命令...>
  local tag="$1" desc="$2"; shift 2
  # --only 过滤：tag 形如 <类别>_<任务名>，类别是最后一个下划线之前的部分
  if [ -n "$ONLY" ] && [ "${tag%_*}" != "$ONLY" ]; then
    return
  fi
  RAN=$((RAN + 1))
  echo "=========== $tag ==========="
  PYTHONPATH=src "$@" > "$OUT/$tag.txt" 2>&1
  local code=$?
  tail -4 "$OUT/$tag.txt"
  if [ $code -ne 0 ]; then
    echo "[FAIL] $tag 退出码 $code —— $desc"
    FAILED="$FAILED $tag(exit=$code)"
  else
    echo "[OK]   $tag"
  fi
  # 退出码之外再兜一层：脚本自己 catch 住异常并返回 0 也不许蒙混过去
  if grep -q "Traceback (most recent call last)" "$OUT/$tag.txt"; then
    echo "[FAIL] $tag 报告里有 traceback —— 有检查没跑完"
    FAILED="$FAILED $tag(traceback)"
  fi
}

for t in $TASKS; do
  name="${t%%:*}"; rest="${t#*:}"; src="${rest%%:*}"; rec="${rest#*:}"
  if [ $DO_EXTRACT -eq 1 ]; then
    run "extract_$name" "提取" "$PY" tools/s4_extract.py "$src" --out "$rec" --jobs "$JOBS"
  fi
  run "verify_$name"       "独立验收"     "$PY" tools/s4_verify_records.py "$rec" "$src" --sample 40
  run "leak_$name"         "泄漏检查九条" "$PY" tools/s4_leak_checks.py "$rec" "$src" --sample 10
  run "region_probe_$name" "region 探针"  "$PY" tools/s4_region_probe.py "$rec"
done

echo
echo "================= S4 汇总 ================="
echo "跑了 $RAN 项，报告在 $OUT"
if [ "$RAN" -eq 0 ]; then
  echo "[FAIL] 一项都没跑 —— --only 的值 '$ONLY' 没匹配上任何检查"
  exit 1
fi
if [ -n "$FAILED" ]; then
  echo "[FAIL]$FAILED"
  exit 1
fi
echo "[PASS] 全部通过"
