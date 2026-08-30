#!/usr/bin/env bash
# S6 的**指令来源**：把探针集的 43 个 (物体, 原语) 组各造一份 interaction command，
# 与三个主任务的 artifact 一起构成 E-I 的训练指令分布（`plan/04` §5.2）。
#
# 为什么探针集要按原语拆、主任务不拆，见 s5_build_transfer.py 的 --split-by-family。
#
# 顺序不能换：
#   1. freeze  —— P-57：surface 必须在**产生数据的那台机器上**冻结，不能现场重算 FPS；
#   2. split   —— 按 (物体, 原语) 分层重划 calibration，否则每组只有 3~27 条，
#                 够不上 split conformal 的下限（只动 train/calibration，测试划分不碰）；
#   3. build   —— 造 artifact。
#
# ⚠️ 判据是退出码，不是报告文件的开头（P-55）。
#
# 用法（服务器上）::
#
#     IT_PY=/isaac-sim/python.sh bash tools/s6_commands.sh
set -uo pipefail

OUT="${IT_S6_OUT:-/tmp/s6}"
PROBE="${IT_PROBE:-/tmp/s4_probe}"
PY="${IT_PY:-python3}"
BINS="${IT_S5_BINS:-32}"
POINTS="${IT_S5_POINTS:-256}"
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:?--only 后面要跟 freeze|split|build}"; shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
FAILED=""
RAN=0

run() {   # run <报告名> <说明> <命令...>
  local tag="$1" desc="$2"; shift 2
  if [ -n "$ONLY" ] && [ "${tag%%_*}" != "$ONLY" ]; then return; fi
  RAN=$((RAN + 1))
  echo "=========== $tag ==========="
  PYTHONPATH=src "$@" > "$OUT/$tag.log" 2>&1
  local code=$?
  tail -4 "$OUT/$tag.log"
  if [ $code -ne 0 ]; then
    echo "[FAIL] $tag 退出码 $code —— $desc"; FAILED="$FAILED $tag(exit=$code)"
  else
    echo "[OK]   $tag"
  fi
  if grep -q "Traceback (most recent call last)" "$OUT/$tag.log"; then
    echo "[FAIL] $tag 报告里有 traceback"; FAILED="$FAILED $tag(traceback)"
  fi
}

# 1) 冻结探针集的 surface（幂等：hash 对得上就 SKIP）
for obj in ball block column dial flap plunger ridge roller slab slider; do
  run "freeze_$obj" "冻结 $obj 的 surface" "$PY" tools/s5_freeze_surfaces.py "$PROBE/$obj"
done

# 2) 按 (物体, 原语) 分层重划 calibration。push 只剩 block 一个承载物体是已知并接受的
#    （D-76），列在 --accept-thin 里让闸门不长红，但**新出现的**变窄照样报错。
run "split_probe" "重划探针 calibration" \
    "$PY" tools/s6_probe_split.py "$PROBE" --target 30 --min-train 12 \
    --accept-thin push --apply --out "$OUT/probe_split.txt"

# 3) 造指令 artifact
for obj in ball block column dial flap plunger ridge roller slab slider; do
  run "build_$obj" "造 $obj 的指令" \
      "$PY" tools/s5_build_transfer.py --manifest "$PROBE/$obj/manifest.json" \
      --output "$OUT/probe" --split train --split-by-family \
      --bins "$BINS" --surface-points "$POINTS"
done

echo
echo "================= S6 指令来源汇总 ================="
echo "artifact 在 $OUT/probe，报告在 $OUT"
ls "$OUT/probe"/*.npz 2>/dev/null | wc -l | xargs echo "探针指令份数："
if [ "$RAN" -eq 0 ]; then echo "[FAIL] 一项都没跑 —— --only 的值 '$ONLY' 没匹配上"; exit 1; fi
if [ -n "$FAILED" ]; then echo "[FAIL]$FAILED"; exit 1; fi
echo "[PASS] 全部通过"
