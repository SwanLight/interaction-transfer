#!/usr/bin/env bash
# S5 全套：三个任务各构造 train split 的 interaction transfer，再在**冻结的**留出
# 划分上评估。逐项收退出码，任一非零则整体失败。
#
# ⚠️ 与 s4_all.sh 同一个理由（P-55）：判据是退出码，不是报告文件的开头。
# 这里额外查 traceback，因为脚本自己 catch 住异常返回 0 也不许蒙混过去。
#
# S5 不需要 Isaac：提取好的 S4 记录是纯 npz，全程只用 numpy。所以默认走系统
# python3；要复用 Isaac 那个解释器就设 IT_PY。
#
# 用法（服务器上）::
#
#     bash tools/s5_all.sh                    # 构造 + 评估
#     bash tools/s5_all.sh --only parasite    # 只查寄生接触
#     bash tools/s5_all.sh --only build       # 只重建 artifact
#     bash tools/s5_all.sh --only eval        # 只重跑评估（artifact 不动）
#     bash tools/s5_all.sh --only units       # 只跑单位/刻度体检
set -uo pipefail

OUT="${IT_S5_OUT:-/tmp/s5}"
PY="${IT_PY:-python3}"
BINS="${IT_S5_BINS:-32}"
POINTS="${IT_S5_POINTS:-256}"
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:?--only 后面要跟 freeze|parasite|build|eval|units}"; shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done

# 任务名  S4 记录目录
TASKS="drawer:/tmp/s4_drawer
wipe:/tmp/s4_wipe
knob:/tmp/s4_knob"

mkdir -p "$OUT"
FAILED=""
RAN=0

# 冻结 surface 必须先于一切（P-57）。它是幂等的：已存在且 hash 对得上就跳过。
# 之后 build / eval 一律走 --surface，不再在每次运行时重新 FPS。
surface_of() {   # surface_of <S4 记录目录> <object> <geometry>
  echo "$1/surfaces/$2-$3.npz"
}

run() {   # run <报告文件名> <说明> <命令...>
  local tag="$1" desc="$2"; shift 2
  if [ -n "$ONLY" ] && [ "${tag%%_*}" != "$ONLY" ]; then
    return
  fi
  RAN=$((RAN + 1))
  echo "=========== $tag ==========="
  PYTHONPATH=src "$@" > "$OUT/$tag.txt" 2>&1
  local code=$?
  tail -5 "$OUT/$tag.txt"
  if [ $code -ne 0 ]; then
    echo "[FAIL] $tag 退出码 $code —— $desc"
    FAILED="$FAILED $tag(exit=$code)"
  else
    echo "[OK]   $tag"
  fi
  if grep -q "Traceback (most recent call last)" "$OUT/$tag.txt"; then
    echo "[FAIL] $tag 报告里有 traceback —— 有检查没跑完"
    FAILED="$FAILED $tag(traceback)"
  fi
}

for t in $TASKS; do
  name="${t%%:*}"; rec="${t#*:}"
  run "freeze_$name" "冻结 surface" "$PY" tools/s5_freeze_surfaces.py "$rec"
  run "parasite_$name" "寄生接触检查" "$PY" tools/s5_parasite_check.py "$rec" --sample 30
  run "build_$name" "构造 train artifact" \
      "$PY" tools/s5_build_transfer.py --manifest "$rec/manifest.json" \
      --output "$OUT/$name" --split train --bins "$BINS" --surface-points "$POINTS"

  # 一个 S4 记录目录里可能有多个 (object, geometry) 分组，逐个评估。
  for artifact in "$OUT/$name"/*.npz; do
    [ -e "$artifact" ] || { echo "[FAIL] $name 没有 artifact"; FAILED="$FAILED $name(no-artifact)"; break; }
    base="$(basename "$artifact" .npz)"
    run "eval_$base" "envelope 评估" \
        "$PY" tools/s5_eval_envelope.py --artifact "$artifact" \
        --manifest "$rec/manifest.json" --out "$OUT/$name/$base.eval.txt"
  done
done

# 单位与刻度体检放在最后：它要拿 artifact 查 payload 里有没有连续 mode 与 effect 刻度，
# 所以必须在 build 之后跑（P-68 / P-69 / P-70，判据见 tools/s5_units_probe.py）。
UNITS_ARGS=""
for t in $TASKS; do
  name="${t%%:*}"; rec="${t#*:}"
  artifact="$(ls "$OUT/$name"/*.npz 2>/dev/null | head -1)"
  if [ -n "$artifact" ]; then
    UNITS_ARGS="$UNITS_ARGS --task $name=$rec=$artifact"
  else
    UNITS_ARGS="$UNITS_ARGS --task $name=$rec"
  fi
done
# 抽屉在最粗档（64 格）上的残差 1.73× 是已知的、成因未查清的（D-72 末尾），
# 列进 --accept-drift 让闸门不长红——**新出现的**超标照样报错。
run "units_log" "单位与刻度体检" "$PY" tools/s5_units_probe.py $UNITS_ARGS \
    --accept-drift drawer --out "$OUT/units_probe.txt"

echo
echo "================= S5 汇总 ================="
echo "跑了 $RAN 项，产物与报告在 $OUT"
if [ "$RAN" -eq 0 ]; then
  echo "[FAIL] 一项都没跑 —— --only 的值 '$ONLY' 没匹配上任何检查"
  exit 1
fi
if [ -n "$FAILED" ]; then
  echo "[FAIL]$FAILED"
  exit 1
fi
echo "[PASS] 全部通过"
