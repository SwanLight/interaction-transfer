#!/usr/bin/env bash
# S1 全部自检。每项独立进程——Isaac Lab 不支持一进程内反复建销 SimulationContext。
set -uo pipefail
OUT="${1:-/tmp/s1}"
CHECKS="knob cabinet wiping plates hook allegro pretrain"
mkdir -p "$OUT"; rm -f "$OUT"/s1_*.txt "$OUT"/s1_*.json

for c in $CHECKS; do
  echo "=========== $c ==========="
  PYTHONPATH=src /isaac-sim/python.sh tools/s1_check.py --out "$OUT" --only "$c" \
    > "$OUT/$c.log" 2>&1
  if [ -f "$OUT/s1_$c.txt" ]; then
    cat "$OUT/s1_$c.txt"
  else
    echo "[CRASH] $c 未产出报告"
    grep "py stderr" "$OUT/$c.log" | sed 's/.*py stderr\]: //' | tail -12
  fi
done

echo; echo "================= 汇总 ================="
cat "$OUT"/s1_*.txt 2>/dev/null | grep -E "^\[(PASS|FAIL|INFO)\]" > "$OUT/s1_report.txt"
P=$(grep -c "^\[PASS\]" "$OUT/s1_report.txt" 2>/dev/null || echo 0)
F=$(grep -c "^\[FAIL\]" "$OUT/s1_report.txt" 2>/dev/null || echo 0)
I=$(grep -c "^\[INFO\]" "$OUT/s1_report.txt" 2>/dev/null || echo 0)
echo "PASS $P   FAIL $F   INFO $I"
if [ "$F" -gt 0 ]; then echo; echo "失败项:"; grep "^\[FAIL\]" "$OUT/s1_report.txt"; fi
exit $([ "$F" -gt 0 ] && echo 1 || echo 0)
