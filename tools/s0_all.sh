#!/usr/bin/env bash
# S0：录制全部场景。每个场景独立进程（规则 11）。
set -uo pipefail
OUT="${1:-/tmp/s0}"
mkdir -p "$OUT"; rm -f "$OUT"/*.mp4
for s in knob_rim knob_pin cabinet wiping hook slider; do
  echo "=========== $s ==========="
  PYTHONPATH=src /isaac-sim/python.sh tools/s0_record.py --scene "$s" --out "$OUT" \
    > "$OUT/$s.log" 2>&1
  grep -E "^(WROTE|RESULT)" "$OUT/$s.log" || {
    echo "[CRASH] $s"; grep "py stderr" "$OUT/$s.log" | sed 's/.*py stderr\]: //' | tail -10; }
done
echo; echo "=== 产出 ==="; ls -la "$OUT"/*.mp4 2>/dev/null || echo "无 mp4"
