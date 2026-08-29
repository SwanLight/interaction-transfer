#!/usr/bin/env bash
# 把 out/ 里的 S3 录像压成报告可内嵌的 *_web.mp4。
#
# `make_report.py` 用 base64 把视频塞进单个 HTML（out/README 的"双击就能看"
# 靠的就是这个），所以必须先压到几十 KB 量级——原片 4.5 MB × 25 段会让
# 报告涨到 100 MB 以上。S0/S1 的 *_web.mp4 也是这么来的（约 16 KB/段）。
set -euo pipefail
cd "$(dirname "$0")/.."
n=0
for d in out/s3_source out/s3_knob out/s3_wipe out/s3_probe; do
  for f in "$d"/videos/*.mp4; do
    [ -e "$f" ] || continue
    case "$f" in *_web.mp4) continue;; esac
    o="${f%.mp4}_web.mp4"
    ffmpeg -loglevel error -y -i "$f" -vf "scale=480:-2,fps=12" \
           -c:v libx264 -crf 34 -preset veryfast -an -movflags +faststart "$o"
    n=$((n+1))
  done
done
echo "压了 $n 段，总计 $(du -ch out/s3_*/videos/*_web.mp4 2>/dev/null | tail -1 | cut -f1)"
