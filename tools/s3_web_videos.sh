#!/usr/bin/env bash
# 把 out/ 里的 S3 录像压成报告可内嵌的 *_web.mp4。
#
# `make_report.py` 用 base64 把视频塞进单个 HTML（out/README 的"双击就能看"
# 靠的就是这个），所以要先压——原片 4.5 MB × 25 段会让报告涨到 100 MB 以上。
#
# **保持原生 1280 宽，不降分辨率。** 早先压到 480p/10fps/crf36（约 11 KB/段）
# 确实小，但画面糊到看不清板面朝向和接触位置——而报告的全部意义就是让人
# 看清这些。1280p/20fps/crf22 约 400 KB/段，25 段 10 MB，
# 单文件报告约 14 MB，双击照样打开。
set -euo pipefail
cd "$(dirname "$0")/.."
n=0
mkdir -p out/s3_web
for d in out/s3_source out/s3_knob out/s3_wipe out/s3_probe; do
  for f in "$d"/videos/*.mp4; do
    [ -e "$f" ] || continue
    o="out/s3_web/$(basename "$f")"
    ffmpeg -loglevel error -y -i "$f" -vf "fps=20" \
           -c:v libx264 -crf 22 -preset slow -an -movflags +faststart "$o"
    n=$((n+1))
  done
done
echo "压了 $n 段 -> out/s3_web/（$(du -sh out/s3_web | cut -f1)）"
