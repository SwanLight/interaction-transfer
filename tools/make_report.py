#!/usr/bin/env python3
"""生成 S0/S1 的 HTML 报告（`plan/06` §1 要求）。

把视频和数字放在一起，供 `plan/06` §7 的人工检查使用。
用法：python3 tools/make_report.py out/s0 log/s1/s1_report.txt out/report.html
"""
import html
import os
import re
import sys

VIDEOS = [
    ("knob_rim", "旋钮 · 轮缘（蓝，μ=0.10）",
     "径向压紧 25 N + 沿圆周拖。<b>橙色销钉三帧位置不变 = 圆盘没转。</b>",
     "Δθ = +0.025 rad，远低于目标下限 1.0 —— 推不动", "fail"),
    ("knob_pin", "旋钮 · 销钉（橙，μ=0.80）",
     "同样 25 N，改为切向直推。<b>销钉明显转过去了。</b>",
     "Δθ = +1.516 rad，落在目标区间 1.0–2.2 —— 推得动", "ok"),
    ("wiping_padrod", "擦拭 · 垫头杆直擦",
     "Z 轴力控压出 5 N，XY 位置 PD 以 0.05 m/s 平移。",
     "Fn = 5.456 ± 0.369 N（变异 6.8%），|v_xy| = 0.0498 m/s", "ok"),
    ("cabinet_travel", "抽屉 · 全行程",
     "±30 N 力矩拉开再推回，检查无穿模、不越限。",
     "拉开 180.0 mm（上限 180），推回 0.0 mm", "ok"),
    ("hook_sweep", "钩杆 · 绕轴扫掠",
     "PD 摆位到销钉后方，绕圆盘轴画弧。",
     "τ_hook = 0.515 N·m > τ_need = 0.420；Δθ = +0.167 rad（方向正确）", "ok"),
    ("slider_pretrain", "预训练物体集 · 滑块",
     "只用于给 E-I 产生交互指令多样性，永不作为任务评估。",
     "15 N 推满 150 mm 行程", "ok"),
]

CSS = """
:root{--bg:#fbfbfd;--fg:#1a1a1e;--mut:#65656e;--line:#e3e3e9;--card:#fff;
      --ok:#0a7d4f;--okbg:#e8f6ef;--fail:#a8341a;--failbg:#fdeee9;--code:#f4f4f7}
@media (prefers-color-scheme:dark){:root{--bg:#131317;--fg:#e9e9ee;--mut:#9a9aa4;
  --line:#2c2c34;--card:#1b1b21;--ok:#4ade9a;--okbg:#12291f;--fail:#f08a6c;
  --failbg:#2c1610;--code:#232329}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 28px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:18px;margin:0 0 20px}
.card h3{margin:0 0 4px;font-size:17px}
.card p{margin:0 0 12px;color:var(--mut);font-size:14px}
video{width:100%;border-radius:8px;background:#000;display:block}
.res{margin-top:12px;padding:10px 13px;border-radius:8px;font-size:14px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.res.ok{background:var(--okbg);color:var(--ok)}
.res.fail{background:var(--failbg);color:var(--fail)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:820px){.pair{grid-template-columns:1fr}}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;
    padding:14px 16px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin:0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:13px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
     font-family:ui-monospace,monospace}
.tag.p{background:var(--okbg);color:var(--ok)}
.tag.i{background:var(--code);color:var(--mut)}
.note{border-left:3px solid var(--line);padding-left:14px;color:var(--mut);font-size:14px}
"""


def main(vid_dir, s1_txt, out_path):
    rel = os.path.relpath(vid_dir, os.path.dirname(out_path) or ".")
    parts = [f"<!doctype html><html lang=zh><meta charset=utf-8>",
             "<title>S0/S1 验证报告 · Functional Interaction Transfer</title>",
             "<meta name=viewport content='width=device-width,initial-scale=1'>",
             f"<style>{CSS}</style><div class=wrap>",
             "<h1>S0 / S1 验证报告</h1>",
             "<p class=sub>Functional Interaction Transfer · Isaac Sim 5.1.0-rc.19 + "
             "Isaac Lab 2.3.1（快照 2ab57ade）· 8×RTX 4090</p>",
             "<div class=note>为什么要有这份报告：S1 过程中出现过多次<b>假 PASS</b>——"
             "第一版旋钮标定报出「轮缘推不动」，实际是推子被底座卡住，不是打滑。"
             "数字看不出来，<b>看一眼画面就能发现</b>。"
             "<code>plan/06</code> §7 因此要求每次正式评估前人工看视频。</div>",
             "<h2>D-14 摩擦标定 · 核心对照</h2>",
             "<p class=sub>同一份物理、同样 25 N 法向力上限，只改接触区域。"
             "这是旋钮任务上 region 可检验性的前提。</p><div class=pair>"]

    for key, title, desc, res, cls in VIDEOS[:2]:
        parts.append(
            f"<div class=card><h3>{html.escape(title)}</h3><p>{desc}</p>"
            f"<video controls loop muted playsinline src='{rel}/{key}.mp4'></video>"
            f"<div class='res {cls}'>{html.escape(res)}</div></div>")
    parts.append("</div>")

    parts.append("<h2>其余场景</h2>")
    for key, title, desc, res, cls in VIDEOS[2:]:
        parts.append(
            f"<div class=card><h3>{html.escape(title)}</h3><p>{desc}</p>"
            f"<video controls loop muted playsinline src='{rel}/{key}.mp4'></video>"
            f"<div class='res {cls}'>{html.escape(res)}</div></div>")

    parts.append("<h2>S1 自检全部结果</h2>")
    if os.path.exists(s1_txt):
        rows = []
        with open(s1_txt) as f:
            for line in f:
                m = re.match(r"\[(PASS|INFO|FAIL)\s*\]\s+(\S+)\s+(.+?)\s{2,}(.*)", line.rstrip())
                if m:
                    lv, grp, chk, det = m.groups()
                    tag = {"PASS": "p", "INFO": "i", "FAIL": "f"}[lv]
                    rows.append(f"<tr><td><span class='tag {tag}'>{lv}</span></td>"
                                f"<td>{html.escape(grp)}</td><td>{html.escape(chk)}</td>"
                                f"<td><code>{html.escape(det)}</code></td></tr>")
        parts.append("<table><tr><th></th><th>组</th><th>检查项</th><th>实测</th></tr>"
                     + "".join(rows) + "</table>")
    parts.append("</div></html>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"WROTE {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
