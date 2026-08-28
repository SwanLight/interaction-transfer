#!/usr/bin/env python3
"""生成 S0/S1 的自包含 HTML 报告（`plan/06` §1 要求）。

视频以 base64 data URI 内嵌，**整个报告是单个文件**——直接双击、在 IDE 预览、
发给别人都能播。早期版本用相对路径引用 mp4，在 VSCode 预览的沙箱里只显示
黑框、播放键点不动。

每个实验都要写清楚：**验证什么、用什么物体、参数多少、怎么做的、判据是什么**。
报告是给别人看的，不是给自己对数字的。

用法：python3 tools/make_report.py out/s0 log/s1/s1_report.txt out/report.html
"""
import base64
import html
import json
import os
import re
import sys

MM = "&thinsp;mm"

SCENES = [
    dict(
        key="knob_rim", group="core", title="旋钮 · 接触轮缘（低摩擦区）",
        verify="在安全法向力上限内，<b>只接触轮缘</b>能否把圆盘转到目标角度。"
               "这是 D-14 的一半——如果轮缘也能转动圆盘，那么「该在哪里接触」这条信息"
               "就是多余的，旋钮任务上无法检验 interaction region。",
        objects=[
            ("圆盘", "半径 70 mm，厚 15 mm，质量 0.35 kg"),
            ("轮缘材质", "μ = <b>0.10</b>（低摩擦，图中<span class=sw style='--c:#3b78d8'></span>蓝色）"),
            ("偏心销钉", "直径 20 mm，长 48 mm，距轴心 52 mm，μ = 0.80（<span class=sw style='--c:#e08a1a'></span>橙）"),
            ("立柱", "半径 30 mm，高 30 mm —— 把圆盘抬离底座，否则径向够不到轮缘"),
            ("底座", "170 × 170 × 40 mm，固定"),
            ("转动关节", "revolute，绕 Z 轴，粘性阻尼 0.28 N·m·s/rad，限位 [-0.175, 3.491] rad"),
            ("推子", "30 mm 立方，0.2 kg，μ = 0.9（<span class=sw style='--c:#d9d234'></span>黄）"),
        ],
        procedure=[
            "关节归零，并清掉上一阶段残留的力矩指令",
            "位置 PD 把推子送到距轮缘 4 mm 处",
            "沿径向（指向轴心）力控压紧，力在 1 s 内斜坡升到 <b>25 N</b>（安全上限）",
            "位置目标<b>沿圆周</b>推进 2.0 rad，把推子拖着蹭轮缘 —— 力矩只能靠摩擦传递",
            "圆盘转到 1.8 rad 即撤力（本例不会触发）",
        ],
        criterion="Δθ &lt; 1.0 rad（目标区间下限）即为「推不动」，符合预期",
        result="Δθ = +0.025 rad", verdict="推不动 —— 符合预期", cls="ok",
        watch="盯住<b>橙色销钉的位置</b>：三帧完全不变，说明圆盘没有转动。"
              "推子在失去接触后会飘开，那是力控没有位置参考所致，不影响结论。",
    ),
    dict(
        key="knob_pin", group="core", title="旋钮 · 接触销钉（高摩擦区）",
        verify="同一份物理、同样 25 N 上限，只把接触点换到销钉。"
               "与上一条构成对照：<b>唯一的变量是接触在哪里</b>。",
        objects=[("物体", "与上一条完全相同，未改动任何参数")],
        procedure=[
            "前两步与上一条相同",
            "改为<b>切向直推</b>销钉侧面 —— 法向力<b>直接</b>产生力矩 τ = F·r，与摩擦系数无关",
            "圆盘转到 1.8 rad 即撤力，避免推子被持续加速飞出画面",
        ],
        criterion="Δθ ≥ 1.0 rad 即为「推得动」",
        result="Δθ = +1.516 rad", verdict="转进目标区间 1.0–2.2 —— 符合预期", cls="ok",
        watch="橙色销钉从右前方转到后方，圆盘明显转动。"
              "与上一条唯一的差别就是接触区域，因此「该在哪里接触」是必需信息。",
        caveat="<b>这个转角数字是脆弱的，不要引用它。</b>开环直线推法下，最终转角对推力"
               "<b>非单调</b>：10 N→0.234、15 N→0.392、20 N→<b>0.217</b>、25 N→1.516 rad。"
               "结果不由「力够不够」决定，而由<b>接触维持多久</b>决定——力越大推子加速越快、"
               "越早脱离销钉，传递的冲量反而可能更少。"
               "<br><br>这不影响本项目的结论：判据用的是<b>接触期间传递到轴的力矩</b>"
               "（τ_rim = 0.170 &lt; τ_need = 0.420 &lt; τ_pin = 23.5 N·m），"
               "在接触瞬间测量，与接触持续多久无关。转角只是给人看的示意。"
               "也不是任务设计的问题——RL 学出的策略是闭环的、会主动维持接触，"
               "只有这段开环脚本才会脱开。",
    ),
    dict(
        key="wiping_padrod", group="rest", title="擦拭 · 垫头杆直擦",
        verify="主任务的力学基础：执行器能否在指定法向力区间内、"
               "在平面上稳定滑移而不跳动、不穿模。这是 Contact Mode（stick/slide）"
               "和 Interaction Region 两个字段能否提取的前提。",
        objects=[
            ("平面", "600 × 500 × 20 mm，<b>kinematic 刚体</b>（不能用静态碰撞体，否则接触过滤失效），μ = 0.35"),
            ("垫头杆", "主杆 φ16 × 200 mm + 末端平垫 40 × 30 × 8 mm，0 自由度，0.3 kg，μ = 0.5"),
            ("摩擦组合", "min 模式 → 实际接触 μ = min(0.35, 0.5) = 0.35"),
        ],
        procedure=[
            "位置 PD 把杆落到板面上方 6 mm，姿态锁竖直",
            "Z 轴切换为<b>纯力控</b>，压出目标法向力 5.5 N（工作区间 3–8 N 的中点）",
            "XY 仍为位置 PD，目标以 0.05 m/s 平移 —— 混合力/位控",
            "滑移 4 s 后再采样 1 s，统计法向力的均值与标准差",
        ],
        criterion="法向力落在 3–8 N；速度跟踪准确；v_z 接近 0；力的变异 &lt; 35%",
        result="Fn = 5.456 ± 0.369 N（变异 6.8%），|v_xy| = 0.0498 m/s（指令 0.05），v_z = −0.011 m/s",
        verdict="全部达标", cls="ok",
        watch="杆保持竖直、贴着板面匀速右移，没有跳动或陷入板内。",
    ),
    dict(
        key="cabinet_travel", group="rest", title="抽屉 · 全行程往返",
        verify="抽屉是<b>阴性对照</b>任务（Gate G）——预期「只给物体结果」就足够，"
               "不需要额外的交互信息。这里只验证几何与动力学没问题。",
        objects=[
            ("面板", "300 × 180 × 18 mm"),
            ("把手", "杆 φ22 × 140 mm，与面板净空 <b>45 mm</b>（供手指/钩杆伸入），支撑柱间距 125 mm"),
            ("抽屉体", "托盘深 250 mm，柜体五面板围成"),
            ("滑动关节", "prismatic 沿 X，行程 0–180 mm，阻尼 3.0"),
        ],
        procedure=["+30 N·m 拉开到底", "−30 N·m 推回", "检查是否越限、是否穿模"],
        criterion="能到 ≥160 mm（任务目标 100–160）且不超过 180 mm 上限；能推回 0",
        result="拉开 180.0 mm（上限 180），推回 0.0 mm",
        verdict="全行程可达，不越限", cls="ok",
        watch="抽屉平稳滑出再收回，托盘不穿过柜体侧板。",
    ),
    dict(
        key="hook_sweep", group="rest", title="钩杆 · 绕轴扫掠",
        verify="第三个执行器的<b>几何可行性</b>：一个 0 自由度的 L 形杆"
               "能否勾住销钉并传递足够力矩。注意判据是<b>能否传递力矩</b>，"
               "不是这段开环脚本能转多少度 —— 后者是控制问题，属 S2 由 RL 解决。",
        objects=[
            ("钩杆", "主杆 φ16 × 250 mm + 横钩 50 mm，L 形，<b>0 自由度</b>，0.25 kg"),
            ("旋钮", "与前两条相同"),
        ],
        procedure=[
            "PD 把横钩摆到销钉切向后方，姿态随扫掠角一起绕 Z 旋转（否则横钩朝向固定，只能擦边）",
            "绕圆盘轴画弧 2.0 rad，角速度 0.24 rad/s",
            "全程记录接触点数与传递到轴的力矩峰值（排除撞击尖峰）",
        ],
        criterion="τ_hook &gt; τ_need = 0.420 N·m，且转矩方向正确",
        result="τ_hook = 0.515 N·m &gt; 0.420；接触点峰值 3；Δθ = +0.167 rad（方向正确）",
        verdict="几何可行性成立", cls="ok",
        watch="黄色钩杆绕着圆盘画弧，横钩跟着转向以保持径向。中途会脱开 —— "
              "那是开环脚本的局限，不是形态不可行。",
    ),
    dict(
        key="slider_pretrain", group="rest", title="预训练物体集 · 滑块导轨",
        verify="`plan/03` §2.4 的预训练物体之一。它<b>只用于给 E-I 执行器产生"
               "交互指令的多样性，永远不作为任务评估</b>——没有它，"
               "「留出任务零样本」就退化成三个任务之间的插值。",
        objects=[
            ("导轨", "300 × 40 × 20 mm，固定"),
            ("滑块", "60 × 50 × 40 mm，0.4 kg"),
            ("滑动关节", "prismatic，行程 150 mm，阻尼 2.0（每 episode 可随机化）"),
        ],
        procedure=["施加 15 N 沿轨推动"],
        criterion="能推满行程",
        result="位移 150.0 mm", verdict="可用", cls="ok",
        watch="紫色滑块沿灰色导轨推到底。",
    ),
]

CSS = """
:root{--bg:#fbfbfd;--fg:#1a1a1e;--mut:#63636d;--line:#e4e4ea;--card:#fff;
      --ok:#0a7d4f;--okbg:#e9f6f0;--warn:#8a5a10;--warnbg:#fdf3e2;--code:#f3f3f7;
      --accent:#2b5fd9}
@media (prefers-color-scheme:dark){:root{--bg:#121216;--fg:#e9e9ee;--mut:#9a9aa5;
  --line:#2b2b33;--card:#1a1a20;--ok:#4cdb99;--okbg:#112a20;--warn:#e8b463;
  --warnbg:#2a2010;--code:#222228;--accent:#7fa4ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:44px 24px 90px}
h1{font-size:30px;margin:0 0 8px;letter-spacing:-.015em}
h2{font-size:21px;margin:52px 0 8px;padding-bottom:9px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 6px;font-size:14px}
.lead{color:var(--mut);margin:0 0 30px;font-size:14.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      padding:22px;margin:0 0 24px}
.card h3{margin:0 0 10px;font-size:18px;letter-spacing:-.01em}
.blk{margin:16px 0 0}
.blk>b{display:block;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;
       color:var(--mut);margin:0 0 6px}
video{width:100%;border-radius:10px;background:#111;display:block;margin:16px 0 0}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:5px 16px;font-size:14px}
dt{color:var(--mut);white-space:nowrap}
dd{margin:0}
ol{margin:0;padding-left:20px;font-size:14px}
ol li{margin:3px 0}
.res{margin-top:16px;padding:12px 15px;border-radius:9px;font-size:14px}
.res.ok{background:var(--okbg);color:var(--ok)}
.res .v{font-weight:700}
.res code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.watch{margin-top:12px;padding:11px 14px;border-radius:9px;background:var(--code);
       font-size:13.5px;color:var(--mut)}
.watch b:first-child{color:var(--fg)}
.cav{margin-top:11px;padding:12px 15px;border-radius:9px;background:var(--warnbg);
     color:var(--warn);font-size:13.5px;line-height:1.65}
.cav b{color:var(--warn)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:860px){.pair{grid-template-columns:1fr}}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 15px;color:var(--mut);
      font-size:14px;margin:0 0 26px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12.5px}
td code{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--mut)}
.tag{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11.5px;
     font-family:ui-monospace,monospace;font-weight:600}
.tag.p{background:var(--okbg);color:var(--ok)}
.tag.i{background:var(--code);color:var(--mut)}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;background:var(--c);
    margin-right:3px;vertical-align:baseline}
.env{font-size:13px;color:var(--mut)}
.env code{background:var(--code);padding:1px 6px;border-radius:4px;
          font-family:ui-monospace,Menlo,monospace;font-size:12px}
"""


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def card(s, vid_dir):
    p = os.path.join(vid_dir, f"{s['key']}_web.mp4")
    v = (f"<video controls loop muted playsinline preload=metadata "
         f"src='data:video/mp4;base64,{b64(p)}'></video>") if os.path.exists(p) else \
        "<p class=sub>（视频缺失）</p>"
    objs = "".join(f"<dt>{k}</dt><dd>{v2}</dd>" for k, v2 in s["objects"])
    proc = "".join(f"<li>{x}</li>" for x in s["procedure"])
    cav = (f"<div class=cav><b>⚠ 注意</b> — {s['caveat']}</div>") if s.get("caveat") else ""
    return f"""<div class=card><h3>{s['title']}</h3>
<div class=blk><b>验证什么</b>{s['verify']}</div>
<div class=blk><b>物体与参数</b><dl>{objs}</dl></div>
<div class=blk><b>怎么做的</b><ol>{proc}</ol></div>
<div class=blk><b>判据</b>{s['criterion']}</div>
{v}
<div class='res {s["cls"]}'><span class=v>{s['result']}</span> &nbsp;→&nbsp; {s['verdict']}</div>
<div class=watch><b>看什么</b> — {s['watch']}</div>{cav}</div>"""


S2_SCENES = [
    dict(
        key="expert_drawer_hook", group="s2", title="抽屉 · 钩杆 Privileged Expert",
        verify="训练好的策略在<b>做什么</b>。成功率 100% 不等于做对了——"
               "策略完全可能找到物理漏洞（穿模、抖动、利用求解器瑕疵），"
               "那些只能看出来。`plan/06` §7 因此把人工看视频列为必经步骤。",
        objects=[
            ("执行器", "L 形钩杆，0 自由度，浮动底座由 PD 外力/力矩驱动"),
            ("动作", "6 维位姿增量（每步 ≤12 mm / 0.05 rad），经 PD 转成 wrench"),
            ("观测", "39 维<b>特权观测</b>：执行器状态、把手位置、开度、目标、接触摘要、物理参数"),
            ("训练", "PPO（rsl-rl 3.0.1），2048 并行环境，<b>只训 100 轮</b>"),
        ],
        procedure=[
            "策略自主决定如何接近——没有脚本，没有预设轨迹",
            "确定性动作（推理模式，无探索噪声）",
            "4 个并行 env 同时录制，episode 完成后自动重置",
        ],
        criterion="画面里必须看到：主杆插进把手净空、沿 +X 拉、抽屉正常滑出，"
                  "且无穿模/瞬移/异常抖动",
        result="4/4 成功，首次成功在第 52 / 51 / 264 / 54 控制步",
        verdict="动作合理，与设计的机制一致", cls="ok",
        watch="钩杆的<b>竖直主杆</b>插进把手与面板之间的 45 mm 净空，然后沿 +X 拉——"
              "拉力靠主杆前面压把手背面，横钩只负责不碰撞。这正是脚本验证时设计的机制，"
              "策略自己学到了同一个解法。",
    ),
]


def main(vid_dir, s1_txt, out_path):
    P = [f"<!doctype html><html lang=zh><meta charset=utf-8>",
         "<title>S0/S1/S2 验证报告 · Functional Interaction Transfer</title>",
         "<meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<style>{CSS}</style><div class=wrap>",
         "<h1>S0 / S1 / S2 验证报告</h1>",
         "<p class=sub>Functional Interaction Transfer —— 资产可行性自检、可视化链路、"
         "与第一个 Privileged Expert</p>",
         "<p class='sub env'>Isaac Sim <code>5.1.0-rc.19</code> + Isaac Lab <code>2.3.1</code>"
         "（快照 <code>2ab57ade</code>）· 8 × RTX 4090 · 物理步长 1/120 s</p>",
         "<p class=lead>本项目要验证的想法是：不迁移人的动作，而迁移动作所实现的"
         "<b>功能性交互</b>，再由不同末端执行器根据自身形态各自重新实现。"
         "在训练任何模型之前，必须先确认仿真里的物体和执行器在几何上、动力学上"
         "都能支撑这套实验设计 —— 这就是 S1，本报告是它的结果。</p>",
         "<div class=note><b>为什么必须看视频。</b>S1 过程中出现过多次<b>假 PASS</b>："
         "第一版旋钮标定报出「轮缘推不动」，实际是推子被底座卡住，不是打滑；"
         "还有一次两种接触方式被错误地写成同一条轨迹，力恰好穿过转轴，"
         "力矩恒为零，看起来像销钉也推不动 —— <b>与真实结论完全相反</b>。"
         "这些数字都发现不了，看一眼画面就能发现。"
         "<code>plan/06</code> §7 因此把人工看视频列为每次正式评估的必经步骤。</div>",
         "<h2>怎么读这份报告</h2>",
         "<p class=lead>每个场景都写了<b>验证什么 / 物体与参数 / 怎么做的 / 判据 / 实测 / 看什么</b>。"
         "凡是标了 <span style='color:var(--warn)'>⚠ 注意</span> 的地方，是这个数字有已知的局限，"
         "读的时候要带着那条限制。</p>",
         "<h2>核心对照 · 接触区域的必要性</h2>",
         "<p class=lead>下面两段是<b>同一个物体、同一份物理参数、同样 25 N 法向力上限</b>，"
         "<b>唯一的变量是接触在哪里</b>。如果两者结果相同，说明「该在哪接触「这条信息是多余的，"
         "旋钮任务就无法用来检验 interaction region，实验设计需要改。</p><div class=pair>"]

    for s in SCENES:
        if s["group"] == "core":
            P.append(card(s, vid_dir))
    P.append("</div>")

    P.append("<h2>其余场景</h2>")
    for s in SCENES:
        if s["group"] != "core":
            P.append(card(s, vid_dir))

    P.append("<h2>S2 · 训练好的策略</h2>")
    P.append("<p class=lead>Privileged Expert 的作用是<b>排除「这个执行器本身做不到」</b>，"
             "它不进入最终系统。门槛是「能明确学会」，不要求完美策略（D-32）。</p>")
    s2dir = os.path.join(os.path.dirname(vid_dir), "s2_expert")
    for sc in S2_SCENES:
        P.append(card(sc, s2dir))
    P.append("""<table><tr><th>环境</th><th>成功率</th><th>飞出边界</th><th>超时</th>
<th>终止开度</th><th>用时</th></tr>
<tr><td>固定</td><td><b>100.0%</b></td><td>0%</td><td>0%</td>
<td>128.9 mm</td><td>59.8 步</td></tr>
<tr><td>随机化</td><td><b>100.0%</b></td><td>0%</td><td>0%</td>
<td>127.7 mm</td><td>61.2 步</td></tr></table>
<p class=lead style="margin-top:14px">目标区间 100–160 mm，episode 上限 400 步，
各 256+ 条 episode。<b>Gate A 通过。</b></p>""")

    P.append("<h2>S1 自检完整结果</h2>")
    P.append("<p class=lead>共 35 项，33 项 PASS、2 项 INFO（仅记录测量值，无判据）、0 项 FAIL。"
             "由 <code>tools/s1_all.sh</code> 生成，每项检查独立进程。</p>")
    if os.path.exists(s1_txt):
        rows = []
        for line in open(s1_txt):
            m = re.match(r"\[(PASS|INFO|FAIL)\s*\]\s+(\S+)\s+(.+?)\s{2,}(.*)", line.rstrip())
            if m:
                lv, grp, chk, det = m.groups()
                rows.append(f"<tr><td><span class='tag {lv[0].lower()}'>{lv}</span></td>"
                            f"<td>{html.escape(grp)}</td><td>{html.escape(chk)}</td>"
                            f"<td><code>{html.escape(det)}</code></td></tr>")
        P.append("<table><tr><th></th><th>组</th><th>检查项</th><th>实测</th></tr>"
                 + "".join(rows) + "</table>")
    P.append("</div></html>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w").write("\n".join(P))
    print(f"WROTE {out_path} ({os.path.getsize(out_path)/1024:.0f} KB, 自包含)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
