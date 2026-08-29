#!/usr/bin/env python3
"""生成 S0–S3 的自包含 HTML 报告（`plan/06` §1 要求）。

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
        verify="这一步只回答一个问题：<b>这个执行器物理上能不能完成任务</b>，"
               "排除「它本身能力就不行」。它<b>不进入最终系统</b>，"
               "门槛是「能明确学会」，不要求完美策略（D-32）。"
               "<br><br>录像不是配图，是<b>独立的一道检查</b>——"
               "上一版策略成功率 100%，画面看着也正常，实测却发现"
               "抽屉打开过程中 60% 的时间根本没有接触。",
        objects=[
            ("执行器", "L 形钩杆，0 自由度，浮动底座由 PD 外力/力矩驱动"),
            ("动作", "6 维位姿增量（每步 ≤12 mm / 0.05 rad），经 PD 转成 wrench"),
            ("观测", "39 维特权观测：执行器状态、把手位置、开度、目标、接触摘要、物理参数"),
            ("抽屉阻尼", "<b>30 N·s/m</b>（原为 3.0，见下方「发现了什么」）"),
            ("训练", "PPO，2048 并行环境，第 100 轮存档；停训于第 171 轮时曲线仍在上升"),
        ],
        procedure=[
            "策略自主决定如何接近——没有脚本，没有预设轨迹",
            "确定性动作（推理模式，无探索噪声）",
            "评估与训练完全分离，各 256+ 条 episode",
            "<b>先查接触是否持续，再看成功率</b>",
        ],
        criterion="成功率显著高于零且随训练上升（D-32）；"
                  "且抽屉的运动必须由持续接触解释，不能靠惯性滑行",
        result="固定环境 43.1% / 随机环境 49.3%；抽屉运动期间 77% 的控制步有接触，"
               "70% 受力方向正确",
        verdict="Gate A 通过（新门槛），S2 到此为止", cls="ok",
        watch="钩杆的 L 形弯头朝向把手，抽屉被<b>逐步</b>拉开。"
              "对比上一版：那时钩杆撞一下就弹开，抽屉靠惯性滑完剩下的行程。",
        caveat="<b>这一版的数字比上一版低得多，但上一版是错的。</b>"
               "<br><br>上一版报 100%/100%，实际上抽屉阻尼只有 3.0，"
               "<b>滑行距离 v·m/c = 80 mm，而总行程才 180 mm</b>——"
               "策略学会了「捅一下让抽屉自己滑」，60% 的开启过程零接触。"
               "那不是操作，而且采出来的接触数据会是垃圾（`plan/02` §3.5 要求"
               "把接触力记进 Oracle Interaction Record）。"
               "<br><br>阻尼提到 30 后滑行降到 8 mm，任务真正变难了，"
               "所以成功率从 100% 掉到 43%。<b>43% 是真的，100% 不是。</b>"
               "<br><br>停训于第 171 轮（reward 70 仍在上升），"
               "因为诊断目的已达到——继续磕一个不进入最终系统的东西没有意义。",
    ),
]


# ---------------------------------------------------------------- S3
S3_TASKS = [
    ("抽屉 × 双板", "drawer", "800 条 / 739 成功（92.4%）",
     ["pinch_center", "pinch_offset", "hook_both", "asym_primary", "single_finger"],
     "五个策略家族。接触 <b>90.2% 落在把手横杆背面</b>、9.8% 在正面（拇指该在的地方），"
     "横杆上下表面 / 支撑柱 / 面板全为 0；<b>100% 落在两根支撑柱之间</b>。"
     "<code>single_finger</code> 只用一块板——<code>plan/02</code> §7 第 3 条"
     "「改变 source 板数量后表示维度不变」这条泄漏检查，只有它能提供数据。"),
    ("旋钮 × 双板", "knob", "2400 条 / 1693 成功（70.5%）",
     ["pin_pinch", "pin_push_single", "pin_push_dual", "pin_regrasp", "rim_only"],
     "销钉是竖直插在盘面上的拨杆，所以<b>推力方向必须随圆盘一起转</b>：参考系挂在"
     "销钉<b>当前的实际角度</b>上，力控方向取该处切向，每个控制步重算。"
     "实测力 <b>100% 打在销钉柱面</b>、<b>96.9~100% 打在推进方向的后侧</b>、"
     "接触法向只偏离圆盘切向 <b>2.5~11.6°</b>（挂在「领先角」上的错版本偏 42°，"
     "三分之一的力顶着销钉往圆心推）。"
     "<code>rim_only</code> 是<b>故意会失败</b>的对照：压足 19.5 N（安全上限 25 N 之内）"
     "只转到 0.019 rad，而销钉家族转到 2.3~2.8 rad —— τ_rim = 0.137 ≪ τ_need = 0.42 N·m。"
     "这是 D-14 第一次有<b>操作级</b>证据。"),
    ("擦拭 × 双板（主任务）", "wipe", "2700 条 / 2606 成功（96.5%）",
     ["tool_center", "tool_offset", "tool_tilt", "tool_heavy_slow",
      "tool_light_fast", "direct_wipe"],
     "六个家族，含<b>不用工具的直擦</b>——<code>plan/02</code> §7 第 8 条"
     "「envelope 与是否使用工具无关」只有它能验。两块板夹黑板擦的"
     "<b>两个 90×25 长侧面</b>（不是 45×25 的短端面）：扫掠沿 Y、长侧面的法向也是 Y，"
     "推力由<b>法向力直接传</b>；夹短端面时只能靠摩擦传，工具跟不住路径"
     "（滑脱 13 mm、路径区域内 68%；改完 0.6 mm / 92.5%，成功率 82.7% → 96.5%）。"
     "<b>录像里的褐色格子就是 dirt</b>，擦掉一格就消失一格。"),
    ("探针物体集（交互原语库）", "probe", "10740 条 / 9250 成功（86.1%）· 15/15 格",
     ["dial_crank", "flap_crank", "plunger_hook_pull", "ridge_rub", "roller_poke",
      "slider_hook_pull", "ball_roll", "ball_twist", "slider_pinch_move"],
     "E-I 执行器要学的是「我这个形态能实现哪些交互」，不是「怎么做这三个任务」。"
     "物体集从一套<b>独立于本项目三个任务</b>的交互分类学推出（Huang 的接触模式枚举 + "
     "Bullock/Ma/Dollar 的手中心描述子 + Lynch &amp; Mason 的非抓握原语谱系），"
     "判据是<b>是否张满该分类学</b>，不是「是否覆盖留出任务」。另加一条硬规则："
     "<b>每条原语至少两个几何不同的物体承载</b>——满足它之后，"
     "「你是照着任务 X 设计物体 Y 的」这句指控就失去力量：删掉 Y，那一格照样在。"
     "10 个物体、15 条原语，<b>15/15 全部满足两条判据</b>。"),
]


def s3_section(web_dir):
    P = ["<h2>S3 · 模拟人采集示教</h2>",
         "<p class=lead>双板采集器扮演「人」，产生<b>多样的</b>示教。多样性不是装饰："
         "S5 要从多条不同做法里归纳出<b>共享的</b>交互要求，"
         "如果所有示教本来就一样，归纳就没有内容。</p>",
         "<div class=note><b>为什么每个任务都要另配一张核对图。</b>"
         "录像能看出板在动、在推、在擦，<b>看不出力打在物体的哪一侧、方向偏了多少</b>。"
         "旋钮的全部力学就在这两件事上——所以 <code>s3_knob_contact.py</code> 在物体局部系里"
         "统计「打在销钉哪一侧 / 法向偏离切向多少度」；擦拭的 effect 只有 dirt 变化，"
         "所以有 dirt 覆盖图（现在录像里也直接画出来了）。"
         "<b>这些都是数字发现不了、也是单看录像发现不了的，两者缺一不可。</b></div>"]
    for title, key, vol, fams, blurb in S3_TASKS:
        P.append(f"<h3 style='font-size:17px;margin:26px 0 6px'>{title}</h3>")
        P.append(f"<p class=sub>{vol}</p><p class=lead>{blurb}</p>")
        P.append("<div class=pair>")
        for f in fams:
            pre = {"drawer": "drawer_", "knob": "knob_", "wipe": "wipe_"}.get(key, "")
            path = os.path.join(web_dir, f"{pre}{f}.mp4")
            if not os.path.exists(path):
                continue
            P.append(f"<div class=card><h3 style='font-size:14px'>{f}</h3>"
                     f"<video controls loop muted playsinline preload=metadata "
                     f"src='data:video/mp4;base64,{b64(path)}'></video></div>")
        P.append("</div>")

    P.append("<h3 style='font-size:17px;margin:26px 0 6px'>数据划分</h3>")
    P.append("<p class=lead><code>plan/03</code> §7 要一个校准集 + <b>五个</b>冻结测试集。"
             "校准集必须独立于训练集和所有测试集——用训练集标定 conformal 阈值会让"
             "覆盖率保证失效。划分一律<b>按 episode</b>，不按帧。</p>")
    P.append("""<table><tr><th>划分</th><th>旋钮</th><th>擦拭</th><th>抽屉</th></tr>
<tr><td>train</td><td>652</td><td>856</td><td>323</td></tr>
<tr><td>calibration</td><td>87</td><td>127</td><td>43</td></tr>
<tr><td>in_distribution_test</td><td>130</td><td>190</td><td>64</td></tr>
<tr><td><b>unseen_geometry_test</b></td><td><b>204</b></td><td><b>297</b></td><td><b>83</b></td></tr>
<tr><td>unseen_physics_test</td><td>173</td><td>219</td><td>66</td></tr>
<tr><td>unseen_strategy_test</td><td>447</td><td>450</td><td>160</td></tr>
<tr><td><b>unseen_implementation_test</b></td><td>—</td><td><b>373</b></td><td>—</td></tr>
<tr><td>failed</td><td>707</td><td>94</td><td>61</td></tr></table>
<p class=lead style="margin-top:14px"><b>失败样本单独存、不进任何测试集。</b>
一条失败的示教不是示教，把它放进 <code>unseen_strategy_test</code>
会让那个集合的泛化数字失去意义。<code>rim_only</code> 那 480 条是<b>设计上就该失败</b>的对照。</p>
<div class=cav><b>⚠ 注意</b> — 几何变体靠 <code>MultiUsdFileCfg</code> 按 env 轮转混采，
而它<b>要求 <code>replicate_physics=False</code></b>：为 True 时 Isaac Lab 把 env_0 的物理
整体复制给所有 env，多资产被抹平。实测 24 个 env 全部拿到名义几何，
<b>而代码以为其中 4 个是变体，标签会照常写出来、全是假的</b>。
改完实测接触半径 47.9 / 50.6 / 59.6 mm 随三档销钉偏心清楚分开。</div>""")

    P.append("<h3 style='font-size:17px;margin:26px 0 6px'>验收</h3>")
    P.append("""<table><tr><th>判据</th><th>出处</th><th>实测</th></tr>
<tr><td>数据集独立验收（划分不重不漏 / 留出集干净 / SHA-256 / 字段隔离）</td>
<td><code>03</code> §6–§7</td><td><span class='tag p'>PASS</span> 旋钮 18 · 擦拭 21 · 抽屉 18 项，0 FAIL</td></tr>
<tr><td>策略分类器在<b>原始 source 动作</b>上显著高于随机</td><td><code>03</code> §4</td>
<td><span class='tag p'>PASS</span> 旋钮 1.000 / 擦拭 0.987 / 抽屉 0.742（随机 0.25 / 0.17 / 0.20）</td></tr>
<tr><td>原语库张满 + 每条原语 ≥2 个几何不同的承载物体</td><td><code>03</code> §2.4</td>
<td><span class='tag p'>PASS</span> 15/15</td></tr>
<tr><td>两块板的朝向标记一致（录像可读）</td><td><code>06</code> §7</td>
<td><span class='tag p'>PASS</span> 62 个家族/原语，矛盾 0 个</td></tr>
<tr><td>旋钮接触部位与受力方向</td><td><code>03</code> §4</td>
<td><span class='tag p'>PASS</span> 销钉 100% · 推进后侧 96.9~100% · 法向偏切向 2.5~11.6°</td></tr>
</table>
<p class=lead style="margin-top:14px">⚠️ 分类器这一项只证明<b>动作层面确实不同</b>。
<code>plan/02</code> §7 第 4 条要的是「从 envelope 预测策略身份<b>显著更难</b>」，
要等 S4/S5 有了 envelope 才能做，届时两个数并排报，一高一低才说明表示抹掉了策略特异性。</p>""")
    return "".join(P)


def main(vid_dir, s1_txt, out_path):
    P = [f"<!doctype html><html lang=zh><meta charset=utf-8>",
         "<title>S0–S3 验证报告 · Functional Interaction Transfer</title>",
         "<meta name=viewport content='width=device-width,initial-scale=1'>",
         f"<style>{CSS}</style><div class=wrap>",
         "<h1>S0 / S1 / S2 / S3 验证报告</h1>",
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
<th>终止开度(中位)</th></tr>
<tr><td>固定</td><td><b>43.1%</b></td><td>0%</td><td>56.9%</td><td>116.6 mm</td></tr>
<tr><td>随机化</td><td><b>49.3%</b></td><td>0%</td><td>50.7%</td><td>120.9 mm</td></tr>
</table>
<p class=lead style="margin-top:14px">目标区间 100–160 mm，episode 上限 400 步，
各 256+ 条 episode。失败全部是超时，<b>零次飞出边界</b>——策略不会把事情弄坏，
只是不够快。</p>
<h3 style="font-size:16px;margin:24px 0 8px">接触持续性（决定性检查）</h3>
<table><tr><th></th><th>上一版（阻尼 3）</th><th>本版（阻尼 30）</th></tr>
<tr><td>开启过程中<b>无接触</b>占比</td><td><b>60%</b></td><td><b>22.8%</b></td></tr>
<tr><td>受力方向正确（+X 推开）</td><td>—</td><td><b>70%</b></td></tr>
<tr><td>运动期间平均接触力</td><td>—</td><td>12.7 N</td></tr>
</table>
<p class=lead style="margin-top:14px">抽屉沿 +X 打开。作用在它身上的力若是 +X 才是正常推开；
若无接触仍在运动，那是惯性滑行。<b>这项检查是数字发现不了的，
只能逐控制步记录接触力与速度。</b></p>""")

    P.append(s3_section(os.path.join(os.path.dirname(vid_dir), "s3_web")))

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
