# out/ —— 给人看的产物

**一个 S 一个目录，按步骤编号。** 后面 S3、S4 加进来照此规则。

| | 内容 | 状态 |
|---|---|---|
| `report.html` | **总报告，双击就能看。** 视频与 region 热图全部内嵌，单文件 ~10 MB | 跨 S0–S4 |
| `plate_orient.txt` | **采集板朝向标记的一致性核对**，四个任务 62 个家族/原语 | ✅ 矛盾 0 个 |
| `diversity.txt` | **策略多样性**（`plan/03` §4）：三个任务的分类器准确率与混淆矩阵 | ✅ 抽屉 0.742 / 擦拭 0.987 / 旋钮 1.000 |
| `s3_web/` | 报告内嵌用的压缩录像（25 段，276 KB），由 `tools/s3_web_videos.sh` 生成 | ✅ |
| `s0_pipeline/` | 可视化管线（产物是能力本身，见其 README） | ✅ |
| `s1_assets/` | 资产与场景录像 + 51 项自检结果 | ✅ 49 PASS / 2 INFO / 0 FAIL |
| `s2_expert/` | 训练好的策略在跑什么 | ✅ |
| `s3_source/` | **抽屉双板 source 示教**：录像、接触部位分布、数据集验收 | ✅ |
| `s3_probe/` | **探针物体集（交互原语库）**：覆盖核对、六条原语录像 | ✅ **15/15 格** |
| `s3_wipe/` | **擦拭（主任务）**：dirt 覆盖图、录像、验收 | ✅ |
| `s3_knob/` | **旋钮**：接触部位与受力方向核对、录像、验收 | ✅ |
| `s4_records/` | **Oracle Interaction Record**：提取 / 独立验收 / 泄漏检查 / region 探针四套报告 + 热图 | ✅ 泄漏检查 **0 FAIL**（抽屉 9 / 擦拭 10 / 旋钮 9 条 PASS，第 4、8 条如实 DEFER 到 S5） |

> **`out/` 是产物**，给别人看的。
> **`log/` 是过程记录**（进度、决策、踩坑），给自己和接手的人看。
> 有的东西两边都有：S1 的自检表原始输出在 `log/s1/`，报告里嵌了一份。
>
> **哪些进版本控制**：这个目录里的 `README.md` 与全部 `*.txt` 摘要**进 git**
> （63 个文件 292 KB）——README 是手写的，txt 是 P-56 要求的"每个引用的数字
> 都得有产物文件"的那些产物，两类都不可重新生成。录像 `*.mp4`、热图 `*.png`
> 和 `report.html`（117 MB）不进，它们跑一遍 `tools/make_report.py` 就有。
> D-23 补记里写了为什么原来那条 `out/` 全排除是错的。

## 十二个步骤

前三步是准备，**S7 才是本工作要证明的东西**。

```
S0  能把画面录出来          ✅
S1  资产做得对不对          ✅
S2  执行器物理上能否做到     ✅
S3  模拟人采集示教          ✅  （抽屉 / 旋钮 / 擦拭 / 探针集 15/15 / 多样性）
S4  从示教里提取交互信息    ✅
S4.6 装置能测的够不够用     ✅  （实验七：硬件指标 = 功能特征尺寸的 ~10%）
S5  归纳共享的交互要求   ← 现在这里
S6  训练执行器
S4.5 表示分辨率定多少       ⛔  判据是下游成功率，**要 S6 的 checkpoint**，
                               编号排在 S4 后面是错的（见 plan/05 §1）
S7  留出任务零样本  ← 主张所在
S8  信息条件对照
S9  扰动恢复
S10 跨形态 + 反事实
```

## 重新生成

```bash
# S1 场景录像
./tools/sync.sh && ./tools/run_remote.sh "bash tools/s0_all.sh /tmp/s0" s0
scp root@10.0.6.98:'/tmp/s0/*.mp4' out/s1_assets/

# S2 策略录像
./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh tools/s2_record.py \
    --run <运行名> --ckpt model_100.pt --out /tmp/s2vid" vid
scp root@10.0.6.98:'/tmp/s2vid/*.mp4' out/s2_expert/

# S3 source 采集与录像
./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \
    tools/s3_source_drawer.py --envs 80 --batches 10 --out /tmp/s3_drawer_v3" s3full
# 逐家族录像与验收见 out/s3_source/README.md

# S4 提取与四类检查（判据是**退出码**，不是报告开头有没有 PASS —— P-55）
./tools/run_remote.sh "bash tools/s4_all.sh" s4all              # 只跑验收，约 25 min
./tools/run_remote.sh "bash tools/s4_all.sh --extract" s4all    # 连提取一起重跑
./tools/run_remote.sh "bash tools/s4_all.sh --only region_probe" s4probe   # 只重跑一类
scp root@10.0.6.98:'/tmp/s4_reports/*.txt' out/s4_records/

# 总报告（第一个参数是**录像目录**，传错了不报错，只会静静变成"（视频缺失）"）
python3 tools/make_report.py out/s1_assets log/s1/s1_report.txt out/report.html
```

> **数据集本身不进版本控制**（D-23）。S3 的 800 条 episode 留在服务器
> `/tmp/s3_drawer_v3`，`out/s3_source/` 只放摘要、录像和验收表。
