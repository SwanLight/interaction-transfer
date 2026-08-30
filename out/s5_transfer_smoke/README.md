# S5 statistical interaction transfer — 本地小样本 smoke（`interaction-transfer-v2`）

这是 S5 数据契约在**真实 S4 records** 上的 smoke，**不是**完整数据集结果，也不是
Shared Structure Model 的性能结论。样本从服务器训练集按 source strategy 均衡抽取。
完整 train split 的 artifact 与冻结留出划分上的评估在 `out/s5/`。

每条 episode 先投影到"**按 phase 分段、段内按交互活动量**"的命令轴上（D-65），
再跨 episode 等权统计；每个 surface cell 上的 engage / mode / traction /
moment density 只用**实际接触过该 cell 的 episode 和帧**（D-66）。

| 任务 | 成功示教 | 策略 | phase 格数 | 空格 | occupied cell | 每 cell 支持中位数 | artifact SHA-256 |
|---|---:|---:|---|---:|---:|---:|---|
| 抽屉 | 12 | 4 | 2/3/25/2 | 0 | 164 | 5 / 12 | `bd4359fc…` |
| 擦拭 | 12 | 4 | 2/4/23/3 | 0 | 373 | 5 / 12 | `cbf391dd…` |
| 旋钮 | 9 | 3 | 2/2/26/2 | 0 | 141 | 5 / 9 | `3a59c5b2…` |

**全量结果在 `out/s5/README.md`，引用数字以那份为准。** 这里只是 33 条真实记录上的
契约 smoke，用来在本机改完代码立刻验链路。

**"每 cell 支持中位数 5/12"是这份 smoke 最该被看见的数**，而且它在 v1 里根本没有
落盘：产物只有逐格的 `support/episodes`，下游完全分不出"12 个采集者都压在这里"
与"只有 1 个人蹭到过这里"。v2 用 `region/support` 与 `region/duty` 把它显式化。

## v2 修掉的两个 v1 缺陷（都不报错，只让指令失效）

**一、命令轴退化（P-58 / D-65）。** v1 按任务完成度 `progress` 分格，而接近段恒为 0、
到达目标后恒为 1。实测**抽屉 83.9%、旋钮 73.3% 的帧落进 32 格里的 2 格**，中间 30 格
每格中位数 2 帧。当时所有校验都通过、`empty_bins` 报 0。
五个候选对齐的实测对比见 `out/s5_align/probe.txt`。

**二、同一个 cell 的字段来自不同的 episode 子集（P-59 / D-66）。** v1 的 region /
engage / mode 用 NaN 排除没碰过的 episode，traction 用 0 把它们算进去。结果：

| 任务 | v1 中"该接触但力中位数恰为零"的 occupied cell 占比 |
|---|---:|
| 抽屉 | **57.9%** |
| 擦拭 | **80.7%** |
| 旋钮 | **48.7%** |

v2 后该比例为 **0，且由 `validate()` 的硬约束保证**。

## 一处必须更正的旧说法（P-60）

v1 的这份 README 曾用"最大 6D wrench 重建误差 6.18e-6 N"证明 mechanics 契约成立。
**那个对拍是代数恒等式**——cell 内的局部力矩加上 `cross(representative, F)` 恒等于
直接对接触点求力矩，换成任何错误的 cell 划分那个数都还是 1e-6。它不能失败，所以
它不是验证。

现在报的是**表面投影完整性**，两个都能失败：

这 33 条样本上三个任务的"被滤掉的力占比"都是 0.0000。**但这正说明小样本靠不住**——
同一个诊断在**全量** train split 上，抽屉是 **2.11%**（与 S4 独立记的 wrench 差 5.65 N），
见 `out/s5/README.md`。能失败的诊断才有用，而它确实失败了。

**并且**：跨 episode 取中位数/分位数之后 6D wrench 守恒**不再成立**（中位数非线性）。
守恒只在聚合前的单条 episode 上成立。不得再写"积分恢复 6D wrench"。

## 这份 smoke 不能说明什么

- 只有 33 条 episode、全部来自 train split、全部是 nominal 几何；
- 10/90 分位数是描述性统计，**不是 conformal guarantee**（D-59/D-63）；
- 没有在冻结的 calibration / test 划分上测 coverage、width、策略子群落差；
- 擦拭样本里只抽到了四个 `tool_*` 家族，**没有 `direct_wipe`**，所以
  `plan/02` §7 第 8 条（跨实现可互换）在这里只能 DEFER。

复现：

```bash
PYTHONPATH=src python3 tools/s5_build_transfer.py \
    --manifest out/s5_smoke_inputs/<task>/manifest.json \
    --output out/s5_transfer_smoke/<task> --surface out/s5_smoke_inputs/<surface>.npz \
    --existing-only
PYTHONPATH=src python3 tools/s5_eval_envelope.py --artifact <artifact>.npz \
    --manifest out/s5_smoke_inputs/<task>/manifest.json --surface <surface>.npz \
    --existing-only --out <task>/eval.txt
```

每份 artifact 的完整元数据与 hash 在各子目录的 `*.report.json` 与 `index.json`。
