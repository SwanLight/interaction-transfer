# S3 · 双板 source 采集（抽屉）

> ⚠️ **数据集路径已更新**（2026-08-30，S4 收尾）：现行的三份是 `/tmp/s3_drawer_v3` / `/tmp/s3_wipe_v5` / `/tmp/s3_knob_v4`。下文里旧的 `_full` / 无后缀路径是上一轮的版本，**缺 `rel_vel_obj` 与几何/跨实现划分**，不要照着用；擦拭的 `_v5` 比 `_v4` 只多了平面的世界位姿（`source/board_pos_w` / `_quat_w`，P-54 的解法），**两版的验收报告逐字相同**，本文件的数照旧有效。


模拟"人"用两块 35×25×3 mm 的功能面把抽屉拉开，采下**物体侧**的交互数据。
这一步的产物不是"抽屉被打开了"，而是**打开它的过程中哪里在接触、接触是什么
状态、传了多大的力**——`plan/02` §3 的 Oracle Interaction Record 就从这些构造。

数据集本身（800 个 npz + manifest，36 MB）按 D-23 不进版本控制，
留在服务器 `/tmp/s3_drawer_v3`。这里只放给人看的摘要。

## 先看这个

| 文件 | 是什么 |
|---|---|
| `videos/*.mp4` | 五个策略家族各一条录像 |
| `pd_*.png` | 同一条录像的六格接触表：接近 → 上升 → 贴合 → 拉动 → 拉开 → 撤出 |
| `report.txt` | 800 条的验收表，含**接触部位分布** |
| `verify.txt` | 独立验收：划分、留出集、SHA-256、字段隔离，18 项 |
| `pretrain_asset_check.txt` | 预训练物体集六个资产的自检，12 项 |

## 怎么看录像

板上有两个**纯视觉**标记（不参与碰撞，物理与 S1 验过的一字未改）：

- **浅色贴片** = 工作面。看得到贴片，说明你正对着接触面。
- **深色顶边鳍** = 这条边朝上。

薄板前后对称、四边接近对称，没有标记就分不清哪一面在接触、哪条边朝上，
`plan/06` §7 那条"人工看视频确认接触发生在该发生的面上"也就无从执行。

两块板的机身颜色不同（橙 = plate0，蓝 = plate1），用来区分彼此。

## 五个家族在做什么

| 家族 | 做法 | 为什么要有它 |
|---|---|---|
| `pinch_center` | 手指伸进净空压杆背，拇指按杆前，中央对捏 | 最接近人开抽屉的做法 |
| `pinch_offset` | 同上，整体沿杆平移 ±14~26 mm | 同一功能、不同接触位置 |
| `hook_both` | 两块板都在杆背，纯勾不夹 | 接触**拓扑**与对捏不同 |
| `asym_primary` | 站位同上，但一块出 3.4~4.6 N、另一块只 0.5~1.2 N | 同一拓扑、力分布明显不同 |
| `single_finger` | **只用一块板**，另一块全程不参与 | `plan/02` §7 第 3 条那条泄漏检查（"改变 source 板数量后表示维度不变"）需要真的存在单接触体的示教才能验 |

## 验收结论

800 条里 739 条成功（92.4%）。接触落在：

```
把手横杆背面   90.33%      面接触度（|n·z|）   ≈0.99
把手横杆正面    9.63%      板的侧边             0.00%
横杆上/下表面   0.00%
支撑柱/面板/其他 0.04%
```

**"板的工作面 vs 背面"这个数不再作为判据**，理由见 `pitfalls.md` P-37：
它靠接触点在板局部系的 z 的正负判，而板半厚只有 1.5 mm，轻接触时那个量在
零附近抖；而且脚本化的板从不掉头，"用背面接触"在几何上根本到不了。
真正要防的是**拿边角在蹭**（D-34 那类问题），那由接触法向与板面法向的夹角判，
稳得多——实测 0.98，侧边接触 0.00%。

另有 **100% 的接触落在两根支撑柱之间的横杆自由段**——
不是 D-34 里钩杆那种"挂在柱外伸出的边角上"省事。

峰值单点接触力 11.55 N（安全上限 25 N），脏帧 0.00%，PD 饱和 0.00%。

接触模式（`plan/02` §3.4）：sticking 35.5% / sliding 35.2% / separating 29.3%。
separating 偏高是**接触报告断续**造成的（P-31），不是接触真的断了——
几何上间隙全程稳定在 −0.5 mm。这一档怎么处理留给 S4 决定。

逐帧核对过：板与横杆的间隙全程稳定在 −0.5 mm，抽屉被匀速拉开，
**不是 P-28 那种"捅一下让物体自己滑"**。

## 重新生成

```bash
./tools/sync.sh
./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \
    tools/s3_source_drawer.py --envs 80 --batches 10 --out /tmp/s3_drawer_v3" s3full

# 独立验收
ssh root@10.0.6.98 'cd /workspace/interaction_transfer && PYTHONPATH=src \
    /isaac-sim/python.sh tools/s3_verify_dataset.py /tmp/s3_drawer_v3 --sample 60'

# 录像（每个家族一条，5 张卡并行）
for i in 0 1 2 3 4; do
  fam=$(echo "pinch_center pinch_offset single_finger hook_both asym_primary" | cut -d' ' -f$((i+1)))
  IT_GPU=$i ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \
      tools/s3_source_drawer.py --video --envs 4 --batches 1 --family $fam \
      --out /tmp/s3vid_$fam" "vid_$fam" &
done; wait
```

## 还没做的

- **预训练物体集的采集脚本**（六个物体，D-39 已冻结覆盖设计，资产已自检通过）
- 擦拭 source（主任务）、旋钮 source
- 多样性验收：策略分类器在原始 source 动作上的准确率
