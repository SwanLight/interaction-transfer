# S3 · 探针物体集（交互原语库）

E-I 执行器要学的是"**我这个形态能实现哪些交互**"，不是"怎么做这三个任务"。
这一组物体只用于产生交互指令的多样性，**永不作为任务评估**——它不进 E-T、
不进 Shared Structure Model、不进任何评估。

## 这组物体是怎么选出来的（这一段是重点）

**先说一个被否掉的做法。** 最初是照着留出任务反推的：擦拭需要"带法向力持续
滑移"就加一块可以在上面滑的板，旋钮需要"绕固定轴转动"就加一个转盘。
这个做法从外面**无法与作弊区分**——如果物体是照着留出任务反推的，
那么执行器在"摸索能力"阶段学到的就是留出任务本身，只换了个物体标签。

**改法不是把边界讲清楚，而是换判据**：物体集必须从一套**独立于本项目三个
任务**的交互分类学推出来，判据是**是否张满该分类学**，不是"是否覆盖留出任务"。
留出任务被覆盖到，是"张满"的**推论**，不是设计目标。

分类学的三条外部依据（详见 `plan/03` §2.4.2）：

- [Huang, *Robotic Manipulation Primitives*](https://publications.ri.cmu.edu/storage/publications/2021/05/main.pdf)（CMU 2021）：用接触模式（separating / sticking / sliding）枚举操作原语，与 `plan/02` §3.4 记录的字段逐字对应
- [Bullock, Ma & Dollar 2013](https://www.eng.yale.edu/grablab/pubs/bullock_icorr2011.pdf)（IEEE T-Haptics）：五个二值描述子 → 15 个互斥类
- [Lynch & Mason 谱系](https://www.ri.cmu.edu/pub_files/pub2/lynch_kevin_1996_2/lynch_kevin_1996_2.pdf)（stable pushing 1996 / toppling 1999 / Aiyama pivoting 1993）：非抓握原语的既有命名

由此得到五个轴（接触结构 / 接触模式 / 力学通道 / 物体自由度 / 接触面几何）
和 **15 条原语**，每个轴的每个取值都被占据。

**另加一条硬规则**：每条原语必须由**至少两个几何结构不同的物体**提供。
满足它之后，"你是照着任务 X 设计物体 Y 的"就失去力量——删掉 Y，那一格照样在。

## 十个物体

| 物体 | 几何 | 关节 |
|---|---|---|
| `block` 方块 | 60×45×40 自由长方体 | 无 |
| `column` 立柱 | Ø56×80 自由圆柱 | 无 |
| `roller` 卧柱 | Ø60×140 圆柱横躺 | 无 |
| `ball` 球 | 自由球体 | 无 |
| `slider` 滑轨块 | 块带竖立挡片，行程 150 | prismatic |
| `plunger` 柱塞 | 圆柱在 U 形槽内滑动，端帽带台肩 | prismatic |
| `dial` 转盘 | Ø90 圆盘 + 3 凸耳 | revolute |
| `flap` 立板门 | 沿竖直边铰接的板 | revolute |
| `ridge` 棱台 | 固定台面 + Ø20 横棱 | 固定 |
| `slab` 斜板 | 200×150 固定板，可倾斜 | 固定 |

## 采集器怎么组织的

**按原语 × 接触位点，不按"每个物体一段脚本"。**

- **物体**只声明它表面有哪些位点：物体局部系的位置、外法向、切向、进入方向；
- **原语**只声明贴上去之后怎么动：法向力多大、切向是扫掠/加力/脉冲、
  一块板还是两块对置、扫掠锚在物体系还是世界系、法向走力控还是速度控制。

两者叉乘，加一个物体不需要写新的阶段机。这也不只是省代码——`plan/02` 的交互
规格本身就是「region（位点）+ engage 方向（法向）+ mode + mechanics」，
脚本按同一结构组织，采出来的数据与规格字段天然对齐。

## 验收结论

见 `coverage.txt`（逐原语的张满与冗余核对）和十个物体各自的 `report_*.txt`。
录像九条在 `videos/`（含新补上的 `ball_roll` / `ball_twist` /
`slider_pinch_move`），对应的六格接触表 `ps_*.png`。

**15 条原语全部满足两条判据**（成功 ≥200 条、≥2 个几何不同的承载物体），
实测 10740 条 episode / 9250 成功（86.1%）。

上一轮是 10/15，五个未满足的格是这样补上的（D-47）：

| 格 | 原来 | 现在 | 补的是什么 |
|---|---|---|---|
| P7 `roll` | 0 / 0 物体 | 588 / 2 | **把球真正采进来**（此前物体集只有九个）：球 210/210、卧柱 189/810 |
| P8 `twist` | 191 / 1 | 263 / 2 | **球**：两侧对置、切向反向搓，绕竖直轴转（71/210） |
| P3 `slide_push` | 204 / 1 | 251 / 2 | **立柱**：与 push 共用低位点，推到打滑（45/205） |
| P13 `pinch_hold` | 204 / 1 | 414 / 2 | **滑轨块挡片的两个窄侧面**（210/210） |
| P14 `pinch_move` | 204 / 1 | 414 / 2 | 同上，捏住再沿导轨推（210/210） |

**试过而不成立的也记在这里**：自由卧柱的**两个端面对捏**不可行——
板还没夹上来圆柱就先滚跑了（1020 条里 pinch_hold 只成功 2 条、
pinch_move 与 twist 各 0 条，物体位移 215~1399 mm）。位点留在采集器的表里
并注明原因，是为了让下一个人不必重试一遍。

**没有动分类学。** 15 条原语、五个轴的取值一个没改——补的是承载物体，
不是把做不到的格删掉（D-41）。

## 重新生成

```bash
./tools/sync.sh
# 每个物体一张卡；batches 按"每条原语 ≥200 条成功轨迹"算
#   block 34 / column 24 / roller 27 / ball 14 / slider 21 / plunger 14
#   dial 10 / flap 7 / ridge 14 / slab 14
PYTHONPATH=src /isaac-sim/python.sh tools/s3_source_probe.py \
    --object block --envs 60 --batches 34 --out /tmp/s3_probe/block

# 独立验收（探针集没有策略留出，门槛按逐原语给）
PYTHONPATH=src /isaac-sim/python.sh tools/s3_verify_dataset.py \
    /tmp/s3_probe/block --min-total 0 --min-per-family 200 --min-families 1

# 录像
PYTHONPATH=src /isaac-sim/python.sh tools/s3_source_probe.py \
    --object flap --primitive crank --video --envs 4 --batches 1 --out /tmp/vid_flap
```

数据集在服务器 `/tmp/s3_probe/`，按 D-23 不进版本控制。
