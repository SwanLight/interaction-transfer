"""Oracle Interaction Record：把一条 S3 示教转成物体中心的交互记录。

字段按 `plan/02` §3 定义：effect / region / engage 方向 / contact mode /
mechanics / phase。它是 S5（Shared Functional Envelope）、S6（E-I 执行器）、
S7（留出任务零样本）共同的输入，**这一层错了下游全错，而且成功率看不出来**
（D-34 的教训就是"接触落在完全出乎意料的地方，而数字正常"）。

三件这一层必须做对的事：

1. **合并所有采集体**。抽屉/旋钮是两块板，擦拭是两块板加一块黑板擦。
   合并之后记录里不再有"第几块板"这个概念——`plan/02` §7 第 3 条
   （改变 source 板数量后表示维度不变）与第 8 条（擦拭的 envelope 与是否
   使用工具无关）都落在这一步上。

2. **法向的正负不能信 PhysX**（P-37）。接触法向指向哪一侧取决于这一对刚体在
   PhysX 内部的先后次序，换个场景就翻。这里改用**表面采样点的外法向**定向：
   接触点归到最近的表面采样点，那个点的外法向是纯几何量，没有约定问题。
   摩擦力的符号与法向同属一个 bundle，因此**整条 episode 判一次**符号
   （P-49：离散选择必须在回路外面算一次），再统一成"作用在物体上的力"。

3. **contact mode 重判，且与原始值并排存**（D-49）。S3 的 mode 只看摩擦锥比值，
   且 ``separation > 0`` 会覆盖一切——于是抽屉 29.3% 的接触被标成 separating，
   而几何上接触是连续的（板与杆的间隙全程 −0.5 mm），那是 PhysX 逐子步报告的
   采样伪影（P-31）。这里用**法向力**判接触有没有、用**接触点处的相对切向
   速度**判 stick/slide。

   那个相对速度**由两个刚体的位姿差分算**，不用 PhysX 报的瞬时速度——后者
   被采集板的姿态极限环污染（P-52，角速度顶在 100 rad/s 上限）。位姿是位置量，
   在 50 Hz 上差分等于对 6 个物理子步取平均，抖动自然抵消。
   **这同时是真实装置能测到的量**：`plan/07` §3 的传感方案给的就是"面相对
   物体的位姿"，差分即得相对速度——oracle 与硬件在这一项上算的是同一个东西。

**这一层读什么、不读什么，要分清两类东西**（`records.is_measurement`）：

| 类别 | 例子 | 这一层 |
|---|---|---|
| 采集侧**动作与身份** | `cmd_delta` / `target_pose` / `root_velocity` / 板编号 | **绝不读** |
| 接触面与物体的**测量位姿** | `*/root_pose`、`*_pos_w`、`*_quat_w` | **读，而且必须读** |

第二类是**真实装置也测得到的量**（`plan/07` §3 给的就是"面相对物体的位姿"），
用它算接触点的相对滑移是在算**交互本身的物理性质**，不是在抄"谁在怎么操作"。
初版为了守"什么 source 都不读"而改用接触斑块位移，结果把抽屉 92.5% 的
sticking 判成一半在滑（D-49 修订）——**规矩要防的是动作泄漏，不是禁用传感**。

名字暂时还在 `source/` 底下是历史原因，概念上属于 `measurement/`（D-52）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from it import geom_cfg as G
from it.records import IR_SCHEMA_VERSION, SCHEMA_VERSION, EpisodeRecord
from it.surfaces import LEVELS, Surface, assign_to_surface, surface_for

#: S4 记录的 schema 版本，跟着 `it.records` 走（v1 已作废，见那里的说明）。
RECORD_SCHEMA = IR_SCHEMA_VERSION

#: 提取器**声明会读**的采集侧字段（按完整尾名匹配）。除此之外的 `source/*`
#: ——指令、目标位姿、本体速度、板面编号、PhysX 报的世界系接触点——一律
#: **不得影响输出的任何一位**。
#:
#: 这不是一句注释里的承诺：泄漏检查 2b 会把这张表以外的 `source/*` 全部删掉
#: 重跑一遍提取，逐字段 `array_equal` 对比，**零容差**。反过来 2c 把这张表里的
#: 删掉，要求提取器**直接报错**——宁可炸，也不在"物体大概在原点"这类假定
#: 之上算出一份看起来正常的记录（P-54）。
SOURCE_READS = ("/root_pose", "/tool_pose",
                "/object_pos_w", "/object_quat_w",
                "/drawer_pos_w", "/drawer_quat_w",
                "/disc_pos_w", "/disc_quat_w",
                "/board_pos_w", "/board_quat_w")


def reads_source(key: str) -> bool:
    """这个 `source/*` 字段在提取器的声明读取集里吗（见 `SOURCE_READS`）。"""
    return key.startswith("source/") and key.endswith(SOURCE_READS)

#: 未来窗口：1 s、10 个采样点（`plan/02` §3，50 Hz 记录 -> 每 5 帧一个）。
FUTURE_HORIZON_S = 1.0
FUTURE_SAMPLES = 10

#: 判"有没有接触"的法向力下限（N）。与 S3 诊断里 `in_contact` 的阈值一致。
CONTACT_FORCE_MIN = 0.05

#: 判 stick/slide 的滑移速度阈值（m/s）。5 mm/s = 每帧 0.1 mm。
#: **这个阈值是敏感的**：抽屉上 sticking 占比从 1 mm/s 的 21% 到 10 mm/s 的 54%。
#: 所以记录里存的是**连续量** `mode/slip_speed`，标签只是按这个阈值切出来的
#: 便利品；`s4_extract` 的报告里附 {1,3,5,10} mm/s 的敏感度表（工作方式第 4 条）。
SLIP_SPEED_MIN = 5.0e-3

#: 判"PhysX 报的瞬时相对速度可不可信"：把它沿时间积出来的路程，与接触斑块在
#: 物体表面上实际走过的路程比。前者远大于后者 = 速度信号与几何不相容（P-52）。
#: 这个量现在只用于**诊断报告**，不再决定 mode——mode 由位姿差分给出。
RELVEL_PATH_RATIO = 3.0
RELVEL_PATH_MARGIN = 5.0e-3

#: 摩擦锥比值到这个数就算"力已经饱和"。与 `contact_utils` 的 `stick_ratio` 一致，
#: 只是这里它不再决定 mode，只作为诊断量并排记录。
CONE_SATURATED = 0.95

#: 接触点离最近表面采样点多远还算"落在这个物体上"（m）。
#: 取 6 mm：求解器允许少量穿透，接触点又报在两个碰撞面之间，
#: 与 `contact_attrib.classify_drawer_local` 的容差同量级。
SURFACE_TOL = 6.0e-3

#: 单帧总接触力超过这个数就判脏帧（N）。P-27：`max_depenetration_velocity`
#: 会制造几千牛的尖峰，均值却只有十几牛；`plan/02` §8 要求异常帧在进入训练前剔除。
MAX_FRAME_FORCE = 200.0

#: 采集板的摩擦系数。规则 9 规定摩擦按 **min** 组合，所以逐点的有效 μ
#: 取 min(板, 该部件)。S3 的 mode 用的是一个标量 μ（旋钮上一律用销钉的 0.8），
#: 轮缘接触因此被按高摩擦判——这里按部件取，轮缘用它自己的 0.10。
PLATE_FRICTION = G.PlateCfg().friction


def _part_friction(obj: str) -> dict[str, float]:
    """逐部件的摩擦系数。没列到的部件用物体的默认值。"""
    if obj == "knob":
        k = G.KnobCfg()
        return {"pin_side": k.pin_friction, "pin_top": k.pin_friction,
                "rim_side": k.rim_friction, "disc_top": k.rim_friction,
                "disc_bottom": k.rim_friction}
    if obj == "drawer":
        return {}          # 抽屉整体一个材质
    return {}


def _default_friction(obj: str) -> float:
    table = {
        "drawer": G.CabinetCfg().friction,
        "knob": G.KnobCfg().rim_friction,
        "board": 0.35,                       # `s3_source_wipe.BOARD_MU`
        "block": G.BlockCfg().friction,
        "column": G.ColumnCfg().friction,
        "roller": G.RollerCfg().friction,
        "ball": G.BallCfg().friction,
        "slider": G.SliderCfg().friction,
        "plunger": G.PlungerCfg().friction,
        "dial": G.DialCfg().friction,
        "flap": G.FlapCfg().friction,
        "ridge": G.RidgeCfg().friction,
        "slab": G.SlabCfg().friction,
    }
    return table[obj]


# ---------------------------------------------------------------- 任务规格


@dataclass(frozen=True)
class TaskSpec:
    """一个被操作物体的 effect 与 task-relevant 广义力怎么算。"""

    obj: str                       # `it.surfaces` 里的物体名
    effect_names: tuple[str, ...]
    generalized_names: tuple[str, ...]
    #: (arrays, meta) -> (T, E) 的 effect 状态量
    effect_fn: Callable[[dict, dict], np.ndarray]
    #: (力 (T,A,3), 力臂 (T,A,3), 有效位 (T,A), meta) -> (T, Gdim)
    generalized_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]
    #: effect 增量是否要转进物体当前系（自由体的位姿增量要，关节量不需要）
    effect_is_pose: bool = False


def _joint_effect(key: str) -> Callable[[dict, dict], np.ndarray]:
    def fn(arrays: dict, meta: dict) -> np.ndarray:
        return np.asarray(arrays[key], dtype=np.float64).reshape(len(arrays[key]), -1)
    return fn


def _probe_effect(arrays: dict, meta: dict) -> np.ndarray:
    """探针物体：有关节就用关节量，自由体用 (位置, 四元数)。"""
    return np.asarray(arrays["object/state"], dtype=np.float64)


def _wipe_effect(arrays: dict, meta: dict) -> np.ndarray:
    """擦拭的 effect **只有平面自身的表面状态变化**（D-42）。

    黑板擦或执行器接触面的位姿都是 source 侧的东西，写进 effect 就等于
    让 envelope 携带"用什么姿态、沿什么路径移动你的手"——那是动作层面的迁移，
    正是本工作声称不做的事（`plan/00` §1）。
    """
    return np.asarray(arrays["object/dirt_cleared"], dtype=np.float64).reshape(-1, 1)


def _gen_along(axis: np.ndarray) -> Callable:
    """沿固定方向的合力（棱柱关节：抽屉、滑轨块、柱塞）。"""
    def fn(force, lever, valid, meta):
        f = (force * valid[..., None]).sum(axis=1)
        return (f @ axis).reshape(-1, 1)
    return fn


def _gen_torque(axis: np.ndarray, anchor: np.ndarray) -> Callable:
    """绕固定轴的力矩（转动关节：旋钮、转盘、立板门）。"""
    def fn(force, lever, valid, meta):
        r = lever - anchor[None, None, :]
        tau = np.cross(r, force) * valid[..., None]
        return (tau.sum(axis=1) @ axis).reshape(-1, 1)
    return fn


def _gen_wrench(force, lever, valid, meta):
    """自由体：能动的方向就是全部六个，广义力就是 6D wrench 本身。"""
    f = (force * valid[..., None]).sum(axis=1)
    tau = (np.cross(lever, force) * valid[..., None]).sum(axis=1)
    return np.concatenate([f, tau], axis=1)


def _gen_press_tan_plane(force, lever, valid, meta):
    """平面（擦拭）：压进平面的法向力 + 面内的切向合力。

    平面的工作面是局部 +Z（`assets.board_cfg`），所以"压"是 −Z 方向的合力。
    """
    f = (force * valid[..., None]).sum(axis=1)
    return np.stack([-f[:, 2], f[:, 0], f[:, 1]], axis=1)


def _gen_press_tan(force, lever, valid, meta):
    """固定物体（棱台、斜板）：法向压力 + 切向作用的幅值。

    物体不动（`plan/03` §2.4.3 的 E5），effect 不在位姿上，所以 mechanics
    只能由力本身表达。法向按逐点的接触法向取，不用世界轴。
    """
    normal = meta["_normal_in"]                     # (T, A, 3)，指向物体内部
    fn_ = (np.einsum("tak,tak->ta", force, normal) * valid).sum(axis=1)
    ft = force - normal * np.einsum("tak,tak->ta", force, normal)[..., None]
    tan = np.linalg.norm((ft * valid[..., None]).sum(axis=1), axis=1)
    return np.stack([fn_, tan], axis=1)


_X, _Y, _Z = np.eye(3)


def _task_specs() -> dict[str, TaskSpec]:
    dial, flap = G.DialCfg(), G.FlapCfg()
    hinge = np.array([0.0, -flap.panel[1] / 2 - flap.post_radius, 0.0])
    free = ("f_x", "f_y", "f_z", "tau_x", "tau_y", "tau_z")
    pose = ("dx", "dy", "dz", "rx", "ry", "rz")
    specs = {
        "drawer": TaskSpec("drawer", ("joint_pos",), ("f_rail",),
                           _joint_effect("object/drawer_joint_pos"),
                           _gen_along(_X)),
        "knob": TaskSpec("knob", ("joint_angle",), ("tau_axis",),
                         _joint_effect("object/disc_angle"),
                         _gen_torque(_Z, np.zeros(3))),
        "wipe": TaskSpec("board", ("dirt_cleared",),
                         ("f_press", "f_tan_x", "f_tan_y"),
                         _wipe_effect, _gen_press_tan_plane),
    }
    # 探针物体：有关节的按关节轴，自由体按 6D，固定的按压/切
    for name, gen_names, gen_fn in (
        ("slider", ("f_rail",), _gen_along(_X)),
        ("plunger", ("f_rail",), _gen_along(_X)),
        ("dial", ("tau_axis",), _gen_torque(_Z, np.zeros(3))),
        ("flap", ("tau_axis",), _gen_torque(_Z, hinge)),
    ):
        specs[f"probe_{name}"] = TaskSpec(name, ("joint_pos",), gen_names,
                                          _probe_effect, gen_fn)
    for name in ("block", "column", "roller", "ball"):
        specs[f"probe_{name}"] = TaskSpec(name, pose, free, _probe_effect,
                                          _gen_wrench, effect_is_pose=True)
    for name in ("ridge", "slab"):
        specs[f"probe_{name}"] = TaskSpec(name, ("static",), ("f_press", "f_tan"),
                                          _probe_effect, _gen_press_tan)
    _ = dial
    return specs


TASK_SPECS = _task_specs()


def spec_for(meta: dict) -> TaskSpec:
    """按 episode 的 meta 取任务规格。未知任务直接抛错，不猜。"""
    task = str(meta.get("task", ""))
    if task in TASK_SPECS:
        return TASK_SPECS[task]
    raise KeyError(f"未知任务 {task!r}；已知：{sorted(TASK_SPECS)}")


# ---------------------------------------------------------------- 工具


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3)。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.linalg.norm(q, axis=1)
    n = np.where(n > 0, n, 1.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def _rotvec(rot: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,3) 旋转矢量（轴 × 角）。"""
    trace = np.clip((np.trace(rot, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    skew = np.stack([rot[:, 2, 1] - rot[:, 1, 2],
                     rot[:, 0, 2] - rot[:, 2, 0],
                     rot[:, 1, 0] - rot[:, 0, 1]], axis=1)
    denom = 2.0 * np.sin(angle)
    small = np.abs(denom) < 1e-9
    denom = np.where(small, 1.0, denom)
    return np.where(small[:, None], skew / 2.0, skew * (angle / denom)[:, None])


def contact_bodies(arrays: dict) -> list[str]:
    """记录里有哪些接触体（``contact/<body>/pos_obj`` 的 body 部分），按名字排序。

    **合并它们是 S4 的职责**：`plan/02` §7 第 3 条要求改变 source 板数量后
    表示维度不变，第 8 条要求擦拭的表示与"有没有工具"无关。名字本身
    （plate0 / plate1 / tool）到此为止，不进 S4 记录。
    """
    return sorted({k.split("/")[1] for k in arrays if k.startswith("contact/")
                   and k.endswith("/pos_obj")})


# ---------------------------------------------------------------- 提取


def extract(record: EpisodeRecord, surface: Surface | None = None,
            *, surface_tol: float = SURFACE_TOL) -> EpisodeRecord:
    """把一条 S3 episode 转成 Oracle Interaction Record。

    Args:
        record: S3 的 `EpisodeRecord`（``s3-episode-v1``）。
        surface: 该物体该几何变体的冻结表面采样；``None`` 时按 meta 自己取。
        surface_tol: 接触点离表面采样点多远还算落在这个物体上。

    Returns:
        新的 `EpisodeRecord`，``schema_version = "s4-record-v2"``。
    """
    if record.meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"输入不是 {SCHEMA_VERSION}：{record.meta.get('schema_version')}")
    meta = record.meta
    spec = spec_for(meta)
    arrays = record.model_arrays()        # fail-closed：source/* 到不了这里
    if surface is None:
        surface = surface_for(spec.obj, str(meta.get("geometry_variant", "nominal")))

    bodies = contact_bodies(arrays)
    if not bodies:
        raise ValueError(f"{meta.get('episode_id')} 里没有任何 contact/<body>/ 数组")

    # --- 1. 合并所有接触体的接触点 ---
    def cat(field: str, dtype) -> np.ndarray:
        return np.concatenate([np.asarray(arrays[f"contact/{b}/{field}"], dtype=dtype)
                               for b in bodies], axis=1)

    pos = cat("pos_obj", np.float64)              # (T, A, 3)
    nrm_rep = cat("normal_obj", np.float64)
    fri_rep = cat("friction_obj", np.float64)
    fn = np.abs(cat("normal_force", np.float64))  # (T, A)
    sep = cat("separation", np.float64)
    valid_slot = cat("valid", np.bool_)
    mode_raw = cat("mode", np.int8)
    has_relvel = all(f"contact/{b}/rel_vel_obj" in arrays for b in bodies)
    rel_vel = cat("rel_vel_obj", np.float64) if has_relvel else np.zeros_like(pos)

    n_frames, n_slots = fn.shape
    active = valid_slot & (fn > CONTACT_FORCE_MIN)

    # --- 2. 归到表面采样点 ---
    flat_ok = active.reshape(-1)
    idx = np.full(n_frames * n_slots, -1, dtype=np.int64)
    on_surf = np.zeros(n_frames * n_slots, dtype=bool)
    surf_dist = np.zeros(n_frames * n_slots)
    if flat_ok.any():
        sel_idx, sel_ok, sel_d = assign_to_surface(pos.reshape(-1, 3)[flat_ok], surface,
                                                   max_dist=surface_tol)
        idx[flat_ok] = np.where(sel_ok, sel_idx, -1)
        on_surf[flat_ok] = sel_ok
        surf_dist[flat_ok] = sel_d
    idx = idx.reshape(n_frames, n_slots)
    on_surf = on_surf.reshape(n_frames, n_slots)
    surf_dist = surf_dist.reshape(n_frames, n_slots)

    # --- 3. 法向定向（P-37）：用表面采样点的外法向，几何量，没有约定问题 ---
    normal_out = np.zeros_like(pos)
    take = idx >= 0
    normal_out[take] = surface.normals[idx[take]].astype(np.float64)
    # 归不到表面的点（几何变体没对上、或接触落在没建模的部件上）退而用
    # PhysX 报的法向，符号由**整条 episode 判一次**的一致性给（见下）
    agree = np.einsum("tak,tak->ta", nrm_rep, normal_out)
    w = fn * take
    sign_normal = 1.0 if float((agree * w).sum()) >= 0.0 else -1.0
    fallback = active & ~take
    normal_out[fallback] = sign_normal * nrm_rep[fallback]
    nl = np.linalg.norm(normal_out, axis=-1, keepdims=True)
    normal_out = np.divide(normal_out, np.where(nl > 0, nl, 1.0))
    normal_in = -normal_out

    # PhysX 报的这一束（法向 + 摩擦）作用在**谁**身上，取决于刚体对的内部次序。
    # 报的法向若与物体外法向同向，这束力就是作用在采集体上的，取负才是作用在
    # 物体上的。整条 episode 判一次（P-49：离散选择不能逐帧算）。
    on_object = -1.0 if sign_normal > 0.0 else 1.0
    denom = float((np.abs(agree) * w).sum())
    normal_conf = float(abs((agree * w).sum()) / denom) if denom > 0 else 0.0

    # --- 4. 作用在物体上的力 ---
    f_normal = fn[..., None] * normal_in                     # 压进物体
    f_friction = on_object * fri_rep
    force = (f_normal + f_friction) * active[..., None]
    lever = pos

    # --- 5. contact mode 重判（D-49） ---
    # 三条证据，主判据是第一条：
    #   一（位姿差分）：把接触点分别钉在两个刚体上，按它们的**位姿**各自搬到
    #     下一帧，位置之差 / dt 就是接触点处的相对速度。位姿是位置量，
    #     差分等于对物理子步取平均，不受采集体的姿态极限环影响（P-52）。
    #   二（几何）：接触斑块在物体表面上的移动速度。⚠️ 它把"平板贴圆柱时接触线
    #     随姿态微动而迁移"也算成滑移（实测抽屉上因此把 93% 的 sticking 判成
    #     一半在滑），所以只作诊断，不当主判据。
    #   三（力）：摩擦锥比值，饱和即在滑；逐点摩擦是斑块级摊派来的，噪声大。
    v_n = np.einsum("tak,tak->ta", rel_vel, normal_out)
    v_t = rel_vel - v_n[..., None] * normal_out
    inst_slip = np.linalg.norm(v_t, axis=-1)
    dt = 1.0 / float(meta.get("control_hz", 50.0))
    patch_slip, trusted_body, trusted_slot, body_has_data = _patch_slip(
        pos, fn, active, len(bodies), inst_slip, dt)
    live_trust = trusted_body[body_has_data]
    pose_slip = _pose_slip(record.arrays, bodies, pos, normal_out, meta, dt)
    if pose_slip is None:
        # 拿不到物体的**世界系**位姿，就算不出正确的相对速度。**宁可炸**——
        # 退回斑块位移会悄悄产出一份看起来正常、mode 却是错的记录（P-54）。
        # 采集器必须把被操作物体的 root pose 记进 episode（一列常量也行）。
        raise ValueError(
            f"{meta.get('episode_id')}：拿不到被操作物体的世界位姿"
            "（source/<物体>_pos_w 与 _quat_w）。没有它就算不出接触点的相对速度，"
            "mode 字段无从谈起——见 P-54，采集器要把物体的 root pose 记下来")
    slip = pose_slip
    slip_source = "pose_diff"
    mu = np.full(n_slots, _default_friction(spec.obj))
    part_mu = _part_friction(spec.obj)
    mu_point = np.full((n_frames, n_slots), float(mu[0]))
    if part_mu:
        per_part = np.array([part_mu.get(name, _default_friction(spec.obj))
                             for name in surface.parts])
        mu_point[take] = per_part[surface.part[idx[take]]]
    mu_eff = np.minimum(mu_point, PLATE_FRICTION)            # 规则 9：min 组合
    cone = np.linalg.norm(fri_rep, axis=-1) / np.maximum(mu_eff * fn, 1e-9)

    mode = np.zeros((n_frames, n_slots), dtype=np.int8)      # 0 = no_contact
    mode[active & (slip <= SLIP_SPEED_MIN)] = 1              # sticking
    mode[active & (slip > SLIP_SPEED_MIN)] = 2               # sliding
    # separating 只留给"力已经没了、几何上正在分开"的槽位。S3 版本里
    # `separation > 0` 会覆盖掉带力的接触，那是 PhysX 逐子步报告的伪影（P-31）。
    mode[valid_slot & ~active & (sep > 0)] = 3

    # --- 6. mechanics ---
    wrench = np.concatenate([
        (force * active[..., None]).sum(axis=1),
        (np.cross(lever, force) * active[..., None]).sum(axis=1),
    ], axis=1)
    gen_meta = dict(meta)
    gen_meta["_normal_in"] = normal_in
    generalized = spec.generalized_fn(force, lever, active.astype(np.float64), gen_meta)

    # --- 7. effect 与未来窗口 ---
    eff = spec.effect_fn(arrays, meta)
    future, future_valid, eff_names = _future_window(eff, spec, meta)
    # 任务无关接口（D-53）：E-I 吃的是这两路，不是上面那两路任务原生量
    rigid, field = _effect_interface(eff, spec, meta, surface, arrays)

    # --- 8. 脏帧 ---
    frame_force = fn.sum(axis=1)
    finite = np.isfinite(pos).all(axis=(1, 2)) & np.isfinite(force).all(axis=(1, 2))
    valid_s3 = np.asarray(arrays.get("valid_frame", np.ones(n_frames, bool)), dtype=bool)
    valid_s4 = valid_s3 & finite & (frame_force <= MAX_FRAME_FORCE)

    order = np.argsort(-(fn * active), axis=1)               # 力大的排前面
    take_rows = np.arange(n_frames)[:, None]

    def sort_by_force(a: np.ndarray) -> np.ndarray:
        return a[take_rows, order] if a.ndim == 2 else a[take_rows, order, :]

    out_arrays: dict[str, np.ndarray] = {
        "phase": np.asarray(arrays["phase"], dtype=np.int8),
        "progress": np.asarray(arrays["progress"], dtype=np.float32),
        "valid_frame": valid_s3,
        "valid_s4": valid_s4,
        "region/pos_obj": sort_by_force(pos).astype(np.float32),
        "region/point_idx": sort_by_force(idx).astype(np.int32),
        "region/weight": sort_by_force(fn * active).astype(np.float32),
        "region/valid": sort_by_force(active),
        "region/on_surface": sort_by_force(on_surf),
        "engage/dir": sort_by_force(normal_in * active[..., None]).astype(np.float32),
        "mode/label": sort_by_force(mode),
        "mode/raw": sort_by_force(mode_raw),
        "mode/slip_speed": sort_by_force(slip * active).astype(np.float32),
        "mode/cone_ratio": sort_by_force(np.where(active, cone, 0.0)).astype(np.float32),
        "mode/inst_slip": sort_by_force(inst_slip * active).astype(np.float32),
        "mode/patch_slip": sort_by_force(patch_slip * active).astype(np.float32),
        "mode/pose_slip": sort_by_force(
            (pose_slip if pose_slip is not None else np.zeros_like(fn)) * active
        ).astype(np.float32),
        "mech/force_obj": sort_by_force(force).astype(np.float32),
        "mech/wrench_obj": wrench.astype(np.float32),
        "mech/generalized": np.asarray(generalized, dtype=np.float32),
        # --- 任务无关接口：E-I 的 effect 通道就是这两路（D-53）---
        "effect/rigid": rigid.astype(np.float32),
        "effect/surface_state": field.astype(np.float32),
        # --- 任务原生量：给 S5 构造 oracle envelope 与诊断用，维度按任务变 ---
        "effect/current": eff.astype(np.float32),
        "effect/future": future.astype(np.float32),
        "effect/future_valid": future_valid,
        "aux/n_contacts": active.sum(axis=1).astype(np.int16),
        "aux/off_surface": (active & ~on_surf).sum(axis=1).astype(np.int16),
        "aux/frame_force": frame_force.astype(np.float32),
        # 接触点离表面采样点多远（m）。穿透期它是穿透深度，正常接触期是
        # 采样 pitch 的量级；分布异常就说明归属出了问题。
        "aux/surface_dist": (surf_dist * active).max(axis=1).astype(np.float32),
    }

    new_meta: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "episode_id": meta["episode_id"],
        "task": meta["task"],
        "object": spec.obj,
        "strategy_family": meta.get("strategy_family", "unknown"),
        "strategy_variant": meta.get("strategy_variant", "default"),
        "implementation": meta.get("implementation", "default"),
        "physics_variant": meta.get("physics_variant", "nominal"),
        "geometry_variant": meta.get("geometry_variant", "nominal"),
        "success": bool(meta.get("success", False)),
        "split": meta.get("split", "unassigned"),
        "control_hz": meta.get("control_hz", 50.0),
        "surface": {
            "object": surface.obj, "geom_tag": surface.geom_tag,
            "n_points": surface.n_points, "sha256": surface.sha256,
            "parts": list(surface.parts), "total_area_m2": surface.total_area,
        },
        "fields": {
            "effect_names": list(eff_names),
            "generalized_names": list(spec.generalized_names),
            "mode_names": ["no_contact", "sticking", "sliding", "separating"],
            # E-I 只许吃这两路 effect；`effect/current` 与 `effect/future` 是
            # **任务原生量**（维度按任务变），只给 S5 与诊断用（D-53）
            "effect_interface": ["effect/rigid", "effect/surface_state"],
            "effect_field_level": EFFECT_FIELD_LEVEL,
            "n_slots": int(n_slots),
            "n_bodies": len(bodies),
        },
        "extraction": {
            "surface_tol_m": surface_tol,
            "contact_force_min_N": CONTACT_FORCE_MIN,
            "slip_speed_min_mps": SLIP_SPEED_MIN,
            "max_frame_force_N": MAX_FRAME_FORCE,
            "future_horizon_s": FUTURE_HORIZON_S,
            "future_samples": FUTURE_SAMPLES,
            # 这一束力作用在谁身上（+1 = 报的就是作用在物体上的）。
            # `normal_sign_agreement` 是判这个符号时的加权一致度，
            # 明显小于 1 说明**同一条 episode 里约定翻过**，那种数据不能用。
            "force_on_object_sign": on_object,
            "normal_sign_agreement": normal_conf,
            "has_rel_vel": bool(has_relvel),
            # 被操作物体的位姿是量出来的，还是按"它不动"假定的。擦拭平面没记位姿，
            # 于是那份数据上的场景旋转检查无法如实进行（见 `_object_pose`）。
            "object_pose_measured": bool(_pose_measured(record.arrays, meta)),
            # 逐接触体：瞬时相对速度与斑块位移的路程相容吗（P-52）。
            # 全为 False 时 mode 完全由斑块位移给出。
            # mode 用哪一路滑移速度。主判据是位姿差分（见上）。
            "mode_source": slip_source,
            # 诊断：PhysX 报的瞬时速度与几何是否相容（P-52）。全 False 说明
            # 那一路被采集体的姿态极限环污染了，只能当诊断看。
            "rel_vel_trusted": [bool(v) for v in trusted_body],
            "slip_threshold_mps": SLIP_SPEED_MIN,
        },
        "source_episode": {
            "episode_id": meta["episode_id"],
            "generator_git_sha": meta.get("generator_git_sha", "unknown"),
            "physics": meta.get("physics", {}),
        },
    }
    return EpisodeRecord(meta=new_meta, arrays=out_arrays)


#: 接触体在记录里的位姿字段名。采集器给板写的是 ``source/plateK/root_pose``，
#: 给黑板擦写的是 ``source/tool_pose``——名字不统一是历史原因，这里兜住。
def _body_pose_key(arrays: dict, body: str) -> str | None:
    for key in (f"source/{body}/root_pose", f"source/{body}_pose"):
        if key in arrays:
            return key
    return None


def _pose_measured(arrays: dict, meta: dict) -> bool:
    """被操作物体的位姿是量出来的（True）还是按"它不动"假定的（False）。"""
    op = _object_pose(arrays, meta)
    return bool(op[2]) if op is not None else False


#: 接触点到接触体原点的距离上限（m）。接触点就长在接触体表面上，几十毫米而已；
#: 出现米级只有一个可能——两个坐标系混用了（P-54 实测 8~9 m）。
#: **这条不变量是那一类 bug 的唯一硬防线**：它零征兆，逐帧数值全部正常，
#: 接触部位分布、动力学一致性、单元测试也全部正常。
MAX_LEVER_M = 0.5


def _assert_lever(world: np.ndarray, pose: np.ndarray, rot: np.ndarray,
                  key: str) -> None:
    """接触点相对接触体原点的力臂必须物理合理，否则直接炸。"""
    loc = np.einsum("tij,tkj->tki", rot[1:-1].transpose(0, 2, 1),
                    world - pose[1:-1, None, :3])
    d = np.linalg.norm(loc, axis=-1)
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size and float(np.median(finite)) > MAX_LEVER_M:
        raise ValueError(
            f"{key} 与接触点的力臂中位 {float(np.median(finite)):.3f} m，"
            f"超过 {MAX_LEVER_M} m —— 接触点长在接触体表面上，不可能这么远。"
            "几乎肯定是物体位姿与接触体位姿不在同一个坐标系（P-54）")


def _pose_slip(arrays: dict, bodies: list, pos: np.ndarray,
               normal_out: np.ndarray, meta: dict, dt: float) -> np.ndarray | None:
    """由**位姿差分**算接触点处的相对切向速度。(T, A) 或 None（缺位姿时）。

    做法：把同一个接触点分别钉在接触体和被操作物体上，按各自的位姿搬到前后帧，
    位移之差除以 2dt 就是相对速度。**位姿是位置量**，在 50 Hz 上差分等于对 6 个
    物理子步取平均——PhysX 报的瞬时速度会被采集板的姿态极限环污染到虚高
    5~10 倍（P-52），这条不会。

    **这一步读了 `source/*`，是有意的**：`plan/02` §1 禁的是把 source 字段
    **写进表示**，不是禁 oracle 用特权数据去算物体中心的物理量。算出来的
    "接触点处的相对速度"是交互本身的性质（Huang 的接触模式分类就按它定义），
    不是板的位姿。而且 `plan/07` §3 的传感方案给的正是"面相对物体的位姿"，
    真实装置差分一下得到的就是同一个量——oracle 与硬件在这一项上没有落差。

    实测（擦拭数据集，板不抖，可作标尺）：与 PhysX 瞬时速度逐帧相关 0.95、
    各阈值下 sticking 占比差 0.5 个百分点以内；而在抽屉/旋钮上它给出
    93~95% sticking，瞬时速度给 3~6%——后者是被抖动污染的那一路。
    """
    keys = [_body_pose_key(arrays, b) for b in bodies]
    if any(k is None for k in keys):
        return None
    op = _object_pose(arrays, meta)
    if op is None:
        return None
    obj_pos, obj_rot, _measured = op
    n_frames, n_slots = pos.shape[0], pos.shape[1]
    per = n_slots // max(len(bodies), 1)
    out = np.zeros((n_frames, n_slots))
    # 接触点的世界坐标（物体系 -> 世界）
    world = obj_pos[:, None, :] + np.einsum("tij,tkj->tki", obj_rot, pos)

    def carried(rot: np.ndarray, org: np.ndarray) -> np.ndarray:
        """把接触点钉在这个刚体上，返回它在前一帧/后一帧的世界位置之差。"""
        loc = np.einsum("tij,tkj->tki", rot.transpose(0, 2, 1), world - org[:, None, :])
        nxt = org[2:, None, :] + np.einsum("tij,tkj->tki", rot[2:], loc[1:-1])
        prv = org[:-2, None, :] + np.einsum("tij,tkj->tki", rot[:-2], loc[1:-1])
        return nxt - prv

    d_obj = carried(obj_rot, obj_pos)
    for b, key in enumerate(keys):
        pose = np.asarray(arrays[key], dtype=np.float64)
        rot = _quat_to_rot(pose[:, 3:7])
        _assert_lever(world[1:-1], pose, rot, key)
        rel = (carried(rot, pose[:, :3]) - d_obj) / (2.0 * dt)
        # ⚠️ `rel` 算在**世界系**，而法向是**物体系**的——不转回来就等于拿两个
        # 不同坐标系的矢量做投影，切向分量整个是错的。这个 bug 是被
        # `plan/02` §7 第 1 条的场景旋转检查逮到的（转过的场景里滑移速度差到
        # 0.17 m/s），逐帧数值上完全看不出来。
        rel = np.einsum("tij,tkj->tki", obj_rot[1:-1].transpose(0, 2, 1), rel)
        sl = slice(b * per, (b + 1) * per)
        nrm = normal_out[1:-1, sl]
        v = rel[:, sl]
        tang = v - np.einsum("tak,tak->ta", v, nrm)[..., None] * nrm
        out[1:-1, sl] = np.linalg.norm(tang, axis=-1)
    # 首末帧没有中心差分，用邻帧顶上（比留 0 诚实：留 0 会被判成 sticking）
    out[0], out[-1] = out[1], out[-2]
    return out


def _object_pose(arrays: dict, meta: dict):
    """被操作物体的**世界系**位姿，返回 ``(位置, 旋转, 是不是量出来的)``。

    ⚠️ **必须是世界系，而且必须与接触体位姿同一个系。** 这是 P-54 的教训：
    接触体位姿（``source/*/root_pose``）是**含 env 原点偏移的世界坐标**，
    而 ``object/state`` 对自由体记的是**减去 env 原点**的相对坐标
    （`s3_source_probe.py` 的 ``obj_state()``）。两者混用会算出 8~9 m 的力臂，
    姿态误差乘上去就是几十毫米的假位移——**而逐帧数值全部正常，零征兆**。

    所以这里只认 ``source/<物体>_pos_w`` / ``_quat_w`` 这一族世界系字段，
    **绝不回退到 `object/state`**；拿不到就返回 None，让 `extract` 直接报错。
    宁可炸，也不要在一个假定之上悄悄算出一份看起来正常的记录——
    擦拭平面曾经走过一条"假定它在原点"的分支，那正是 P-54 的另一半。
    """
    for pk, qk in (("source/drawer_pos_w", "source/drawer_quat_w"),
                   ("source/disc_pos_w", "source/disc_quat_w"),
                   ("source/board_pos_w", "source/board_quat_w"),
                   ("source/object_pos_w", "source/object_quat_w")):
        if pk in arrays and qk in arrays:
            return (np.asarray(arrays[pk], dtype=np.float64),
                    _quat_to_rot(np.asarray(arrays[qk], dtype=np.float64)), True)
    return None


def _patch_slip(pos: np.ndarray, fn: np.ndarray, active: np.ndarray,
                n_bodies: int, inst_slip: np.ndarray,
                dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑移速度：接触斑块在**物体表面上**移动得多快，外加"瞬时速度可不可信"。

    为什么不直接用 PhysX 报的接触点相对速度：采集板的姿态 PD 处在数值极限环上，
    角速度在抽屉 96.9%、旋钮 100% 的操作帧顶在 PhysX 的 100 rad/s 上限（P-52）。
    ω×r 让瞬时相对速度虚高到 70 mm/s，把它沿时间积出来是 59 mm 的滑移，
    而同一段里接触点在物体系里的坐标**一动没动**（0.1 mm 以内）——那 59 mm
    没有发生。斑块位移是**位置量**，不受这个抖动影响。

    这条判据在干净数据上与瞬时速度等价：擦拭数据集的板角速度是 0，两者
    逐帧相关 0.95、各阈值下的 sticking 比例差 0.2 个百分点（实测）。

    **它的已知盲区**：接触斑块停在物体的同一处、而接触体沿自身表面滑过去时
    （平板沿圆柱轴向平移），斑块位移为零而实际在滑。摩擦锥比值
    （`mode/cone_ratio`）是这一处的交叉证据，两者都存进记录。

    Args:
        pos: (T, A, 3) 物体系接触点，A = n_bodies × 每体槽位数。
        fn: (T, A) 法向力幅值；active: (T, A) 有效接触。
        inst_slip: (T, A) 瞬时切向相对速度，用来做路程相容性检验。

    Returns:
        (斑块滑移速度 (T, A), 逐接触体是否可信 (n_bodies,), 逐槽位是否可信 (A,),
        逐接触体有没有数据 (n_bodies,))
    """
    n_frames, n_slots = fn.shape
    per = n_slots // max(n_bodies, 1)
    patch = np.zeros((n_frames, n_slots))
    trusted_body = np.zeros(n_bodies, dtype=bool)
    # 全程没有接触的接触体（`single_finger` 里闲置的那块板）既不算可信也不算
    # 不可信——把它算进去会让"这条 episode 用了哪个判据"变成 mixed，读不出信息。
    has_data = np.zeros(n_bodies, dtype=bool)
    for b in range(n_bodies):
        sl = slice(b * per, (b + 1) * per)
        w = (fn[:, sl] * active[:, sl])[..., None]
        tot = w.sum(axis=1)
        live = tot[:, 0] > 0
        cen = np.full((n_frames, 3), np.nan)
        cen[live] = ((pos[:, sl] * w).sum(axis=1) / np.maximum(tot, 1e-12))[live]
        # 中心差分；两端各退一格。相邻帧都要有接触，否则那一步不是"斑块移动"，
        # 而是"接触断了又接上"，两者不能混为一谈。
        speed = np.zeros(n_frames)
        both = live[:-2] & live[2:] & live[1:-1]
        step = np.linalg.norm(cen[2:] - cen[:-2], axis=-1) / (2.0 * dt)
        speed[1:-1] = np.where(both, np.nan_to_num(step), 0.0)
        patch[:, sl] = speed[:, None]

        # 路程相容性：瞬时速度积出来的路程 vs 斑块真实走过的路程
        m = live[1:-1] & both
        path_inst = float((inst_slip[:, sl] * active[:, sl]).max(axis=1)[1:-1][m].sum() * dt)
        path_patch = float(speed[1:-1][m].sum() * dt)
        has_data[b] = bool(m.any())
        trusted_body[b] = path_inst <= RELVEL_PATH_RATIO * path_patch + RELVEL_PATH_MARGIN
    trusted_slot = np.repeat(trusted_body, per)
    if trusted_slot.shape[0] != n_slots:            # 槽位数不能整除时不猜
        trusted_slot = np.zeros(n_slots, dtype=bool)
    return patch, np.where(has_data, trusted_body, True), trusted_slot, has_data


#: 任务无关 effect 接口的横向分辨率：surface_state 用最粗那一档（64 点）。
#: 它只需要表达"哪一片表面的状态在变"，不需要 region 那种精度。
EFFECT_FIELD_LEVEL = LEVELS[0]


def _effect_interface(eff: np.ndarray, spec: TaskSpec, meta: dict, surface,
                      arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """把各任务**语义完全不同**的 effect 统一成一份定长、无分支的接口。

    问题出在这里：抽屉的 effect 是一个关节位移、旋钮是一个关节角、自由体是 6D
    位姿增量、擦拭是平面上的 dirt 场——维度和语义都不一样。E-I 若直接吃这些，
    网络里就得写 ``if task == "drawer" ...``，**"任务无关"当场作废**。

    统一办法：任何 effect 都可以写成"**被操作物体的表面接下来会怎么变**"，
    而表面的变化只有两种：

    - ``effect/rigid`` (T, H, 6)：物体整体的**刚体位移**，表达在物体**当前**系里
      （平移 3 + 旋转矢量 3）。关节量按关节轴换算过来，自由体直接就是位姿增量，
      固定物体是零。于是抽屉的"拉开 20 mm"和旋钮的"转 0.3 rad"变成了同一种东西。
    - ``effect/surface_state`` (T, H, L)：逐表面采样点的**表面状态变化**。
      目前只有擦拭用它（dirt 被擦掉多少），其余任务是零。

    两者都是定长的，且**不含任何任务标识**。E-I 的指令通道吃这两路 + region /
    engage / mode / mechanics，全程没有分支。

    ⚠️ 原来那两路 ``effect/current`` 与 ``effect/future`` 保留，但它们是
    **任务原生量**，只给 S5 构造 oracle envelope 和做诊断用，
    **不是 E-I 的接口**——这一条写在 meta 的 ``fields.effect_interface`` 里。
    """
    n_frames = eff.shape[0]
    hz = float(meta.get("control_hz", 50.0))
    stride = max(1, int(round(FUTURE_HORIZON_S * hz / FUTURE_SAMPLES)))
    offsets = np.arange(1, FUTURE_SAMPLES + 1) * stride
    rigid = np.zeros((n_frames, FUTURE_SAMPLES, 6))

    axis, anchor, kind = _effect_axis(spec)
    for j, off in enumerate(offsets):
        last = n_frames - off
        if last <= 0:
            continue
        t = np.arange(last)
        if kind == "free":
            pos, quat = eff[:, :3], eff[:, 3:7]
            rot = _quat_to_rot(quat)
            dp = np.einsum("tij,tj->ti", rot[t].transpose(0, 2, 1), pos[t + off] - pos[t])
            dr = _rotvec(np.einsum("tij,tjk->tik", rot[t].transpose(0, 2, 1), rot[t + off]))
            rigid[t, j] = np.concatenate([dp, dr], axis=1)
        elif kind == "prismatic":
            dq = (eff[t + off, 0] - eff[t, 0])[:, None]
            rigid[t, j, :3] = dq * axis[None, :]
        elif kind == "revolute":
            dth = (eff[t + off, 0] - eff[t, 0])[:, None]
            rigid[t, j, 3:] = dth * axis[None, :]
            # 转轴不过物体原点时（立板门的铰链在边上），原点本身也会被搬走
            if np.linalg.norm(anchor) > 0:
                ang = dth[:, 0]
                c, si = np.cos(ang), np.sin(ang)
                r = -anchor
                perp = r - (r @ axis) * axis
                cross = np.cross(axis, perp)
                rigid[t, j, :3] = ((c - 1.0)[:, None] * perp[None, :]
                                   + si[:, None] * cross[None, :])
        # kind == "static"：物体不动，rigid 全零

    field = np.zeros((n_frames, FUTURE_SAMPLES, EFFECT_FIELD_LEVEL), dtype=np.float32)
    if spec.obj == "board" and "object/dirt_grid" in arrays:
        field = _dirt_field(arrays, surface, offsets, meta)
    return rigid, field


def _effect_axis(spec: TaskSpec):
    """物体能动的方向：(轴, 转轴锚点, 类型)。类型 ∈ free/prismatic/revolute/static。"""
    if spec.effect_is_pose:
        return _Z, np.zeros(3), "free"
    table = {
        "drawer": (_X, np.zeros(3), "prismatic"),
        "slider": (_X, np.zeros(3), "prismatic"),
        "plunger": (_X, np.zeros(3), "prismatic"),
        "knob": (_Z, np.zeros(3), "revolute"),
        "dial": (_Z, np.zeros(3), "revolute"),
        "flap": (_Z, np.array([0.0, -G.FlapCfg().panel[1] / 2 - G.FlapCfg().post_radius,
                               0.0]), "revolute"),
    }
    return table.get(spec.obj, (_Z, np.zeros(3), "static"))


def _dirt_field(arrays: dict, surface, offsets: np.ndarray,
                meta: dict) -> np.ndarray:
    """擦拭：把 dirt 网格的未来变化摊到表面采样点上（`plan/02` §3.1，D-42）。

    dirt 只减不增，所以"变化"就是这一格在窗口内**被擦干净**。把每个格心映射到
    最近的表面采样点，于是它与 region 用的是同一套点——两个字段可以直接比对，
    这也是 `plan/02` §7 第 9 条那个可推导性探针能做的前提。
    """
    dirt = np.asarray(arrays["object/dirt_grid"], dtype=np.float32)
    n_frames, gx, gy = dirt.shape
    cell = float(meta.get("cell_m", 0.01))
    region = meta.get("region_m", [gx * cell, gy * cell])
    xs = (np.arange(gx) + 0.5) * cell - region[0] / 2
    ys = (np.arange(gy) + 0.5) * cell - region[1] / 2
    top_z = float(surface.points[:, 2].max())
    centers = np.stack(np.broadcast_arrays(xs[:, None], ys[None, :],
                                           np.full((gx, gy), top_z)), axis=-1)
    idx, ok, _ = assign_to_surface(centers.reshape(-1, 3), surface, max_dist=0.05)
    par = surface.parent[EFFECT_FIELD_LEVEL][idx]
    out = np.zeros((n_frames, FUTURE_SAMPLES, EFFECT_FIELD_LEVEL), dtype=np.float32)
    flat = dirt.reshape(n_frames, -1)
    for j, off in enumerate(offsets):
        last = n_frames - off
        if last <= 0:
            continue
        cleaned = (flat[:last] & ~flat[off:off + last]) if flat.dtype == bool else \
            np.clip(flat[:last] - flat[off:off + last], 0, None)
        w = cleaned.astype(np.float32) * ok[None, :]
        np.add.at(out[:last, j], (slice(None), par), w.T.T)
    return out


def _future_window(eff: np.ndarray, spec: TaskSpec,
                   meta: dict) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """未来 1 s、10 个采样点的 effect **增量**（`plan/02` §3.1）。

    自由体的位姿增量转进**物体当前系**，于是整个场景转一下时它逐元素不变
    （`plan/02` §7 第 1 条）。关节量本来就是标量，不需要转。
    末尾不足一个窗口的地方 `future_valid=False`，**不外插**。
    """
    n_frames = eff.shape[0]
    hz = float(meta.get("control_hz", 50.0))
    stride = max(1, int(round(FUTURE_HORIZON_S * hz / FUTURE_SAMPLES)))
    offsets = np.arange(1, FUTURE_SAMPLES + 1) * stride

    if spec.effect_is_pose:
        pos, quat = eff[:, :3], eff[:, 3:7]
        rot = _quat_to_rot(quat)
        dim, names = 6, spec.effect_names
    else:
        dim, names = eff.shape[1], spec.effect_names

    future = np.zeros((n_frames, FUTURE_SAMPLES, dim), dtype=np.float64)
    ok = np.zeros((n_frames, FUTURE_SAMPLES), dtype=bool)
    for j, off in enumerate(offsets):
        last = n_frames - off
        if last <= 0:
            continue
        t = np.arange(last)
        if spec.effect_is_pose:
            # 位置增量转进当前帧的物体系；姿态增量写成物体系里的旋转矢量
            dp = np.einsum("tij,tj->ti", rot[t].transpose(0, 2, 1), pos[t + off] - pos[t])
            dr = _rotvec(np.einsum("tij,tjk->tik", rot[t].transpose(0, 2, 1), rot[t + off]))
            future[t, j] = np.concatenate([dp, dr], axis=1)
        else:
            future[t, j] = eff[t + off] - eff[t]
        ok[t, j] = True
    return future, ok, names


def part_force_share(record: EpisodeRecord, surface: Surface,
                     phase: int | None = None) -> dict[str, float]:
    """S4 记录里接触力按物体部件的分布——用来与 S3 的接触部位统计对拍。

    抽屉的 S3 实测是"把手横杆背面 90.3%、正面 9.6%、其余 0%"。提取器如果把
    法向定向或表面归属做错了，这张表立刻对不上；而成功率、接触力、脏帧比例
    全都看不出来（D-34）。
    """
    idx = np.asarray(record.arrays["region/point_idx"])
    w = np.asarray(record.arrays["region/weight"], dtype=np.float64)
    ok = np.asarray(record.arrays["region/valid"]) & (idx >= 0)
    if phase is not None:
        ok = ok & (np.asarray(record.arrays["phase"]) == phase)[:, None]
    if not ok.any():
        return {name: 0.0 for name in surface.parts}
    heat = np.zeros(surface.n_points)
    np.add.at(heat, idx[ok], w[ok])
    return surface.part_share(heat)


def region_heatmap(record: EpisodeRecord, n_points: int,
                   phase: int | None = None) -> np.ndarray:
    """把一条 episode 的 region 累计成 (n_points,) 的热图（法向力加权）。"""
    idx = np.asarray(record.arrays["region/point_idx"])
    w = np.asarray(record.arrays["region/weight"], dtype=np.float64)
    ok = np.asarray(record.arrays["region/valid"]) & (idx >= 0)
    if phase is not None:
        ok = ok & (np.asarray(record.arrays["phase"]) == phase)[:, None]
    heat = np.zeros(n_points)
    if ok.any():
        np.add.at(heat, idx[ok], w[ok])
    return heat


_ = math
