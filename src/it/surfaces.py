"""被操作物体的表面采样：冻结的点 + 外法向 + 部件标签（物体局部系）。

`plan/02` §1 要求"同一物体在所有执行器实验中使用同一坐标系和**同一表面采样**"，
§2.2 要求每个采样点带**坐标 / 表面法向 / 部件标签**。Interaction Region
（§3.2）就是把接触点按法向力加权摊到这些点上得到的热图。

**为什么是解析采样而不是读 USD 网格**：D-07 规定资产由参数生成，参数在
`it.geom_cfg`（不依赖 pxr）。从同一份参数解析地采样，本机无 Isaac 也能算、
能测、能复现；读 USD 则要起 Isaac Sim，还会把"资产文件当前是什么样"这个
额外状态引进来。

三件设计上要留意的事：

1. **嵌套多分辨率**。先按面积均匀撒候选点，再用 FPS（最远点采样）定序，
   于是前 64 个 ⊂ 前 256 个 ⊂ 前 1024 个 ⊂ 全部 4096 个。S4.5 要扫
   {64, 256, 1024, 4096}（`plan/05` 实验零），四套**互不包含**的采样会把
   "分辨率的影响"与"采样点恰好落在哪的影响"混在一起，那个扫描就白做了。
   `parent[level]` 给出每个点在低分辨率下的代表点，热图可无损池化下去。

2. **埋在别的部件里的面要剔掉**。销钉底面贴在圆盘顶面上、圆盘顶面被销钉盖住的
   那一圈，都不是可接触的表面。留着它们会让 `plan/03` §8.1 的 width
   （region 占物体表面积的比例）分母虚高，看起来 envelope 比实际更窄。

3. **采样与几何变体绑定**。`GEOM_VARIANTS` 会改把手净空、销钉偏心距、
   黑板擦长度，采样必须跟着 episode 的 `geometry_variant` 走，
   否则 `unseen_geometry_test` 那一档的接触点会被归到错误的表面点上。

用法::

    from it.surfaces import surface_for
    s = surface_for("drawer", "g1")        # 800 条抽屉里的几何变体那一档
    s.points.shape, s.normals.shape        # (4096, 3), (4096, 3)
    s.parts[s.part[i]]                     # 第 i 个点在哪个部件上
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from it import geom_cfg as G

#: S4.5 要扫的四档（`plan/05` 实验零 §1.1）。它们是采样序列的**前缀**，
#: 因此低分辨率是高分辨率的真子集。基准点数可以比 4096 更多（见 `_base_n`），
#: 这四档照样是它的前缀。
LEVELS = (64, 256, 1024, 4096)
N_FULL = LEVELS[-1]

#: 目标 pitch（m）。`plan/02` §2.1 明确"按目标 pitch 换算点数，而非固定点数"，
#: 那里给的目标是 2 mm；这里取 3 mm 并把点数封在 16384，是算力上的折中——
#: 逐部件 FPS 的代价随点数线性涨，而 600×500 的平面按 2 mm 要 16 万点。
#: 真正的分辨率结论由 S4.5 的扫描给，本模块只保证**够细且嵌套**。
_TARGET_PITCH = 3.0e-3
_N_MIN, _N_MAX = 4096, 16384

#: 每个部件至少分到的点数。**这一条不是美观问题**：抽屉把手横杆只占物体总面积的
#: 3.5%，纯按面积分配时"横杆背面"（S3 实测 90.3% 的接触落在那里）只有 20 个采样点，
#: region 热图在最要紧的部位上反而最糊。有了地板，小而关键的部件也能被解析出来。
_PART_FLOOR = 64

#: 候选点数 = 输出点数 × 这个倍数。FPS 从候选里挑，倍数越大点越均匀。
_OVERSAMPLE = 4

#: 剔除"埋在别的部件里"的候选点时，实体外扩这么多（m）。
#: 必须 > 0：销钉底面与圆盘顶面**恰好共面**，取 0 的话两边都判成"不在内部"，
#: 于是贴合面上会留下两层永远接触不到的采样点。
_BURY_MARGIN = 0.5e-3


#: 力散射用的**固定物理带宽**（m）。它是"我们假设的触觉分辨率"，
#: 与表面采样分辨率**无关**——这一点是整件事的关键。
#:
#: ⚠️ D-58 早就写明"直接 nearest-point force/area 会随分辨率产生伪尖峰，禁止使用"，
#: 但代码里一直是最近点散射。实测把命令表面从 64 格换到 1024 格，擦拭的 traction
#: 中位数从 179 涨到 2500 N/m²——几乎正好按 1/面积 线性放大，因为接触斑块远小于格子，
#: 力整个落进一格，于是 F/A 里的 A 是**我们任意选的采样分辨率**（P-68）。
#:
#: 取 4 mm 是与 `plan/07` 的触觉 taxel pitch 同量级；它必须随产物一起记录，
#: 因为 traction 的数值含义就是"在这个尺度上平滑后的面密度"。
SCATTER_SIGMA = 4.0e-3
#: 每个表面点保留多少个同部件邻居。3σ 截断 + 3 mm pitch 下约 50 个，取 64 留余量。
SCATTER_K = 64


@dataclass(frozen=True)
class Surface:
    """一个物体的冻结表面采样（物体局部系，单位 m）。"""

    obj: str
    geom_tag: str
    points: np.ndarray          # (S, 3) float32
    normals: np.ndarray         # (S, 3) float32，外法向，已归一化
    part: np.ndarray            # (S,) int8，索引到 `parts`
    parts: tuple[str, ...]
    area: np.ndarray            # (S,) float32，每个点代表的面积
    parent: dict[int, np.ndarray]   # level -> (S,) int32，该点在该档的代表点
    sha256: str
    #: 判"点在不在物体内部"的实体列表（造面片时用的同一套），以及本采样相对
    #: 标准姿态的旋转。接触点常常**穿进物体内部**（撞击瞬态下实测球上中位
    #: 6 mm），那不是归属错误而是穿透，两者必须分开判。
    solids: tuple = ()
    rot: np.ndarray | None = None

    def inside(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """这些点在不在物体实体内部。(N,3) -> (N,) bool。"""
        if not self.solids:
            return np.zeros(len(points), dtype=bool)
        q = points if self.rot is None else np.einsum("ji,nj->ni", self.rot, points)
        out = np.zeros(len(q), dtype=bool)
        for solid in self.solids:
            out |= solid(np.asarray(q, dtype=np.float64), margin)
        return out

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def scatter_kernel(self, sigma: float = SCATTER_SIGMA, k: int = SCATTER_K
                       ) -> tuple[np.ndarray, np.ndarray]:
        """把一个接触点的力摊到邻近表面点的核。返回 (邻居下标 (S,k), 权重 (S,k))。

        三条性质缺一不可：

        1. **同部件**。跨部件摊力会把正面的接触摊到背面去——薄物体上尤其致命，
           与 D-68 修的池化是同一条规矩（`plan/02` 的力学量都定义在部件内部）；
        2. **合力守恒**。每行权重归一化到 1，散射前后合力逐位相等；
        3. **带宽是固定物理尺度**，不随表面分辨率变。这样 traction 才是
           "在 4 mm 尺度上的面密度"，而不是"力 ÷ 我们碰巧选的格子"（P-68）。

        权重按代表面积加权，避免采样密度不均的地方被高估。
        """
        cache = getattr(self, "_kernel_cache", None)
        if cache is not None and cache[0] == (sigma, k):
            return cache[1], cache[2]
        points = np.asarray(self.points, dtype=np.float64)
        part = np.asarray(self.part)
        area = np.asarray(self.area, dtype=np.float64)
        n = len(points)
        width = min(k, n)
        index = np.zeros((n, width), dtype=np.int32)
        weight = np.zeros((n, width), dtype=np.float64)
        cutoff = (3.0 * sigma) ** 2
        for label in np.unique(part):
            rows = np.flatnonzero(part == label)
            local = points[rows]
            for start in range(0, len(rows), 512):
                block = local[start:start + 512]
                d2 = ((block[:, None, :] - local[None, :, :]) ** 2).sum(-1)
                d2 = np.where(d2 <= cutoff, d2, np.inf)
                take = min(width, len(rows))
                order = np.argpartition(d2, take - 1, axis=1)[:, :take]
                picked = np.take_along_axis(d2, order, axis=1)
                w = np.where(np.isfinite(picked), np.exp(-picked / (2 * sigma ** 2)), 0.0)
                w = w * area[rows[order]]
                total = w.sum(axis=1, keepdims=True)
                # 孤立点（3σ 内只有自己）退化成把力全给自己——仍然守恒。
                degenerate = total[:, 0] <= 0
                if degenerate.any():
                    w[degenerate] = 0.0
                    w[degenerate, 0] = 1.0
                    order[degenerate, 0] = np.arange(len(rows))[start:start + 512][degenerate]
                    total[degenerate] = 1.0
                index[rows[start:start + 512], :take] = rows[order]
                weight[rows[start:start + 512], :take] = w / total
        object.__setattr__(self, "_kernel_cache", ((sigma, k), index, weight))
        return index, weight

    @property
    def total_area(self) -> float:
        return float(self.area.sum())

    def part_share(self, weight: np.ndarray) -> dict[str, float]:
        """把逐点权重（通常是累计法向力）按部件汇总成占比。"""
        total = float(np.abs(weight).sum())
        if total <= 0.0:
            return {name: 0.0 for name in self.parts}
        return {
            name: float(np.abs(weight)[self.part == i].sum() / total)
            for i, name in enumerate(self.parts)
        }

    def rotated(self, rot: np.ndarray) -> "Surface":
        """整体旋转 R 之后的同一份采样（点序、部件、面积、hash 全部不变）。

        给 `plan/02` §7 第 1 条的等变性测试用：把接触数据和表面采样一起转，
        region 索引就该逐元素不变。**hash 保持原值**是有意的——它标识的是
        "哪一份采样"，不是"这份采样此刻摆成什么姿态"。
        """
        rot = np.asarray(rot, dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(f"旋转矩阵形状应为 (3,3)，实际 {rot.shape}")
        return Surface(
            obj=self.obj, geom_tag=self.geom_tag,
            # 用 einsum 而不是 `@`：某些 BLAS 后端在这种小矩阵 + 混精度上会刷
            # 一串无害但吵人的 RuntimeWarning，而产物日志要能一眼看出真问题。
            points=np.einsum("ij,nj->ni", rot, self.points.astype(np.float64)
                             ).astype(np.float32),
            normals=np.einsum("ij,nj->ni", rot, self.normals.astype(np.float64)
                              ).astype(np.float32),
            part=self.part, parts=self.parts, area=self.area,
            parent=self.parent, sha256=self.sha256, solids=self.solids,
            rot=rot if self.rot is None else rot @ self.rot,
        )


SURFACE_SCHEMA_VERSION = "frozen-surface-v1"


def _surface_sha256(obj: str, geom_tag: str, points: np.ndarray,
                    normals: np.ndarray, part: np.ndarray) -> str:
    """与历史 S4 surface hash 完全相同的摘要算法。"""
    digest = hashlib.sha256()
    digest.update(f"{obj}|{geom_tag}|{len(points)}".encode("utf-8"))
    for arr in (np.round(points, 9), np.round(normals, 9)):
        digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(part, dtype=np.int8).tobytes())
    return digest.hexdigest()


def _surface_content_sha256(points: np.ndarray, normals: np.ndarray, part: np.ndarray,
                            area: np.ndarray, parent: dict[int, np.ndarray]) -> str:
    """对实际落盘数组做可重算的完整摘要（历史 surface.sha256 做不到这一点）。"""
    digest = hashlib.sha256()
    for arr in (points, normals, part, area):
        value = np.asarray(arr)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    for level, value in sorted(parent.items()):
        digest.update(str(level).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def save_surface(surface: Surface, path: str | os.PathLike[str]) -> str:
    """保存冻结 surface 本身，而不要求下游在另一 NumPy 环境重新生成。

    FPS 在几何对称点上可能出现跨 BLAS/NumPy 版本的 tie-breaking 差异；只在 S4
    record 中写 hash 仍不足以复现点序。S4 数据集应把这个 artifact 与 manifest 同存。
    """
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("frozen surface 路径必须以 .npz 结尾")
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "object": surface.obj,
        "geom_tag": surface.geom_tag,
        "parts": list(surface.parts),
        "sha256": surface.sha256,
        "content_sha256": _surface_content_sha256(
            surface.points, surface.normals, surface.part, surface.area, surface.parent),
    }
    payload = {
        "__meta__": np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        "points": np.asarray(surface.points),
        "normals": np.asarray(surface.normals),
        "part": np.asarray(surface.part),
        "area": np.asarray(surface.area),
    }
    payload.update({f"parent/{level}": np.asarray(parent)
                    for level, parent in surface.parent.items()})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)
    return str(destination.resolve())


def load_surface(path: str | os.PathLike[str]) -> Surface:
    """读取冻结 surface，并校验点序 hash 与全部数组形状。"""
    with np.load(path, allow_pickle=False) as data:
        if "__meta__" not in data.files or data["__meta__"].ndim != 0:
            raise ValueError(f"{path} 缺少标量 __meta__")
        meta = json.loads(str(data["__meta__"].item()))
        if meta.get("schema_version") != SURFACE_SCHEMA_VERSION:
            raise ValueError(f"不支持的 surface schema：{meta.get('schema_version')!r}")
        points = np.array(data["points"], copy=True)
        normals = np.array(data["normals"], copy=True)
        part = np.array(data["part"], copy=True)
        area = np.array(data["area"], copy=True)
        parent = {int(name.split("/", 1)[1]): np.array(data[name], copy=True)
                  for name in data.files if name.startswith("parent/")}
    n = len(points)
    if points.shape != (n, 3) or normals.shape != (n, 3):
        raise ValueError("surface points/normals 必须是 (S,3)")
    if part.shape != (n,) or area.shape != (n,) or np.any(area <= 0):
        raise ValueError("surface part/area 形状或数值非法")
    if any(value.shape != (n,) or np.any(value < 0) or np.any(value >= level)
           for level, value in parent.items()):
        raise ValueError("surface parent 映射非法")
    content_hash = _surface_content_sha256(points, normals, part, area, parent)
    if meta.get("content_sha256") is not None and content_hash != meta["content_sha256"]:
        raise ValueError("frozen surface 内容与 meta.content_sha256 不一致")
    # 历史 surface.sha256 是对 cast float32 *之前*的 float64 点计算的，无法从 S4
    # 保存的 float32 surface 反算。它继续作为 record 与 artifact 的 identity key；
    # 新增的 content_sha256 才承担文件内容完整性校验。
    identity_hash = str(meta.get("sha256", ""))
    if len(identity_hash) != 64:
        raise ValueError("frozen surface meta.sha256 非法")
    return Surface(
        obj=str(meta["object"]), geom_tag=str(meta["geom_tag"]),
        points=points, normals=normals, part=part,
        parts=tuple(str(item) for item in meta["parts"]), area=area,
        parent=parent, sha256=identity_hash,
    )


# ---------------------------------------------------------------- 参数化面片


@dataclass
class _Patch:
    """一块可参数化采样的面：``sample(u, v) -> (点, 外法向)``，附带面积与部件标签。"""

    area: float
    part: str
    fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _plane(origin, edge_u, edge_v, normal, part: str) -> _Patch:
    o = np.asarray(origin, dtype=np.float64)
    a = np.asarray(edge_u, dtype=np.float64)
    b = np.asarray(edge_v, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    area = float(np.linalg.norm(np.cross(a, b)))

    def fn(u, v):
        p = o[None, :] + u[:, None] * a[None, :] + v[:, None] * b[None, :]
        return p, np.repeat(n[None, :], len(u), axis=0)

    return _Patch(area=area, part=part, fn=fn)


def _box_patches(size, center, part_of_face) -> list[_Patch]:
    """长方体的六个面。``part_of_face`` 按 (+X,-X,+Y,-Y,+Z,-Z) 给部件名。"""
    sx, sy, sz = (float(v) for v in size)
    c = np.asarray(center, dtype=np.float64)
    h = np.array([sx, sy, sz]) / 2.0
    axes = np.eye(3)
    out: list[_Patch] = []
    for ax in range(3):
        i, j = (ax + 1) % 3, (ax + 2) % 3
        for k, sgn in enumerate((1.0, -1.0)):
            n = sgn * axes[ax]
            # 让 (edge_u, edge_v, n) 成右手系，法向朝外
            eu = axes[i] * (2 * h[i])
            ev = axes[j] * (2 * h[j]) * sgn
            origin = c + n * h[ax] - eu / 2 - ev / 2
            out.append(_plane(origin, eu, ev, n, part_of_face[2 * ax + k]))
    return out


def _cyl_patches(radius: float, height: float, axis: str, center,
                 part_side: str, part_cap_pos: str, part_cap_neg: str,
                 cap_pos: bool = True, cap_neg: bool = True,
                 side: bool = True) -> list[_Patch]:
    """圆柱的侧面与两个端盖。``axis`` ∈ {"X","Y","Z"}，``height`` 是全长。"""
    c = np.asarray(center, dtype=np.float64)
    k = "XYZ".index(axis.upper())
    e_ax = np.eye(3)[k]
    e1, e2 = np.eye(3)[(k + 1) % 3], np.eye(3)[(k + 2) % 3]
    half = height / 2.0
    out: list[_Patch] = []

    def side_fn(u, v):
        th = 2.0 * math.pi * u
        radial = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
        p = c[None, :] + radius * radial + ((2.0 * v - 1.0) * half)[:, None] * e_ax[None, :]
        return p, radial

    if side:
        out.append(_Patch(area=2.0 * math.pi * radius * height, part=part_side,
                          fn=side_fn))

    for want, sgn, part in ((cap_pos, 1.0, part_cap_pos), (cap_neg, -1.0, part_cap_neg)):
        if not want:
            continue

        def cap(u, v, sgn=sgn):
            # 极坐标下按 sqrt(v) 取半径，才是圆盘上的均匀分布
            th = 2.0 * math.pi * u
            r = radius * np.sqrt(v)
            radial = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
            p = c[None, :] + r[:, None] * radial + sgn * half * e_ax[None, :]
            return p, np.repeat((sgn * e_ax)[None, :], len(u), axis=0)

        out.append(_Patch(area=math.pi * radius ** 2, part=part, fn=cap))
    return out


def _cyl_sector(radius: float, height: float, axis: str, center,
                ang_lo: float, ang_hi: float, part: str,
                ref_axis: str = "X") -> _Patch:
    """圆柱侧面上的一段角度区间。角度从 ``ref_axis`` 起算，按右手系绕 ``axis``。

    抽屉把手要按"背/前/上/下"分部件，而那个划分**是角度定义的**
    （`it.contact_attrib.classify_drawer_local`）。直接按角度切面片，
    好过先整块采样再事后改标签——后者会让"部件配额"无从谈起。
    """
    c = np.asarray(center, dtype=np.float64)
    k = "XYZ".index(axis.upper())
    e_ax = np.eye(3)[k]
    kr = "XYZ".index(ref_axis.upper())
    e1 = np.eye(3)[kr]
    e2 = np.cross(e_ax, e1)
    # 抽屉的角度是 atan2(z - hz, x - hx)，即从 +X 转向 +Z；杆轴是 +Y，
    # 而 cross(+Y, +X) = -Z，所以这里取负号让转向与那个 atan2 一致。
    e2 = -e2 if k == 1 and kr == 0 else e2
    half = height / 2.0
    span = ang_hi - ang_lo

    def fn(u, v):
        th = ang_lo + span * u
        radial = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
        p = c[None, :] + radius * radial + ((2.0 * v - 1.0) * half)[:, None] * e_ax[None, :]
        return p, radial

    return _Patch(area=abs(span) * radius * height, part=part, fn=fn)


def _sphere_patches(radius: float, center, part: str) -> list[_Patch]:
    c = np.asarray(center, dtype=np.float64)

    def fn(u, v):
        z = 2.0 * v - 1.0                      # 等面积：z 均匀
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        th = 2.0 * math.pi * u
        n = np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1)
        return c[None, :] + radius * n, n

    return [_Patch(area=4.0 * math.pi * radius ** 2, part=part, fn=fn)]


# ---------------------------------------------------------------- 实体（用于剔除埋点）


def _inside_box(p: np.ndarray, size, center, margin: float) -> np.ndarray:
    h = np.asarray(size, dtype=np.float64) / 2.0 + margin
    d = np.abs(p - np.asarray(center, dtype=np.float64))
    return np.all(d < h, axis=-1)


def _inside_cyl(p: np.ndarray, radius: float, height: float, axis: str,
                center, margin: float) -> np.ndarray:
    k = "XYZ".index(axis.upper())
    d = p - np.asarray(center, dtype=np.float64)
    along = np.abs(d[:, k]) < height / 2.0 + margin
    others = [i for i in range(3) if i != k]
    radial = np.linalg.norm(d[:, others], axis=-1) < radius + margin
    return along & radial


def _inside_sphere(p: np.ndarray, radius: float, center, margin: float) -> np.ndarray:
    return np.linalg.norm(p - np.asarray(center, dtype=np.float64), axis=-1) < radius + margin


# ---------------------------------------------------------------- 物体定义

#: 抽屉的部件名与 `it.contact_attrib.DRAWER_PARTS` **必须逐字一致**——
#: S3 的接触部位统计（把手背面 90.3% 等）就是按那套名字报的，S4 要能对拍。
_DRAWER_PARTS = ("bar_back", "bar_front", "bar_top", "bar_bottom",
                 "post", "panel", "other")


def _drawer_patches(cfg) -> tuple[list[_Patch], list[Callable], tuple[str, ...]]:
    inner_w = cfg.panel_w - 2 * cfg.wall_t
    hx = cfg.panel_t + cfg.handle_clearance + cfg.handle_radius
    hz = cfg.panel_h / 2
    post_len = cfg.handle_clearance + cfg.handle_radius
    post_cx = cfg.panel_t + post_len / 2

    patches: list[_Patch] = []
    patches += _box_patches((cfg.panel_t, cfg.panel_w, cfg.panel_h),
                            (cfg.panel_t / 2, 0.0, cfg.panel_h / 2),
                            ("panel",) * 6)
    patches += _box_patches((cfg.tray_depth, inner_w - 2 * G.MM, cfg.tray_t),
                            (-cfg.tray_depth / 2, 0.0, cfg.tray_t / 2),
                            ("other",) * 6)
    # 横杆按**绕轴角度**切成四块。判据与 `it.contact_attrib.classify_drawer_local`
    # 逐字一致（±3π/4 以外是背面、±π/4 以内是正面），S4 的 region 热图按部件汇总
    # 之后才能与 S3 已验收的接触部位统计直接对拍。
    for part, (lo, hi) in (("bar_front", (-math.pi / 4, math.pi / 4)),
                           ("bar_top", (math.pi / 4, 3 * math.pi / 4)),
                           ("bar_back", (3 * math.pi / 4, 5 * math.pi / 4)),
                           ("bar_bottom", (-3 * math.pi / 4, -math.pi / 4))):
        patches.append(_cyl_sector(cfg.handle_radius, cfg.handle_bar_len, "Y",
                                   (hx, 0.0, hz), lo, hi, part, ref_axis="X"))
    # 两个端盖（|y| = 70 mm 处）。`classify_drawer_local` 把它们也算作"在杆上"，
    # 按角度归类，这里跟着它走：端盖点的角度用同一个 atan2 判。
    patches += _cyl_patches(cfg.handle_radius, cfg.handle_bar_len, "Y",
                            (hx, 0.0, hz), "bar_back", "bar_back", "bar_back",
                            side=False)
    for sgn in (1.0, -1.0):
        patches += _cyl_patches(cfg.post_radius, post_len, "X",
                                (post_cx, sgn * cfg.post_spacing / 2, hz),
                                "post", "post", "post")

    solids = [
        lambda p, m: _inside_box(p, (cfg.panel_t, cfg.panel_w, cfg.panel_h),
                                 (cfg.panel_t / 2, 0.0, cfg.panel_h / 2), m),
        lambda p, m: _inside_box(p, (cfg.tray_depth, inner_w - 2 * G.MM, cfg.tray_t),
                                 (-cfg.tray_depth / 2, 0.0, cfg.tray_t / 2), m),
        lambda p, m: _inside_cyl(p, cfg.handle_radius, cfg.handle_bar_len, "Y",
                                 (hx, 0.0, hz), m),
        lambda p, m: _inside_cyl(p, cfg.post_radius, post_len, "X",
                                 (post_cx, cfg.post_spacing / 2, hz), m),
        lambda p, m: _inside_cyl(p, cfg.post_radius, post_len, "X",
                                 (post_cx, -cfg.post_spacing / 2, hz), m),
    ]
    return patches, solids, _DRAWER_PARTS


def _knob_patches(cfg):
    parts = ("pin_side", "pin_top", "disc_top", "disc_bottom", "rim_side")
    pin_z = cfg.disc_thickness / 2 + cfg.pin_length / 2
    patches: list[_Patch] = []
    patches += _cyl_patches(cfg.disc_radius, cfg.disc_thickness, "Z", (0.0, 0.0, 0.0),
                            "rim_side", "disc_top", "disc_bottom")
    patches += _cyl_patches(cfg.pin_radius, cfg.pin_length, "Z",
                            (cfg.pin_offset, 0.0, pin_z),
                            "pin_side", "pin_top", "pin_top", cap_neg=False)
    solids = [
        lambda p, m: _inside_cyl(p, cfg.disc_radius, cfg.disc_thickness, "Z",
                                 (0.0, 0.0, 0.0), m),
        lambda p, m: _inside_cyl(p, cfg.pin_radius, cfg.pin_length, "Z",
                                 (cfg.pin_offset, 0.0, pin_z), m),
    ]
    return patches, solids, parts


def _board_patches(size=(600 * G.MM, 500 * G.MM, 20 * G.MM)):
    """擦拭平面。局部 +Z 是工作面（`assets.board_cfg` 把它摆到世界 z=0）。"""
    parts = ("work_face", "side", "back")
    patches = _box_patches(size, (0.0, 0.0, 0.0),
                           ("side", "side", "side", "side", "work_face", "back"))
    solids = [lambda p, m: _inside_box(p, size, (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _block_patches(cfg):
    parts = ("face_px", "face_nx", "face_py", "face_ny", "top", "bottom")
    patches = _box_patches(cfg.size, (0.0, 0.0, 0.0), parts)
    solids = [lambda p, m: _inside_box(p, cfg.size, (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _column_patches(cfg):
    parts = ("side", "top", "bottom")
    patches = _cyl_patches(cfg.radius, cfg.height, "Z", (0.0, 0.0, 0.0),
                           "side", "top", "bottom")
    solids = [lambda p, m: _inside_cyl(p, cfg.radius, cfg.height, "Z",
                                       (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _roller_patches(cfg):
    parts = ("side", "end_py", "end_ny")
    patches = _cyl_patches(cfg.radius, cfg.length, "Y", (0.0, 0.0, 0.0),
                           "side", "end_py", "end_ny")
    solids = [lambda p, m: _inside_cyl(p, cfg.radius, cfg.length, "Y",
                                       (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _ball_patches(cfg):
    parts = ("sphere",)
    patches = _sphere_patches(cfg.radius, (0.0, 0.0, 0.0), "sphere")
    solids = [lambda p, m: _inside_sphere(p, cfg.radius, (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _slider_patches(cfg):
    """接触系是 ``/Slider/Block``：块在原点，挡片沉在块顶上、内缩 6 mm。"""
    parts = ("block_face_px", "block_face_nx", "block_side", "block_top",
             "block_bottom", "tab_px", "tab_nx", "tab_side", "tab_top")
    bw, bd, bh = cfg.block
    tw, td, th = cfg.tab
    tc = (bw / 2 - tw / 2 - 6 * G.MM, 0.0, bh / 2 + th / 2 - 6 * G.MM)
    patches = _box_patches((bw, bd, bh), (0.0, 0.0, 0.0),
                           ("block_face_px", "block_face_nx", "block_side",
                            "block_side", "block_top", "block_bottom"))
    patches += _box_patches((tw, td, th), tc,
                            ("tab_px", "tab_nx", "tab_side", "tab_side",
                             "tab_top", "tab_top"))
    solids = [
        lambda p, m: _inside_box(p, (bw, bd, bh), (0.0, 0.0, 0.0), m),
        lambda p, m: _inside_box(p, (tw, td, th), tc, m),
    ]
    return patches, solids, parts


def _plunger_patches(cfg):
    """接触系是 ``/Plunger/Rod``：杆沿 X，端帽在 +X 端。"""
    parts = ("rod_side", "rod_back", "cap_side", "cap_front", "cap_back")
    cap_c = (cfg.rod_len / 2 - cfg.cap_len / 2, 0.0, 0.0)
    patches = _cyl_patches(cfg.rod_radius, cfg.rod_len, "X", (0.0, 0.0, 0.0),
                           "rod_side", "rod_side", "rod_back")
    patches += _cyl_patches(cfg.cap_radius, cfg.cap_len, "X", cap_c,
                            "cap_side", "cap_front", "cap_back")
    solids = [
        lambda p, m: _inside_cyl(p, cfg.rod_radius, cfg.rod_len, "X",
                                 (0.0, 0.0, 0.0), m),
        lambda p, m: _inside_cyl(p, cfg.cap_radius, cfg.cap_len, "X", cap_c, m),
    ]
    return patches, solids, parts


def _dial_patches(cfg):
    parts = ("disc_top", "disc_side", "disc_bottom", "lug_side", "lug_top")
    lug_z = cfg.disc_thickness / 2 + cfg.lug_height / 2
    centers = [(cfg.lug_offset * math.cos(2 * math.pi * i / 3),
                cfg.lug_offset * math.sin(2 * math.pi * i / 3), lug_z)
               for i in range(3)]
    patches = _cyl_patches(cfg.disc_radius, cfg.disc_thickness, "Z", (0.0, 0.0, 0.0),
                           "disc_side", "disc_top", "disc_bottom")
    for c in centers:
        patches += _cyl_patches(cfg.lug_radius, cfg.lug_height, "Z", c,
                                "lug_side", "lug_top", "lug_top", cap_neg=False)
    solids = [lambda p, m: _inside_cyl(p, cfg.disc_radius, cfg.disc_thickness, "Z",
                                       (0.0, 0.0, 0.0), m)]
    solids += [(lambda c: (lambda p, m: _inside_cyl(p, cfg.lug_radius, cfg.lug_height,
                                                    "Z", c, m)))(c) for c in centers]
    return patches, solids, parts


def _flap_patches(cfg):
    """接触系是 ``/Flap/Panel``：板 10(X)×150(Y)×120(Z)，铰链在局部 y = −pd/2−r。"""
    parts = ("face_px", "face_nx", "edge_far", "edge_hinge", "top", "bottom")
    pw, pd, ph = cfg.panel
    patches = _box_patches((pw, pd, ph), (0.0, 0.0, 0.0),
                           ("face_px", "face_nx", "edge_far", "edge_hinge",
                            "top", "bottom"))
    solids = [lambda p, m: _inside_box(p, (pw, pd, ph), (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


def _ridge_patches(cfg):
    parts = ("base_top", "base_side", "base_bottom", "ridge_side", "ridge_end")
    bw, bd, bh = cfg.base
    rc = (0.0, 0.0, bh / 2 + cfg.ridge_radius)
    patches = _box_patches((bw, bd, bh), (0.0, 0.0, 0.0),
                           ("base_side", "base_side", "base_side", "base_side",
                            "base_top", "base_bottom"))
    patches += _cyl_patches(cfg.ridge_radius, cfg.ridge_len, "Y", rc,
                            "ridge_side", "ridge_end", "ridge_end")
    solids = [
        lambda p, m: _inside_box(p, (bw, bd, bh), (0.0, 0.0, 0.0), m),
        lambda p, m: _inside_cyl(p, cfg.ridge_radius, cfg.ridge_len, "Y", rc, m),
    ]
    return patches, solids, parts


def _slab_patches(cfg):
    """接触系是 ``/Slab/Board``：15° 倾角在**帧的姿态**里，局部几何就是个正方体。"""
    parts = ("top", "side", "bottom")
    patches = _box_patches(cfg.size, (0.0, 0.0, 0.0),
                           ("side", "side", "side", "side", "top", "bottom"))
    solids = [lambda p, m: _inside_box(p, cfg.size, (0.0, 0.0, 0.0), m)]
    return patches, solids, parts


#: 物体名 -> (取 cfg 的 geom_cfg 名字, 造面片的函数)。
#: 键就是数据集里 `task` / `probe_object` 用的名字，中间不再转一层。
_OBJECTS: dict[str, tuple[str | None, Callable]] = {
    "drawer": ("cabinet", _drawer_patches),
    "knob": ("knob", _knob_patches),
    "board": (None, lambda _: _board_patches()),
    "block": ("block", _block_patches),
    "column": ("column", _column_patches),
    "roller": ("roller", _roller_patches),
    "ball": ("ball", _ball_patches),
    "slider": ("slider", _slider_patches),
    "plunger": ("plunger", _plunger_patches),
    "dial": ("dial", _dial_patches),
    "flap": ("flap", _flap_patches),
    "ridge": ("ridge", _ridge_patches),
    "slab": ("slab", _slab_patches),
}

#: 哪些物体的几何会随 episode 变（`GEOM_VARIANTS`）。其余物体给 g1/g2 是错误，
#: 不是"沉默地退回 nominal"——那种退回会让 `unseen_geometry_test` 的点悄悄归错。
_VARIANT_OBJECTS = {"drawer": "cabinet", "knob": "knob"}


# ---------------------------------------------------------------- 采样


def _r2_sequence(n: int) -> tuple[np.ndarray, np.ndarray]:
    """R2 低差异序列。确定性、无 RNG 状态，两次调用逐位相同。"""
    g = 1.32471795724474602596
    a1, a2 = 1.0 / g, 1.0 / (g * g)
    i = np.arange(1, n + 1, dtype=np.float64)
    return np.mod(0.5 + a1 * i, 1.0), np.mod(0.5 + a2 * i, 1.0)


def _fps_order(points: np.ndarray, n: int, start: int) -> np.ndarray:
    """最远点采样的下标序列。前 k 个永远是"取 k 个点"的那个解，因此天然嵌套。"""
    d2 = np.full(len(points), np.inf)
    out = np.empty(n, dtype=np.int64)
    cur = start
    for i in range(n):
        out[i] = cur
        diff = points - points[cur]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
        d2[cur] = -1.0
        cur = int(np.argmax(d2))
    return out


def _nearest(query: np.ndarray, ref: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """每个 query 点最近的 ref 点下标。分块算，避免 N×M 一次性开出来。"""
    out = np.empty(len(query), dtype=np.int64)
    for s in range(0, len(query), chunk):
        q = query[s: s + chunk]
        d = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
        out[s: s + chunk] = np.argmin(d, axis=1)
    return out


def _base_n(area: float) -> int:
    """按目标 pitch 换算点数，取 2 的幂，封在 [_N_MIN, _N_MAX]。

    取 2 的幂只是为了让 `LEVELS` 那四档整整齐齐地落在前缀上，没有别的含义。
    """
    want = area / (_TARGET_PITCH ** 2)
    n = _N_MIN
    while n < want and n < _N_MAX:
        n *= 2
    return int(min(max(n, _N_MIN), _N_MAX))


def _quotas(part_area: np.ndarray, n_points: int) -> np.ndarray:
    """部件配额：先给每个部件一个地板，剩下的按面积分。

    没有地板时小部件会被大面积部件淹掉（抽屉横杆背面只剩 20 个点），
    而那恰恰是接触真正发生的地方。
    """
    n_parts = len(part_area)
    floor = min(_PART_FLOOR, max(1, n_points // (2 * n_parts)))
    rest = n_points - floor * n_parts
    if rest < 0:
        raise RuntimeError(f"点数 {n_points} 不够给 {n_parts} 个部件各 {floor} 个")
    extra = np.floor(rest * part_area / part_area.sum()).astype(int)
    quota = floor + extra
    # 取整的余数按面积从大到小补回去，保证总数精确等于 n_points
    for i in np.argsort(-part_area)[: n_points - int(quota.sum())]:
        quota[i] += 1
    while quota.sum() < n_points:            # 部件数很少时上面一轮可能补不够
        quota[int(np.argmax(part_area))] += 1
    return quota


def _interleave(orders: list[np.ndarray]) -> np.ndarray:
    """把逐部件的 FPS 序列交错成一条全局序列，保持"前缀 = 各部件的前缀之并"。

    每一步挑**当前填充比例最低**的那个部件（同比例时取部件下标小的），
    于是任意长度的前缀在各部件之间大致按配额分配，`LEVELS` 的每一档
    都能覆盖到全部部件——64 点那一档也不会整个部件是空的。
    """
    total = sum(len(o) for o in orders)
    taken = [0] * len(orders)
    quota = [len(o) for o in orders]
    out = np.empty(total, dtype=np.int64)
    for i in range(total):
        best, best_ratio = -1, None
        for k in range(len(orders)):
            if taken[k] >= quota[k]:
                continue
            ratio = taken[k] / quota[k]
            if best_ratio is None or ratio < best_ratio - 1e-12:
                best, best_ratio = k, ratio
        out[i] = orders[best][taken[best]]
        taken[best] += 1
    return out


def object_geometry(obj: str, geom_tag: str = "nominal"):
    """取某个物体的 (面片列表, 实体列表, 部件名)。

    实体列表是"点在不在物体内部"的判据，`_build` 用它剔除埋点，
    单元测试用它验法向朝外——两处必须是同一套几何，否则测试测的是另一个东西。
    """
    cfg_name, fn = _OBJECTS[obj]
    if cfg_name is None:
        return fn(None)
    if geom_tag != "nominal" and obj not in _VARIANT_OBJECTS:
        raise ValueError(
            f"{obj} 没有几何变体，却收到 geom_tag={geom_tag!r}；"
            "沉默地退回 nominal 会让变体 episode 的接触点归到错误的表面上")
    variant_name = _VARIANT_OBJECTS.get(obj)
    cfg = (G.variant_cfg(variant_name, geom_tag) if variant_name
           else getattr(G.BuildCfg(), cfg_name))
    return fn(cfg)


def _build(obj: str, geom_tag: str, n_points: int | None) -> Surface:
    patches, solids, parts = object_geometry(obj, geom_tag)

    total_area = sum(p.area for p in patches)
    if n_points is None:
        n_points = _base_n(total_area)

    # --- 按面积撒候选点 ---
    n_cand = n_points * _OVERSAMPLE
    areas = np.array([p.area for p in patches], dtype=np.float64)
    quota = np.maximum(8, np.round(n_cand * areas / areas.sum()).astype(int))
    pts, nrm, lab, wgt = [], [], [], []
    for patch, q in zip(patches, quota):
        u, v = _r2_sequence(int(q))
        p, n = patch.fn(u, v)
        pts.append(p)
        nrm.append(n)
        lab.append(np.full(len(p), parts.index(patch.part), dtype=np.int8))
        wgt.append(np.full(len(p), patch.area / len(p), dtype=np.float64))
    pts = np.concatenate(pts)
    nrm = np.concatenate(nrm)
    lab = np.concatenate(lab)
    wgt = np.concatenate(wgt)

    # --- 剔掉埋在别的部件里的点 ---
    # 一个点必然"在自己所属的实体表面上"，外扩后也会被自己判成内部。
    # 所以判据是**被至少两个实体盖住**（自己 + 别人）才算埋起来。
    cover = np.zeros(len(pts), dtype=np.int32)
    for solid in solids:
        cover += solid(pts, _BURY_MARGIN).astype(np.int32)
    keep = cover < 2
    pts, nrm, lab, wgt = pts[keep], nrm[keep], lab[keep], wgt[keep]

    # --- 逐部件 FPS，再交错成全局序列 ---
    present = np.array([int((lab == i).sum()) for i in range(len(parts))])
    if (present == 0).any():
        missing = [parts[i] for i in np.where(present == 0)[0]]
        raise RuntimeError(f"{obj}/{geom_tag} 的部件 {missing} 剔除埋点后一个候选都不剩")
    part_area = np.array([float(wgt[lab == i].sum()) for i in range(len(parts))])
    want = _quotas(part_area, n_points)
    want = np.minimum(want, present)         # 候选不够时不能凭空造点
    deficit = n_points - int(want.sum())
    while deficit > 0:                       # 缺口补给还有余量的部件
        room = present - want
        if room.max() <= 0:
            raise RuntimeError(f"{obj}/{geom_tag} 候选点不足 {n_points}，请调大 _OVERSAMPLE")
        i = int(np.argmax(room))
        add = min(deficit, int(room[i]))
        want[i] += add
        deficit -= add

    orders, sel_index = [], []
    for i in range(len(parts)):
        where = np.where(lab == i)[0]
        sub = pts[where]
        centroid = sub.mean(axis=0)
        start = int(np.argmax(((sub - centroid) ** 2).sum(axis=1)))
        orders.append(where[_fps_order(sub, int(want[i]), start)])
    order = _interleave(orders)

    sel_pts, sel_nrm, sel_lab = pts[order], nrm[order], lab[order]

    # --- 每个点代表多少面积：把候选点的面积摊给最近的入选点 ---
    owner = _nearest(pts, sel_pts)
    area = np.zeros(n_points, dtype=np.float64)
    np.add.at(area, owner, wgt)

    # --- 低分辨率的代表点：**只在同部件内部找** ---
    #
    # ⚠️ 纯几何最近邻会把薄物体的两个面并进同一个 cell。实测在 S5 用的 256 档上：
    # 黑板 **26.0%** 的表面点被判给了另一个部件的代表点（back <-> work_face 各 1500 点
    # 左右），旋钮 13.3%（rim_side <-> disc），抽屉 5.1%（含 bar_back <-> bar_bottom）。
    # 黑板厚 20 mm，而 256 档的粗粒 pitch 约 50 mm——比厚度还大，两个面必然混。
    #
    # 后果不是精度问题而是**语义错误**：擦拭的 effect 定义在工作面上，旋钮的反证
    # 恰恰是"销钉换成低摩擦轮缘"，抽屉唯一干净的 region 论证靠的是横杆背面 vs 正面。
    # D-58 早就为力散射定过"同部件"这条规矩，池化这一侧当时漏了。
    parent = {}
    for level in LEVELS:
        if level > n_points:
            continue
        mapping = np.empty(n_points, dtype=np.int64)
        for label in np.unique(sel_lab):
            rows = np.flatnonzero(sel_lab == label)
            heads = rows[rows < level]
            if not len(heads):
                raise ValueError(
                    f"{obj}/{geom_tag}: 部件 {parts[label]!r} 在 {level} 档里一个代表点都没有，"
                    "该档粗到无法表示这个部件；宁可报错也不要静默并进别的部件")
            mapping[rows] = heads[_nearest(sel_pts[rows], sel_pts[heads])]
        parent[level] = mapping.astype(np.int32)

    surface_hash = _surface_sha256(obj, geom_tag, sel_pts, sel_nrm, sel_lab)

    return Surface(
        obj=obj, geom_tag=geom_tag,
        points=sel_pts.astype(np.float32), normals=sel_nrm.astype(np.float32),
        part=sel_lab.astype(np.int8), parts=tuple(parts),
        area=area.astype(np.float32), parent=parent,
        sha256=surface_hash, solids=tuple(solids),
    )


_CACHE: dict[tuple[str, str, int], Surface] = {}


def surface_for(obj: str, geom_tag: str = "nominal",
                n_points: int | None = None) -> Surface:
    """取某个物体某个几何变体的冻结表面采样（带进程内缓存）。

    Args:
        obj: `_OBJECTS` 里的名字。擦拭任务的被操作物体是 ``"board"``
            （`plan/02` §1.1：是平面，不是黑板擦）。
        geom_tag: ``nominal`` / ``g1`` / ``g2``，取自 episode 的 ``geometry_variant``。
        n_points: 采样点数。默认 ``None`` = 按目标 pitch 自动换算（`_base_n`），
            物体越大点越多，`LEVELS` 那四档始终是它的前缀。
    """
    if obj not in _OBJECTS:
        raise KeyError(f"未知物体 {obj!r}，可选：{sorted(_OBJECTS)}")
    key = (obj, geom_tag, -1 if n_points is None else int(n_points))
    if key not in _CACHE:
        _CACHE[key] = _build(obj, geom_tag, n_points)
    return _CACHE[key]


def object_names() -> tuple[str, ...]:
    return tuple(sorted(_OBJECTS))


def assign_to_surface(contact_pos_obj: np.ndarray, surface: Surface,
                      max_dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把接触点归到最近的表面采样点。

    返回 ``(下标, 是否落在这个物体上, 到最近采样点的距离)``。

    "落在这个物体上"分两种情形，**必须分开判**：

    - 点在实体**内部** —— 那是求解器允许的穿透（撞击瞬态下实测球上中位 6 mm，
      P-22 说峰值力里本来就含撞击瞬态）。归属仍然成立，最近的表面点就是
      接触斑块所在处；
    - 点在实体**外部**且超过容差 —— 那才是归属错了：几何变体没对上，
      或者接触落在没建模的部件上。这类点不硬塞给最近点，单列成诊断量。

    Args:
        contact_pos_obj: (K, 3) 物体系接触点。
        max_dist: 外侧容差（m）。取采样 pitch 的量级即可。
    """
    if contact_pos_obj.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float64))
    d = ((contact_pos_obj[:, None, :] - surface.points[None, :, :]) ** 2).sum(axis=-1)
    idx = np.argmin(d, axis=1)
    dist = np.sqrt(d[np.arange(len(idx)), idx])
    ok = (dist <= max_dist) | surface.inside(contact_pos_obj)
    return idx, ok, dist
