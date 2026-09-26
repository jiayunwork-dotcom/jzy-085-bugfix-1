"""几何、方向余弦与坐标变换。

局部坐标定义：

- x̄ 轴沿杆件从 i 端指向 j 端；
- ȳ 轴为 x̄ 轴逆时针旋转 90°（右手平面）。

位移 / 力的变换（6 维向量，顺序为 [i 端 x、y、转角, j 端 x、y、转角]）：

    d_local = T @ d_global
    f_global = T.T @ f_local

其中 T = diag(R, R)，R 为平面旋转矩阵（转角在两个坐标系中数值相同）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ZeroLengthMemberError


@dataclass(frozen=True)
class FrameGeometry:
    """单根杆件的几何信息。"""

    length: float
    cos: float  # 方向余弦 c
    sin: float  # 方向正弦 s


def frame_geometry(xi: float, yi: float, xj: float, yj: float) -> FrameGeometry:
    """由两端整体坐标求杆长与方向余弦；杆长为零时拒绝计算。"""
    dx = xj - xi
    dy = yj - yi
    length = float(np.hypot(dx, dy))
    if length == 0.0:
        # 正常路径下 validation 已拦截，这里是最后一道防线
        raise ZeroLengthMemberError("杆件两端坐标完全重合，杆长为零")
    return FrameGeometry(length=length, cos=dx / length, sin=dy / length)


def rotation_block(c: float, s: float) -> np.ndarray:
    """单个节点的 3×3 旋转块（转角分量恒等不变）。"""
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def transformation_matrix(geom: FrameGeometry) -> np.ndarray:
    """六阶坐标变换矩阵 T，满足 d_local = T @ d_global。"""
    r = rotation_block(geom.cos, geom.sin)
    t = np.zeros((6, 6), dtype=float)
    t[0:3, 0:3] = r
    t[3:6, 3:6] = r
    return t


def to_local(global_vector: np.ndarray, t: np.ndarray) -> np.ndarray:
    """整体 -> 局部：d_local = T d_global。"""
    return t @ np.asarray(global_vector, dtype=float)


def stiffness_to_global(k_local: np.ndarray, t: np.ndarray) -> np.ndarray:
    """单元刚度从局部转到整体：K_g = T.T K_l T。"""
    return t.T @ k_local @ t


def force_to_global(f_local: np.ndarray, t: np.ndarray) -> np.ndarray:
    """单元等效节点力从局部转到整体：F_g = T.T F_l。"""
    return t.T @ np.asarray(f_local, dtype=float)
