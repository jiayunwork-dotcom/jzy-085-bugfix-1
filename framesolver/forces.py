"""内力与反力回代。

杆端力严格由本杆的刚度关系回代，不另起查表：

    f_local = P_eq_local - k_local @ (T @ d_global)

其物理意义统一约定为**杆件作用于节点**的局部坐标六维力
[i 端轴向, i 端横向, i 端弯矩, j 端轴向, j 端横向, j 端弯矩]，
正方向沿局部坐标轴、弯矩逆时针为正。这样：

- 节点平衡可直接用：外荷载 + 交汇各杆的杆端力 + 支座反力 = 0；
- 轴力标量取 i 端轴向分量，拉力为正（受拉时杆把 i 端节点往杆方向拽，
  沿 +x̄；j 端分量则等幅反号）；
- 两端固定梁受向下荷载时，i 端杆对节点的横向力向下、弯矩顺时针，
  与教材固端力的物理方向一致。

支座反力由被约束自由度对应的平衡行直接回代：

    R = (K @ d - P) 中受约束的分量

即“支座作用于结构（节点）”的力；自由自由度处反力恒为零。
"""

from __future__ import annotations

import numpy as np


def member_end_forces_local(
    k_local: np.ndarray,
    t: np.ndarray,
    d_global_member: np.ndarray,
    eq_local: np.ndarray,
) -> np.ndarray:
    """返回杆件作用于节点的局部坐标六维杆端力 f = P_eq - K_l T d_g。"""
    d_local = t @ d_global_member
    return np.asarray(eq_local, dtype=float) - k_local @ d_local


def reactions(
    stiffness: np.ndarray,
    displacement: np.ndarray,
    load: np.ndarray,
    restrained: np.ndarray,
) -> np.ndarray:
    """由全部平衡行回代支座反力（未约束自由度处为 0）。"""
    full = np.zeros(displacement.shape[0], dtype=float)
    if restrained.size:
        full[restrained] = stiffness[restrained, :] @ displacement - load[restrained]
    return full
