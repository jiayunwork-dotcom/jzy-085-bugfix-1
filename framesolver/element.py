"""单元刚度生成（局部坐标）与分布荷载的等效节点力。

六阶局部单元刚度矩阵同时包含轴向变形与弯曲变形，
自由度顺序：[i 端轴向, i 端横向, i 端转角, j 端轴向, j 端横向, j 端转角]。

记号：a = EA/L，b = 12EI/L³，c = 6EI/L²，d = 4EI/L，e = 2EI/L

    [ a   0   0  -a   0   0]
    [ 0   b   c   0  -b   c]
K = [ 0   c   d   0  -c   e]
    [-a   0   0   a   0   0]
    [ 0  -b  -c   0   b  -c]
    [ 0   c   e   0  -c   d]

满跨横向均布荷载 q（沿局部 +ȳ）的等效节点力由 Hermite 形函数积分得到，
采用 v 沿 +ȳ、θ = dv/dx（逆时针为正）的梁约定：

    Qθi = q ∫₀ᴸ x(1−x/L)² dx = +qL²/12
    Qθj = q ∫₀ᴸ [-x²/L + x³/L²] dx = −qL²/12

    P_eq = [0, qL/2, +qL²/12, 0, qL/2, −qL²/12]

对向下（q<0）荷载，i 端固端弯矩项为负（顺时针），与教材固端力一致。
注意 P_eq 是“代替分布荷载施加在节点上的等效外力”；杆端内力由
forces.member_end_forces_local 按 f = P_eq − K_l·d_l 恢复（杆件作用于节点）。
"""

from __future__ import annotations

import numpy as np


def local_stiffness(length: float, e: float, a: float, inertia: float) -> np.ndarray:
    """形成 6×6 局部坐标系单元刚度矩阵。

    参数
    ----
    length: 杆长 L
    e:      弹性模量 E
    a:      截面面积 A
    inertia: 截面惯性矩 I
    """
    l = float(length)
    l2 = l * l
    l3 = l2 * l

    ea_l = e * a / l          # 轴向刚度系数
    ei = e * inertia
    b = 12.0 * ei / l3        # 横向刚度系数
    c = 6.0 * ei / l2
    d = 4.0 * ei / l
    g = 2.0 * ei / l          # 远端转动刚度

    return np.array(
        [
            [ea_l, 0.0, 0.0, -ea_l, 0.0, 0.0],
            [0.0, b, c, 0.0, -b, c],
            [0.0, c, d, 0.0, -c, g],
            [-ea_l, 0.0, 0.0, ea_l, 0.0, 0.0],
            [0.0, -b, -c, 0.0, b, -c],
            [0.0, c, g, 0.0, -c, d],
        ],
        dtype=float,
    )


def uniform_transverse_equivalent_loads(length: float, qy: float) -> np.ndarray:
    """满跨局部 +ȳ 方向均布荷载的单元等效节点力（6 维，局部坐标）。"""
    l = float(length)
    q = float(qy)
    return np.array(
        [
            0.0,
            q * l / 2.0,
            q * l * l / 12.0,
            0.0,
            q * l / 2.0,
            -q * l * l / 12.0,
        ],
        dtype=float,
    )
