"""补偿精度（double-double）运算：用两个 float64 表示一个数。

动机：直接刚度法总刚的条件数随网格加密以 O(n⁴) 增长（300 段悬臂
均衡后仍达 4×10¹⁰）。此时 float64 组装 / 求解中逐元素的随机舍入
（~1e-16 相对）会被条件数放大到 1e-6 量级的解误差，且大小随单位制、
自由度排序等无关因素漂移——同一副结构换套自洽单位就可能算出
不可信的数。把**组装、均衡、残差**三个环节改用补偿精度（有效精度
~1e-32），配合 float64 Cholesky 因子做混合精度迭代精化，即可把解的
前向误差压回 1e-12 量级以下，且行为在任何平台上完全一致（不依赖
longdouble 等硬件相关类型）。

约定：一个 DD 数是一对 float64 ``(hi, lo)``，真值为 ``hi + lo``，
其中 ``|lo| <= ulp(hi)/2`` 量级。所有函数对 numpy 数组逐元素工作。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DD:
    """双 float64 补偿精度数组：真值 = hi + lo。"""

    hi: np.ndarray
    lo: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.hi.shape


def as_dd(a: np.ndarray) -> DD:
    """float64 数组原样提升为 DD（低阶部分为零）。"""
    a = np.asarray(a, dtype=float)
    return DD(a, np.zeros_like(a))


def dd_zeros(shape: tuple[int, ...]) -> DD:
    return DD(np.zeros(shape), np.zeros(shape))


def to_float(x: DD) -> np.ndarray:
    """舍入回 float64（误差 ~1e-16 相对，仅在输出前使用）。"""
    return x.hi + x.lo


def dd_neg(x: DD) -> DD:
    return DD(-x.hi, -x.lo)


def two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knuth TwoSum：s = fl(a+b)，e = 精确舍入误差（a+b = s+e 精确成立）。"""
    s = a + b
    b_virt = s - a
    a_virt = s - b_virt
    e = (a - a_virt) + (b - b_virt)
    return s, e


def two_prod(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dekker TwoProd（无 FMA）：p = fl(a·b)，e = 精确舍入误差。"""
    p = a * b
    splitter = 134217729.0  # 2^27 + 1，把 53 位尾数劈成两段
    ca = a * splitter
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = b * splitter
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def dd_add(x: DD, y: DD) -> DD:
    """DD 加法（误差 ~2⁻¹⁰⁶ 相对）。"""
    s, e = two_sum(x.hi, y.hi)
    lo = e + (x.lo + y.lo)
    hi, lo = two_sum(s, lo)  # 归一化
    return DD(hi, lo)


def dd_sub(x: DD, y: DD) -> DD:
    return dd_add(x, dd_neg(y))


def dd_mul(x: DD, y: DD) -> DD:
    """DD 乘法（误差 ~2⁻¹⁰⁶ 相对）。"""
    p, e = two_prod(x.hi, y.hi)
    lo = e + (x.hi * y.lo + x.lo * y.hi)
    hi, lo = two_sum(p, lo)
    return DD(hi, lo)


def dd_div(x: DD, y: DD) -> DD:
    """DD 除法：先 float64 试商，再用 DD 余数修正一次。"""
    q1 = x.hi / y.hi
    prod = dd_mul(as_dd(q1), y)  # q1·y 的 DD
    r = dd_sub(x, prod)
    q2 = r.hi / y.hi
    hi, lo = two_sum(q1, q2)
    return DD(hi, lo)


def dd_sqrt(x: DD) -> DD:
    """DD 平方根（x 必须为正）：牛顿法一步修正。"""
    s = np.sqrt(x.hi)
    sq = dd_mul(as_dd(s), as_dd(s))
    r = dd_sub(x, sq)
    corr = r.hi / (2.0 * s)
    hi, lo = two_sum(s, corr)
    return DD(hi, lo)


def dd_outer(d: DD) -> DD:
    """向量 d 的 DD 外积：结果[i, j] = d_i · d_j。"""
    n = d.hi.shape[0]
    x = DD(
        np.broadcast_to(d.hi[:, None], (n, n)),
        np.broadcast_to(d.lo[:, None], (n, n)),
    )
    y = DD(
        np.broadcast_to(d.hi[None, :], (n, n)),
        np.broadcast_to(d.lo[None, :], (n, n)),
    )
    return dd_mul(x, y)


def dd_matvec(k: DD, y: np.ndarray) -> DD:
    """DD 矩阵 × float64 向量的补偿精度乘积（Dot2，Ogita–Rump–Oishi）。

    逐元素 TwoProd 得到主乘积与舍入误差，再沿列方向做**配对 DD 归约**
    （log₂n 层、每层全部向量化），结果精度 ~1e-31 相对且与矩阵
    条件数无关；这是混合精度迭代精化中残差计算的核心。
    """
    p_hi, p_err = two_prod(k.hi, y[None, :])  # 主乘积与其舍入误差
    lo = p_err + k.lo * y[None, :]            # 各乘积的低阶部分
    hi = p_hi
    # 沿列方向两两配对的 DD 归约：每轮 (hi[:,2k], hi[:,2k+1]) 做 TwoSum，
    # 同时把对应的低阶项并入 lo
    while hi.shape[1] > 1:
        if hi.shape[1] % 2 == 1:
            hi = np.pad(hi, ((0, 0), (0, 1)))
            lo = np.pad(lo, ((0, 0), (0, 1)))
        hi_next, err = two_sum(hi[:, 0::2], hi[:, 1::2])
        lo = err + lo[:, 0::2] + lo[:, 1::2]
        hi = hi_next
    return DD(hi[:, 0].copy(), lo[:, 0].copy())
