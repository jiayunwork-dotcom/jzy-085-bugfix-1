"""补偿精度（double-double）运算的单元测试。

用 fractions.Fraction 做任意精度参考：每个 float64 都是精确的二进制
有理数，DD 数 (hi, lo) 的真值为 Fraction(hi) + Fraction(lo)。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from framesolver.precision import (
    DD,
    as_dd,
    dd_add,
    dd_div,
    dd_matvec,
    dd_mul,
    dd_sqrt,
    dd_sub,
    to_float,
    two_prod,
    two_sum,
)

rng = np.random.default_rng(20260926)


def _exact(x: DD) -> list[Fraction]:
    """DD 数组的精确有理数值列表。"""
    return [Fraction(h) + Fraction(lo) for h, lo in
            zip(np.atleast_1d(x.hi), np.atleast_1d(x.lo))]


def _rel_err(x: DD, ref: Fraction) -> float:
    (val,) = _exact(x)
    return float(abs(val - ref) / abs(ref))


def test_two_sum_recovers_exact_sum():
    """Knuth TwoSum：s = fl(a+b)，且 s + e 精确等于 a + b。"""
    a = rng.uniform(-1e6, 1e6, 500)
    b = rng.uniform(-1e-6, 1e-6, 500)  # 数量级悬殊，舍入误差非平凡
    s, e = two_sum(a, b)
    assert np.array_equal(s, a + b)
    for ai, bi, si, ei in zip(a, b, s, e):
        assert Fraction(ai) + Fraction(bi) == Fraction(si) + Fraction(ei)


def test_two_prod_recovers_exact_product():
    """Dekker TwoProd：p = fl(a·b)，且 p + e 精确等于 a · b。"""
    a = rng.uniform(0.5, 2.0, 500)
    b = rng.uniform(0.5, 2.0, 500)
    p, e = two_prod(a, b)
    assert np.array_equal(p, a * b)
    for ai, bi, pi, ei in zip(a, b, p, e):
        assert Fraction(ai) * Fraction(bi) == Fraction(pi) + Fraction(ei)


def test_dd_add_and_sub_track_fraction_reference():
    x = DD(rng.uniform(1.0, 2.0, 8), rng.uniform(-1e-17, 1e-17, 8))
    y = DD(rng.uniform(1.0, 2.0, 8), rng.uniform(-1e-17, 1e-17, 8))
    for k in range(8):
        xk = DD(x.hi[k:k+1], x.lo[k:k+1])
        yk = DD(y.hi[k:k+1], y.lo[k:k+1])
        (xe,), (ye,) = _exact(xk), _exact(yk)
        assert _rel_err(dd_add(xk, yk), xe + ye) < 1e-30
        assert _rel_err(dd_sub(xk, yk), xe - ye) < 1e-30


def test_dd_mul_and_div_track_fraction_reference():
    a = DD(np.array([1.0 + 1e-13]), np.array([3e-17]))
    b = DD(np.array([2.0 - 1e-13]), np.array([-5e-17]))
    (ae,), (be,) = _exact(a), _exact(b)
    assert _rel_err(dd_mul(a, b), ae * be) < 1e-30
    assert _rel_err(dd_div(a, b), ae / be) < 1e-30


def test_dd_sqrt_squares_back():
    x = DD(np.array([2.0]), np.array([1e-17]))
    y = dd_sqrt(x)
    # y 的 DD 平方应回到 x（相对误差 ~1e-32）
    assert _rel_err(dd_mul(y, y), _exact(x)[0]) < 1e-30
    # 且 y 自身接近 sqrt(2 + 1e-17)
    assert abs(to_float(y)[0] - np.sqrt(2.0 + 1e-17)) < 1e-15


def test_dd_matvec_matches_exact_dot():
    """DD 矩阵 × float64 向量：与精确点积比较，相对误差 ~1e-31。"""
    n = 64
    k = DD(rng.uniform(0.5, 2.0, (n, n)), rng.uniform(-1e-17, 1e-17, (n, n)))
    y = rng.uniform(-1.0, 1.0, n)
    result = dd_matvec(k, y)
    exact = _exact(result)
    for i in range(n):
        ref = sum(
            (Fraction(k.hi[i, j]) + Fraction(k.lo[i, j])) * Fraction(y[j])
            for j in range(n)
        )
        assert float(abs(exact[i] - ref) / max(abs(ref), Fraction(1))) < 1e-28


def test_as_dd_roundtrip():
    a = rng.uniform(-1.0, 1.0, 10)
    x = as_dd(a)
    assert np.array_equal(x.hi, a)
    assert np.all(x.lo == 0.0)
    assert np.array_equal(to_float(x), a)
