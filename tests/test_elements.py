"""单元层模块测试：方向余弦 / 坐标变换 / 单元刚度 / 组装。"""

from __future__ import annotations

import numpy as np
import pytest

from framesolver import assembly
from framesolver.element import local_stiffness, uniform_transverse_equivalent_loads
from framesolver.errors import ZeroLengthMemberError
from framesolver.geometry import (
    force_to_global,
    frame_geometry,
    stiffness_to_global,
    transformation_matrix,
)


def test_frame_geometry_direction_cosines():
    g = frame_geometry(0.0, 0.0, 3.0, 4.0)
    assert g.length == pytest.approx(5.0)
    assert g.cos == pytest.approx(0.6)
    assert g.sin == pytest.approx(0.8)


def test_frame_geometry_rejects_zero_length():
    with pytest.raises(ZeroLengthMemberError):
        frame_geometry(2.0, -1.0, 2.0, -1.0)


def test_transformation_is_orthogonal():
    g = frame_geometry(0.0, 0.0, 3.0, 4.0)
    t = transformation_matrix(g)
    assert np.allclose(t.T @ t, np.eye(6), atol=1e-12)
    # 水平杆：T 为单位阵
    t_h = transformation_matrix(frame_geometry(0, 0, 1, 0))
    assert np.allclose(t_h, np.eye(6))


def test_local_stiffness_symmetry_nullspace_and_rank():
    l = 4.0
    k = local_stiffness(l, 2.1e8, 1.0e-2, 1.0e-4)
    assert np.allclose(k, k.T, atol=1e-8)
    # 平面梁单元有 3 个精确零能刚体模态：
    rigid_modes = [
        np.array([1.0, 0, 0, 1, 0, 0]),          # 轴向平移
        np.array([0.0, 1, 0, 0, 1, 0]),          # 横向平移
        np.array([0.0, 0, 1, 0, l, 1]),          # 绕 i 端刚体转动（j 端横移 +Lθ）
    ]
    for v in rigid_modes:
        assert np.allclose(k @ v, 0.0, atol=1e-4), f"刚体模态非零能：{v}"
    # 6 阶矩阵、零空间维数 3 => 秩 3
    assert int(np.linalg.matrix_rank(np.round(k, 6))) == 3


def test_axial_stiffness_entry():
    l, e, a = 5.0, 2.0e7, 3.0e-2
    k = local_stiffness(l, e, a, 1.0e-6)
    assert k[0, 0] == pytest.approx(e * a / l)
    assert k[3, 3] == pytest.approx(e * a / l)
    assert k[0, 3] == pytest.approx(-e * a / l)


def test_bending_stiffness_entries():
    l, e, i = 4.0, 2.1e8, 8.0e-3
    k = local_stiffness(l, e, 1.0, i)
    ei = e * i
    assert k[1, 1] == pytest.approx(12 * ei / l ** 3)
    assert k[2, 2] == pytest.approx(4 * ei / l)
    assert k[5, 5] == pytest.approx(4 * ei / l)
    assert k[2, 5] == pytest.approx(2 * ei / l)
    assert k[1, 2] == pytest.approx(6 * ei / l ** 2)


def test_equivalent_uniform_loads_resultants():
    l, q = 4.0, -3.0  # 向下 3 kN/m
    p = uniform_transverse_equivalent_loads(l, q)
    # 两个竖向等效节点力之和等于合力 qL
    assert p[1] + p[4] == pytest.approx(q * l, abs=1e-12)
    # 对 i 端取矩：P4*L + M_i + M_j = 均布荷载对 i 端的矩 qL·L/2
    moment_about_i = p[4] * l + p[2] + p[5]
    assert moment_about_i == pytest.approx(q * l * l / 2, abs=1e-12)
    # 固端弯矩大小 qL²/12，i 端正、j 端负（q 沿 +y 为正时）
    p_pos = uniform_transverse_equivalent_loads(l, 3.0)
    assert p_pos[2] == pytest.approx(3.0 * l ** 2 / 12)
    assert p_pos[5] == pytest.approx(-3.0 * l ** 2 / 12)


def test_global_stiffness_rotation_invariant_in_eigenvalues():
    g = frame_geometry(0.0, 0.0, 3.0, 4.0)
    k = local_stiffness(5.0, 2.1e8, 1.0e-2, 8.0e-4)
    t = transformation_matrix(g)
    kg = stiffness_to_global(k, t)
    assert np.allclose(kg, kg.T, atol=1e-6)
    # 相似变换（T T^T=I）下非零特征值不变
    eig_local = np.linalg.eigvalsh(k)
    eig_global = np.linalg.eigvalsh(kg)
    assert np.allclose(np.sort(eig_local), np.sort(eig_global), rtol=1e-9)


def test_force_vector_rotation():
    g = frame_geometry(0.0, 0.0, 0.0, 1.0)  # 竖直杆 c=0,s=1
    t = transformation_matrix(g)
    # 局部 +x（竖直向上）的力 -> 整体 +y
    fg = force_to_global(np.array([1.0, 0, 0, 0, 0, 0]), t)
    assert fg[1] == pytest.approx(1.0)
    # 局部 +y（水平向左）-> 整体 -x
    fg2 = force_to_global(np.array([0, 1.0, 0, 0, 0, 0]), t)
    assert fg2[0] == pytest.approx(-1.0)


def test_assembly_scatter_add_two_elements():
    from framesolver.models import Member, Node

    nodes = [Node(id="a", x=0, y=0, restraints=[False] * 3),
             Node(id="b", x=1, y=0, restraints=[False] * 3),
             Node(id="c", x=2, y=0, restraints=[False] * 3)]
    members = [
        Member(id="m1", node_i="a", node_j="b", elastic_modulus=1, area=1, inertia=1),
        Member(id="m2", node_i="b", node_j="c", elastic_modulus=1, area=1, inertia=1),
    ]
    k1 = np.eye(6)
    k2 = 2 * np.eye(6)
    eq1 = np.ones(6)
    eq2 = np.ones(6)
    index = {"a": 0, "b": 1, "c": 2}
    k, p = assembly.assemble(9, members, index, [k1, k2], [eq1, eq2], [])
    # 中间节点自由度同时收到两根杆的贡献
    assert np.allclose(np.diag(k)[3:6], 3.0)
    assert np.allclose(p[3:6], 2.0)
    assert np.allclose(np.diag(k)[0:3], 1.0)
    assert np.allclose(np.diag(k)[6:9], 2.0)
