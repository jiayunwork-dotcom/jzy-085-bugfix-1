"""单位制 / 网格细化不变性测试（问题复现用例的自动化守死）。

一副结构能不能解，只取决于结构自身，与用户选用的自洽单位制、
一根杆切成几段无关。具体容差（全部写死）：

- 同一副 4 跨 20 层刚架分别用 N·mm 与 kN·m 输入：两套结果按单位
  换算后，节点位移、杆端内力、支座反力的相对误差 ≤ 1e-6；
- 悬臂梁集中力工况细分 1 / 8 / 16 / 200 / 300 段、N·mm 与 N·m
  两套单位：端部挠度 PL³/(3EI)、转角 PL²/(2EI)、固端弯矩 PL 的
  相对误差 ≤ 1e-6；
- 旧实现的固定阈值（1e-10）会误判的高条件数合法矩阵（cond≈1e11）
  必须正常求解；全零 / 秩亏 / 零对角行的矩阵依旧报 SINGULAR_MATRIX。
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import by_id
from framesolver.analysis import analyze_frame
from framesolver.errors import SingularMatrixError
from framesolver.main import app
from framesolver.models import (
    DistributedLoad,
    FrameInput,
    Member,
    Node,
    NodalLoad,
)
from framesolver.solver import solve_system

# 全部断言用的具体容差
RTOL = 1.0e-6          # 题目要求：跨单位 / 对教材解的相对容差
N_SEGMENTS = (1, 8, 16, 200, 300)


# ---------------------------------------------------------------------------
# 模型构造
# ---------------------------------------------------------------------------

def make_multistory_frame(unit: str) -> FrameInput:
    """4 跨 20 层办公楼框架：底层 5 个柱脚三向固定，节点全部刚接。

    unit='Nmm' 与 'kNm' 两版是同一副物理结构的自洽单位换算。
    """
    n_bay, n_story = 4, 20
    if unit == "Nmm":
        span, story = 6000.0, 3600.0
        e = 2.06e5
        a_col, i_col = 250000.0, 500.0**4 / 12.0
        a_beam, i_beam = 180000.0, 300.0 * 600.0**3 / 12.0
        h_force, qy = 20000.0, -30.0
    elif unit == "kNm":
        span, story = 6.0, 3.6
        e = 2.06e8
        a_col, i_col = 0.25, 0.5**4 / 12.0
        a_beam, i_beam = 0.18, 0.3 * 0.6**3 / 12.0
        h_force, qy = 20.0, -30.0
    else:
        raise ValueError(unit)

    nodes = []
    for iy in range(n_story + 1):
        for ix in range(n_bay + 1):
            fixed = iy == 0
            nodes.append(
                Node(id=f"n{ix}_{iy}", x=ix * span, y=iy * story,
                     restraints=[fixed, fixed, fixed])
            )

    members = []
    # 柱
    for iy in range(n_story):
        for ix in range(n_bay + 1):
            members.append(Member(id=f"c{ix}_{iy}", node_i=f"n{ix}_{iy}",
                                  node_j=f"n{ix}_{iy + 1}", elastic_modulus=e,
                                  area=a_col, inertia=i_col))
    # 梁（每根梁满跨向下均布荷载）
    for iy in range(1, n_story + 1):
        for ix in range(n_bay):
            mid = f"b{ix}_{iy}"
            members.append(Member(id=mid, node_i=f"n{ix}_{iy}",
                                  node_j=f"n{ix + 1}_{iy}", elastic_modulus=e,
                                  area=a_beam, inertia=i_beam))
    nodal_loads = [NodalLoad(node_id=f"n0_{iy}", fx=h_force)
                   for iy in range(1, n_story + 1)]
    distributed_loads = []
    for iy in range(1, n_story + 1):
        for ix in range(n_bay):
            distributed_loads.append(DistributedLoad(member_id=f"b{ix}_{iy}", qy=qy))

    return FrameInput(nodes=nodes, members=members, nodal_loads=nodal_loads,
                      distributed_loads=distributed_loads)


def make_cantilever(unit: str, n_seg: int) -> tuple[FrameInput, dict]:
    """左端固定、右端向下集中力 P 的悬臂梁，均分为 n_seg 个单元。"""
    if unit == "Nmm":
        length, e, a, i_, p = 6000.0, 2.0e5, 1.0e4, 1.0e8, 10000.0
    elif unit == "Nm":
        length, e, a, i_, p = 6.0, 2.0e11, 1.0e-2, 1.0e-4, 10000.0
    else:
        raise ValueError(unit)
    h = length / n_seg
    nodes = [
        Node(id=f"n{k}", x=k * h, y=0.0,
             restraints=[True, True, True] if k == 0 else [False, False, False])
        for k in range(n_seg + 1)
    ]
    members = [
        Member(id=f"m{k}", node_i=f"n{k}", node_j=f"n{k + 1}",
               elastic_modulus=e, area=a, inertia=i_)
        for k in range(n_seg)
    ]
    frame = FrameInput(nodes=nodes, members=members,
                       nodal_loads=[NodalLoad(node_id=f"n{n_seg}", fy=-p)])
    return frame, dict(L=length, E=e, A=a, I=i_, P=p)


def _assert_group_close(values_mm_converted, values_ref, group_name):
    """逐分量断言 |a - b| ≤ RTOL·group_scale（整组的最大参考值做尺度）。

    用组尺度而不是逐项相对误差，避免恰好为零 / 接近零的分量除以零；
    对所有显著分量这与 1e-6 相对误差等价。
    """
    a = np.asarray(values_mm_converted, dtype=float)
    b = np.asarray(values_ref, dtype=float)
    assert a.shape == b.shape
    group_scale = float(np.max(np.abs(b)))
    if group_scale == 0.0:
        assert np.allclose(a, b, atol=0.0), f"{group_name} 参考全为零但结果非零"
        return
    diff = np.max(np.abs(a - b))
    assert diff <= RTOL * group_scale, (
        f"{group_name} 跨单位最大偏差 {diff:.3e} 超过 {RTOL:.0e}·组尺度 "
        f"{group_scale:.6g}"
    )


# ---------------------------------------------------------------------------
# 悬臂梁：手算解 + 网格细化不变性
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["Nmm", "Nm"])
@pytest.mark.parametrize("n_seg", N_SEGMENTS)
def test_cantilever_point_load_mesh_and_units(unit, n_seg):
    frame, par = make_cantilever(unit, n_seg)
    result = analyze_frame(frame)
    assert result["success"] is True

    ei = par["E"] * par["I"]
    v_exact = par["P"] * par["L"] ** 3 / (3.0 * ei)   # 向下
    th_exact = par["P"] * par["L"] ** 2 / (2.0 * ei)  # 顺时针
    m_exact = par["P"] * par["L"]                     # 固定端反力矩

    tip = result["displacements"][-1]
    base = result["reactions"][0]
    assert abs(abs(tip["uy"]) - v_exact) < RTOL * v_exact
    assert abs(abs(tip["theta"]) - th_exact) < RTOL * th_exact
    assert abs(base["moment"] - m_exact) < RTOL * m_exact
    assert abs(base["fy"] - par["P"]) < RTOL * par["P"]
    assert abs(base["fx"]) < RTOL * m_exact  # 无水平荷载，水平反力为零

    # 全程不许出现非有限值
    for item in result["displacements"]:
        assert np.all(np.isfinite([item["ux"], item["uy"], item["theta"]]))
    for m in result["member_forces"]:
        for end in ("end_i", "end_j"):
            assert np.all(np.isfinite([m[end]["axial"], m[end]["shear"], m[end]["moment"]]))
    for r in result["reactions"]:
        assert np.all(np.isfinite([r["fx"], r["fy"], r["moment"]]))


# ---------------------------------------------------------------------------
# 4 跨 20 层刚架：两套单位制都能解，结果互换后一致
# ---------------------------------------------------------------------------

def test_multistory_frame_solves_in_both_unit_systems():
    """复现：同一副刚架 N·mm 输入曾误报 SINGULAR_MATRIX，kN·m 可解。"""
    for unit in ("Nmm", "kNm"):
        result = analyze_frame(make_multistory_frame(unit))
        assert result["success"] is True, f"单位制 {unit} 不应再误报奇异"


def test_multistory_frame_reference_values():
    """与题目给出的核算结果对齐（kN·m 版）。"""
    result = analyze_frame(make_multistory_frame("kNm"))
    disp = by_id(result["displacements"])
    reac = by_id(result["reactions"])

    ux_top_left = disp["n0_20"]["ux"]
    moment_base_left = reac["n0_0"]["moment"]
    # 题面给的是六位 / 两位有效数字的近似值 0.010292 m、154.41 kN·m
    assert abs(ux_top_left - 0.010292) < 0.5e-6
    assert abs(moment_base_left - 154.41) < 0.005
    # 精确参考值（本次求解），容差仍按题目 1e-6
    assert abs(ux_top_left - 0.0102916963) < RTOL * 0.0102916963
    assert abs(moment_base_left - 154.412599) < RTOL * 154.412599


def test_multistory_frame_results_unit_invariant():
    """N·mm 结果换算到 kN·m 后，位移 / 杆端力 / 反力与 kN·m 版一致（≤1e-6）。"""
    mm = analyze_frame(make_multistory_frame("Nmm"))
    knm = analyze_frame(make_multistory_frame("kNm"))

    d_mm, d_knm = by_id(mm["displacements"]), by_id(knm["displacements"])
    # 线位移 mm→m（×1e-3），转角无量纲（×1）
    _assert_group_close(
        [d_mm[nid]["ux"] * 1e-3 for nid in d_knm],
        [d_knm[nid]["ux"] for nid in d_knm], "节点水平位移 ux",
    )
    _assert_group_close(
        [d_mm[nid]["uy"] * 1e-3 for nid in d_knm],
        [d_knm[nid]["uy"] for nid in d_knm], "节点竖向位移 uy",
    )
    _assert_group_close(
        [d_mm[nid]["theta"] for nid in d_knm],
        [d_knm[nid]["theta"] for nid in d_knm], "节点转角 theta",
    )

    f_mm = {m["member_id"]: m for m in mm["member_forces"]}
    f_knm = {m["member_id"]: m for m in knm["member_forces"]}
    assert set(f_mm) == set(f_knm)
    # 力 N→kN（×1e-3），弯矩 N·mm→kN·m（×1e-6）
    axial_mm, axial_knm, shear_mm, shear_knm, moment_mm, moment_knm = [], [], [], [], [], []
    for mid in f_knm:
        for end in ("end_i", "end_j"):
            axial_mm.append(f_mm[mid][end]["axial"] * 1e-3)
            axial_knm.append(f_knm[mid][end]["axial"])
            shear_mm.append(f_mm[mid][end]["shear"] * 1e-3)
            shear_knm.append(f_knm[mid][end]["shear"])
            moment_mm.append(f_mm[mid][end]["moment"] * 1e-6)
            moment_knm.append(f_knm[mid][end]["moment"])
    _assert_group_close(axial_mm, axial_knm, "杆端轴力")
    _assert_group_close(shear_mm, shear_knm, "杆端剪力")
    _assert_group_close(moment_mm, moment_knm, "杆端弯矩")
    _assert_group_close(
        [f_mm[mid]["axial_force"] * 1e-3 for mid in f_knm],
        [f_knm[mid]["axial_force"] for mid in f_knm], "轴力标量",
    )

    r_mm, r_knm = by_id(mm["reactions"]), by_id(knm["reactions"])
    _assert_group_close(
        [r_mm[nid]["fx"] * 1e-3 for nid in r_knm],
        [r_knm[nid]["fx"] for nid in r_knm], "支座水平反力",
    )
    _assert_group_close(
        [r_mm[nid]["fy"] * 1e-3 for nid in r_knm],
        [r_knm[nid]["fy"] for nid in r_knm], "支座竖向反力",
    )
    _assert_group_close(
        [r_mm[nid]["moment"] * 1e-6 for nid in r_knm],
        [r_knm[nid]["moment"] for nid in r_knm], "支座反力矩",
    )


def test_multistory_frame_nmm_http_success():
    """HTTP 端到端：N·mm 大模型必须返回 200 而不是 SINGULAR_MATRIX。"""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    payload = make_multistory_frame("Nmm").model_dump()
    resp = client.post("/solve", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    top_left = next(d for d in body["displacements"] if d["node_id"] == "n0_20")
    # N·mm 版顶层侧移约 10.2917 mm
    assert abs(top_left["ux"] - 10.2916963) < RTOL * 10.2916963


# ---------------------------------------------------------------------------
# 求解器层：合法病态必须放行，真奇异必须继续报错
# ---------------------------------------------------------------------------

def test_solve_system_accepts_legitimately_ill_conditioned():
    """条件数 1e11 的对称正定矩阵（对角矩阵，类似 N·mm 旧误报场景）必须可解。

    旧实现固定阈值 1e-10 会把 s_min/s_max=1e-11 的合法矩阵误判为奇异。
    """
    k = np.diag([1.0, 1.0e-5, 1.0e-11])  # cond = 1e11
    p = np.array([1.0, 2.0, 3.0])
    d = solve_system(k, p)
    assert np.all(np.isfinite(d))
    assert d == pytest.approx([1.0, 2.0e5, 3.0e11], rel=1e-9)


def test_solve_system_rejects_zero_stiffness_dof():
    """某自由度完全没有刚度（K 中一整行 / 列为零）：典型机构，必须报错。"""
    k = np.diag([1.0, 1.0, 0.0])
    with pytest.raises(SingularMatrixError) as exc:
        solve_system(k, np.array([1.0, 2.0, 0.0]))
    assert exc.value.code == "SINGULAR_MATRIX"


@pytest.mark.parametrize(
    "k,p",
    [
        (np.zeros((3, 3)), np.zeros(3)),                    # 全零矩阵
        (np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]]),
         np.array([1.0, 2.0, 3.0])),                        # 秩 2
    ],
)
def test_solve_system_still_detects_truly_singular(k, p):
    """真奇异矩阵继续报 SINGULAR_MATRIX（不允许为修误报而放过去）。"""
    with pytest.raises(SingularMatrixError) as exc:
        solve_system(k, p)
    assert exc.value.code == "SINGULAR_MATRIX"
    assert "奇异" in exc.value.message or "机构" in exc.value.message
