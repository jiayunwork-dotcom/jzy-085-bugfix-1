"""复现测试：求解结果必须与单位制选择、网格切分无关。

场景（来自实际使用反馈）：

1. 4 跨 20 层刚架：跨 6000、层高 3600，柱 500×500、梁 300×600，
   E = 2.06e5 N/mm²，每层最左节点 +x 向 20000 N，每根梁满跨
   qy = -30 N/mm。同一副框架用 N·mm 与 kN·m 两套自洽单位输入，
   都必须返回 success=true，且换算后位移 / 杆端内力 / 支座反力
   相对误差 ≤ 1e-6。
2. 单根悬臂：长 6000（或 6），E = 2.0e5（或 2.0e11），A = 1.0e4
   （或 1.0e-2），I = 1.0e8（或 1.0e-4），端部向下集中力 10000 N。
   分 1 / 8 / 16 / 200 / 300 段，端部挠度与转角都必须落在手算
   精确解 PL³/3EI、PL²/2EI 上，相对误差 ≤ 1e-6。

容差全部写成具体数值（TOL_REL = 1e-6，与需求一致）。
"""

from __future__ import annotations

import pytest

from conftest import by_id
from framesolver.analysis import analyze_frame
from framesolver.models import (
    DistributedLoad,
    FrameInput,
    Member,
    Node,
    NodalLoad,
)

# 需求给定的相对容差，写死具体数值
TOL_REL = 1.0e-6
# 与题述“约”值（4~5 位有效数字）比较用的容差
TOL_ANCHOR_REL = 1.0e-4

# ---------------------------------------------------------------------------
# 悬臂：两套自洽单位（力都是 N，长度 mm 或 m）
# ---------------------------------------------------------------------------
CANTILEVER_LOAD = 1.0e4  # N
CANTILEVER_UNITS = {
    "N·mm": dict(length=6000.0, e=2.0e5, area=1.0e4, inertia=1.0e8),
    "N·m": dict(length=6.0, e=2.0e11, area=1.0e-2, inertia=1.0e-4),
}
SEGMENT_COUNTS = [1, 8, 16, 200, 300]


def _make_cantilever(n_seg: int, length: float, e: float, area: float, inertia: float):
    """左端固定、右端向下集中力的悬臂，杆件从左到右等分 n_seg 段。"""
    h = length / n_seg
    nodes = [Node(id="0", x=0.0, y=0.0, restraints=[True, True, True])]
    nodes += [
        Node(id=str(k), x=k * h, y=0.0, restraints=[False, False, False])
        for k in range(1, n_seg + 1)
    ]
    members = [
        Member(id=f"e{k}", node_i=str(k), node_j=str(k + 1),
               elastic_modulus=e, area=area, inertia=inertia)
        for k in range(n_seg)
    ]
    return FrameInput(
        nodes=nodes,
        members=members,
        nodal_loads=[NodalLoad(node_id=str(n_seg), fx=0.0, fy=-CANTILEVER_LOAD, moment=0.0)],
    )


def _cantilever_exact(units: str) -> tuple[float, float]:
    """手算精确解：端部挠度 PL³/3EI（向下为负）、端部转角 PL²/2EI（顺时针为负）。"""
    p = CANTILEVER_LOAD
    u = CANTILEVER_UNITS[units]
    ei = u["e"] * u["inertia"]
    return -p * u["length"] ** 3 / (3.0 * ei), -p * u["length"] ** 2 / (2.0 * ei)


@pytest.mark.parametrize("n_seg", SEGMENT_COUNTS)
@pytest.mark.parametrize("units", sorted(CANTILEVER_UNITS))
def test_cantilever_tip_matches_hand_calculation(n_seg, units):
    """分 1/8/16/200/300 段、两套单位，端部挠度与转角都落在手算值上。"""
    frame = _make_cantilever(n_seg, **CANTILEVER_UNITS[units])
    result = analyze_frame(frame)
    assert result["success"] is True

    v_exact, theta_exact = _cantilever_exact(units)
    tip = by_id(result["displacements"])[str(n_seg)]
    assert abs(tip["uy"] - v_exact) <= TOL_REL * abs(v_exact), (
        f"{units} {n_seg} 段：挠度 {tip['uy']} vs 手算 {v_exact}"
    )
    assert abs(tip["theta"] - theta_exact) <= TOL_REL * abs(theta_exact), (
        f"{units} {n_seg} 段：转角 {tip['theta']} vs 手算 {theta_exact}"
    )


@pytest.mark.parametrize("n_seg", SEGMENT_COUNTS)
def test_cantilever_identical_across_unit_systems(n_seg):
    """同一根悬臂用 N·mm 与 N·m 求解，换算后结果相对误差 ≤ 1e-6。"""
    r_mm = analyze_frame(_make_cantilever(n_seg, **CANTILEVER_UNITS["N·mm"]))
    r_m = analyze_frame(_make_cantilever(n_seg, **CANTILEVER_UNITS["N·m"]))

    # 位移 mm→m 除以 1000；转角无量纲不变
    disp_mm = by_id(r_mm["displacements"])
    disp_m = by_id(r_m["displacements"])
    v_exact, theta_exact = _cantilever_exact("N·m")
    for nid in disp_m:
        assert abs(disp_mm[nid]["uy"] / 1000.0 - disp_m[nid]["uy"]) <= TOL_REL * abs(v_exact)
        assert abs(disp_mm[nid]["theta"] - disp_m[nid]["theta"]) <= TOL_REL * abs(theta_exact)

    # 力都是 N 不变；弯矩 N·mm→N·m 除以 1000
    forces_mm = by_id(r_mm["member_forces"], key="member_id")
    forces_m = by_id(r_m["member_forces"], key="member_id")
    for mid in forces_m:
        for end in ("end_i", "end_j"):
            f_mm, f_m = forces_mm[mid][end], forces_m[mid][end]
            assert abs(f_mm["shear"] - f_m["shear"]) <= TOL_REL * CANTILEVER_LOAD
            assert abs(f_mm["moment"] / 1000.0 - f_m["moment"]) <= (
                TOL_REL * CANTILEVER_LOAD * CANTILEVER_UNITS["N·m"]["length"]
            )

    # 支座反力：竖向反力 = P，反力矩 = P·L
    react_mm = by_id(r_mm["reactions"])["0"]
    react_m = by_id(r_m["reactions"])["0"]
    assert abs(react_mm["fy"] - react_m["fy"]) <= TOL_REL * CANTILEVER_LOAD
    assert abs(react_mm["moment"] / 1000.0 - react_m["moment"]) <= (
        TOL_REL * CANTILEVER_LOAD * CANTILEVER_UNITS["N·m"]["length"]
    )


# ---------------------------------------------------------------------------
# 4 跨 20 层刚架：N·mm 与 kN·m 两套自洽单位
# ---------------------------------------------------------------------------
N_BAYS, N_FLOORS = 4, 20
# 题述锚点（kN·m 单位制）：顶层最左节点水平位移、左下角柱脚反力矩
ANCHOR_TOP_UX_M = 0.010292
ANCHOR_BASE_MOMENT_KNM = 154.41


def _make_tower(units: str):
    """4 跨 20 层办公刚架。units="mm" 用 N·mm，units="m" 用 kN·m。"""
    if units == "mm":
        bay, story = 6000.0, 3600.0
        e = 2.06e5                       # N/mm²
        a_col, i_col = 250000.0, 500.0 ** 4 / 12.0
        a_beam, i_beam = 180000.0, 300.0 * 600.0 ** 3 / 12.0
        lateral, qy = 20000.0, -30.0     # N、N/mm
    else:
        bay, story = 6.0, 3.6
        e = 2.06e8                       # kN/m²
        a_col, i_col = 0.25, 0.5 ** 4 / 12.0
        a_beam, i_beam = 0.18, 0.3 * 0.6 ** 3 / 12.0
        lateral, qy = 20.0, -30.0        # kN、kN/m

    nid = lambda c, f: f"n{f}_{c}"
    nodes, members, dist_loads, nodal_loads = [], [], [], []
    for f in range(N_FLOORS + 1):
        for c in range(N_BAYS + 1):
            fixed = f == 0
            nodes.append(Node(id=nid(c, f), x=c * bay, y=f * story,
                              restraints=[fixed, fixed, fixed]))
    k = 0
    for f in range(1, N_FLOORS + 1):
        for c in range(N_BAYS + 1):
            members.append(Member(id=f"col{k}", node_i=nid(c, f - 1), node_j=nid(c, f),
                                  elastic_modulus=e, area=a_col, inertia=i_col))
            k += 1
    k = 0
    for f in range(1, N_FLOORS + 1):
        for c in range(N_BAYS):
            mid = f"bm{k}"
            members.append(Member(id=mid, node_i=nid(c, f), node_j=nid(c + 1, f),
                                  elastic_modulus=e, area=a_beam, inertia=i_beam))
            dist_loads.append(DistributedLoad(member_id=mid, load_type="uniform_transverse", qy=qy))
            k += 1
    for f in range(1, N_FLOORS + 1):
        nodal_loads.append(NodalLoad(node_id=nid(0, f), fx=lateral, fy=0.0, moment=0.0))
    return FrameInput(nodes=nodes, members=members,
                      nodal_loads=nodal_loads, distributed_loads=dist_loads)


def _convert_mm_to_knm(result: dict) -> dict:
    """N·mm 结果换算到 kN·m：位移 /1000，转角不变，力 /1000，弯矩 /1e6。"""
    out = {"displacements": [], "member_forces": [], "reactions": []}
    for d in result["displacements"]:
        out["displacements"].append({
            "node_id": d["node_id"],
            "ux": d["ux"] / 1000.0, "uy": d["uy"] / 1000.0, "theta": d["theta"],
        })
    for m in result["member_forces"]:
        out["member_forces"].append({
            "member_id": m["member_id"],
            "end_i": {"axial": m["end_i"]["axial"] / 1000.0,
                      "shear": m["end_i"]["shear"] / 1000.0,
                      "moment": m["end_i"]["moment"] / 1.0e6},
            "end_j": {"axial": m["end_j"]["axial"] / 1000.0,
                      "shear": m["end_j"]["shear"] / 1000.0,
                      "moment": m["end_j"]["moment"] / 1.0e6},
            "axial_force": m["axial_force"] / 1000.0,
        })
    for r in result["reactions"]:
        out["reactions"].append({
            "node_id": r["node_id"], "fx": r["fx"] / 1000.0, "fy": r["fy"] / 1000.0,
            "moment": r["moment"] / 1.0e6, "restrained": r["restrained"],
        })
    return out


def _assert_close(a: float, b: float, scale: float, label: str) -> None:
    """相对误差 ≤ TOL_REL；近零分量以该类物理量的最大量级为基准。"""
    assert abs(a - b) <= TOL_REL * max(abs(b), scale), f"{label}: {a} vs {b}（量级 {scale}）"


def test_tower_solves_in_both_unit_systems():
    """同一副 20 层刚架，N·mm 与 kN·m 都必须解出，且对上题述锚点。"""
    for units in ("mm", "m"):
        result = analyze_frame(_make_tower(units))
        assert result["success"] is True
        disp = by_id(result["displacements"])
        reactions = by_id(result["reactions"])
        if units == "mm":
            top_ux = disp["n20_0"]["ux"] / 1000.0
            base_moment = reactions["n0_0"]["moment"] / 1.0e6
        else:
            top_ux = disp["n20_0"]["ux"]
            base_moment = reactions["n0_0"]["moment"]
        assert abs(top_ux - ANCHOR_TOP_UX_M) <= TOL_ANCHOR_REL * ANCHOR_TOP_UX_M, (
            f"{units}: 顶层 ux={top_ux}，锚点 {ANCHOR_TOP_UX_M}"
        )
        assert abs(base_moment - ANCHOR_BASE_MOMENT_KNM) <= (
            TOL_ANCHOR_REL * ANCHOR_BASE_MOMENT_KNM
        ), f"{units}: 柱脚 M={base_moment}，锚点 {ANCHOR_BASE_MOMENT_KNM}"


def test_tower_results_identical_across_unit_systems():
    """20 层刚架两套单位的全部结果，换算后相对误差 ≤ 1e-6。"""
    r_mm = analyze_frame(_make_tower("mm"))
    r_m = analyze_frame(_make_tower("m"))
    conv = _convert_mm_to_knm(r_mm)

    # 位移与转角
    disp_conv = by_id(conv["displacements"])
    disp_m = by_id(r_m["displacements"])
    u_scale = max(abs(d["ux"]) for d in r_m["displacements"])
    th_scale = max(abs(d["theta"]) for d in r_m["displacements"])
    for nid in disp_m:
        _assert_close(disp_conv[nid]["ux"], disp_m[nid]["ux"], u_scale, f"{nid}.ux")
        _assert_close(disp_conv[nid]["uy"], disp_m[nid]["uy"], u_scale, f"{nid}.uy")
        _assert_close(disp_conv[nid]["theta"], disp_m[nid]["theta"], th_scale, f"{nid}.theta")

    # 杆端内力（轴力、剪力量纲 kN；弯矩量纲 kN·m）
    forces_conv = by_id(conv["member_forces"], key="member_id")
    forces_m = by_id(r_m["member_forces"], key="member_id")
    f_scale = max(
        abs(m[end][comp]) for m in r_m["member_forces"]
        for end in ("end_i", "end_j") for comp in ("axial", "shear")
    )
    m_scale = max(
        abs(m[end]["moment"]) for m in r_m["member_forces"] for end in ("end_i", "end_j")
    )
    for mid in forces_m:
        for end in ("end_i", "end_j"):
            fc, fm = forces_conv[mid][end], forces_m[mid][end]
            _assert_close(fc["axial"], fm["axial"], f_scale, f"{mid}.{end}.axial")
            _assert_close(fc["shear"], fm["shear"], f_scale, f"{mid}.{end}.shear")
            _assert_close(fc["moment"], fm["moment"], m_scale, f"{mid}.{end}.moment")
        _assert_close(forces_conv[mid]["axial_force"], forces_m[mid]["axial_force"],
                      f_scale, f"{mid}.axial_force")

    # 支座反力
    react_conv = by_id(conv["reactions"])
    react_m = by_id(r_m["reactions"])
    rf_scale = max(max(abs(r["fx"]), abs(r["fy"])) for r in r_m["reactions"])
    rm_scale = max(abs(r["moment"]) for r in r_m["reactions"])
    for nid in react_m:
        _assert_close(react_conv[nid]["fx"], react_m[nid]["fx"], rf_scale, f"{nid}.fx")
        _assert_close(react_conv[nid]["fy"], react_m[nid]["fy"], rf_scale, f"{nid}.fy")
        _assert_close(react_conv[nid]["moment"], react_m[nid]["moment"], rm_scale, f"{nid}.M")
