"""力学平衡恒等式测试（自动化守死的硬性不变量）。

对含斜杆、混合支承、节点荷载与均布荷载的一般刚架，逐条验证：

1. 全部支座反力 + 全部外荷载，整体坐标下水平、竖向合力为零；
2. 对任一固定点的合力矩为零（取原点和另一个任意点分别检查）；
3. 同一根杆两端轴力等值反号（“杆对节点”约定下 i、j 端局部分量互为相反数）；
4. 每个刚性节点处，交汇各杆的杆端弯矩（转到整体即 z 向矩分量）
   与外加节点力矩、支座反力矩之和为零。

容差在 conftest.py 中写成具体数值：TOL_FORCE=1e-8、TOL_MOMENT=1e-8。
"""

from __future__ import annotations

import numpy as np

from conftest import (
    TOL_FORCE,
    TOL_MOMENT,
    by_id,
    make_mixed_frame,
    make_portal_frame,
    member_global_end_forces,
)
from framesolver.analysis import analyze_frame


def _external_global_force(frame, result):
    """外荷载 + 支座反力在整体坐标下对每个节点的三维 [fx, fy, m] 汇总。

    作用在杆件上的均布荷载合力计入全局平衡（作用点取杆中点，
    仅影响力矩；力本身并入全局合力）。
    """
    nodes = {n.id: n for n in frame.nodes}
    members = {m.id: m for m in frame.members}
    ext = {nid: np.zeros(3) for nid in nodes}
    for nl in frame.nodal_loads:
        ext[nl.node_id] += np.array([nl.fx, nl.fy, nl.moment])
    for dl in frame.distributed_loads:
        m = members[dl.member_id]
        ni, nj = nodes[m.node_i], nodes[m.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        length = float(np.hypot(dx, dy))
        c, s = dx / length, dy / length
        fx_total = dl.qy * length * (-s)
        fy_total = dl.qy * length * c
        # 无偏心分布荷载对自身中点无附加力矩；力附在中点节点的记账位置即可
        mid_nid = ni.id
        ext[mid_nid] += np.array([fx_total, fy_total, 0.0])
    for r in result["reactions"]:
        ext[r["node_id"]] += np.array([r["fx"], r["fy"], r["moment"]])
    return ext


def test_global_force_equilibrium():
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    ext = _external_global_force(frame, result)

    sum_fx = sum(v[0] for v in ext.values())
    sum_fy = sum(v[1] for v in ext.values())
    assert abs(sum_fx) < TOL_FORCE, f"ΣFx={sum_fx}"
    assert abs(sum_fy) < TOL_FORCE, f"ΣFy={sum_fy}"


def test_global_moment_equilibrium_about_any_point():
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    nodes = {n.id: n for n in frame.nodes}

    # 分布荷载合力（局部 y_bar 向）在整体坐标下的合力与合力矩
    dist_loads = []
    members = {m.id: m for m in frame.members}
    for dl in frame.distributed_loads:
        m = members[dl.member_id]
        ni, nj = nodes[m.node_i], nodes[m.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        length = float(np.hypot(dx, dy))
        c, s = dx / length, dy / length
        # 局部 y_bar 单位向量在整体坐标下为 (-s, c)
        fx_total = dl.qy * length * (-s)
        fy_total = dl.qy * length * c
        mx_mid = 0.5 * (ni.x + nj.x)
        my_mid = 0.5 * (ni.y + nj.y)
        dist_loads.append((fx_total, fy_total, mx_mid, my_mid))

    def total_moment_about(px: float, py: float) -> float:
        total = 0.0
        # 节点荷载 + 反力（矩为自由向量，绕任一点都一样）
        for nl in frame.nodal_loads:
            n = nodes[nl.node_id]
            total += nl.moment
            total += (n.x - px) * nl.fy - (n.y - py) * nl.fx
        for r in result["reactions"]:
            n = nodes[r["node_id"]]
            total += r["moment"]
            total += (n.x - px) * r["fy"] - (n.y - py) * r["fx"]
        for fx_t, fy_t, cx, cy in dist_loads:
            total += (cx - px) * fy_t - (cy - py) * fx_t
        return total

    # “任一固定点”：原点 + 两个任意点
    for px, py in [(0.0, 0.0), (3.17, -2.4), (-5.0, 8.6)]:
        msum = total_moment_about(px, py)
        assert abs(msum) < TOL_MOMENT, f"绕点({px},{py}) ΣM={msum}"


def test_member_axial_forces_equal_and_opposite():
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    for m in result["member_forces"]:
        ni_axial = m["end_i"]["axial"]
        nj_axial = m["end_j"]["axial"]
        assert abs(ni_axial + nj_axial) < TOL_FORCE, (
            f"杆件 {m['member_id']} 两端轴力不反向：{ni_axial} vs {nj_axial}"
        )
        # 轴力标量（拉正）应等于 i 端分量
        assert abs(m["axial_force"] - ni_axial) < TOL_FORCE

    # 门式刚架同样检查（无分布荷载，纯节点力情形）
    portal, _ = make_portal_frame()
    result_p = analyze_frame(portal)
    for m in result_p["member_forces"]:
        assert abs(m["end_i"]["axial"] + m["end_j"]["axial"]) < TOL_FORCE


def test_member_free_body_equilibrium():
    """杆件隔离体自平衡：节点对杆的端力 + 分布荷载合力 = 0（含力矩）。"""
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    nodes = {n.id: n for n in frame.nodes}
    members = {m.id: m for m in frame.members}
    qy_by_member = {dl.member_id: dl.qy for dl in frame.distributed_loads}

    for m in result["member_forces"]:
        model = members[m["member_id"]]
        ni, nj = nodes[model.node_i], nodes[model.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        length = float(np.hypot(dx, dy))
        c, s = dx / length, dy / length
        r = np.array([[c, s], [-s, c]])
        # 杆对节点 -> 节点对杆：取反号，再转整体
        fi = -(r.T @ np.array([m["end_i"]["axial"], m["end_i"]["shear"]]))
        fj = -(r.T @ np.array([m["end_j"]["axial"], m["end_j"]["shear"]]))
        qy = qy_by_member.get(m["member_id"], 0.0)
        qg = qy * length * np.array([-s, c])
        # 合力
        fsum = fi + fj + qg
        assert abs(fsum[0]) < TOL_FORCE, f"杆{m['member_id']} 隔离体ΣFx={fsum[0]}"
        assert abs(fsum[1]) < TOL_FORCE, f"杆{m['member_id']} 隔离体ΣFy={fsum[1]}"
        # 绕 i 端力矩：节点对杆的 i、j 端弯矩（杆对节点取反），j 端剪力力臂，荷载中点
        m_on_member_i = -m["end_i"]["moment"]
        m_on_member_j = -m["end_j"]["moment"]
        fj_local = -np.array([m["end_j"]["axial"], m["end_j"]["shear"]])
        moment_sum = (
            m_on_member_i
            + m_on_member_j
            + fj_local[1] * length
            + qy * length * (length / 2.0)
        )
        assert abs(moment_sum) < TOL_MOMENT, (
            f"杆{m['member_id']} 隔离体ΣM={moment_sum}"
        )


def test_joint_moment_equilibrium():
    """每个刚性节点：交汇各杆端弯矩 + 外加节点力矩 + 支座反力矩 = 0。

    平面问题中局部转角即整体转角，故直接对各杆在该节点的杆端矩分量求和。
    """
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    ext_moment = {n.id: 0.0 for n in frame.nodes}
    for nl in frame.nodal_loads:
        ext_moment[nl.node_id] += nl.moment
    for r in result["reactions"]:
        ext_moment[r["node_id"]] += r["moment"]

    member_moment_at = {n.id: 0.0 for n in frame.nodes}
    for m in result["member_forces"]:
        member_moment_at[m["node_i"]] += m["end_i"]["moment"]
        member_moment_at[m["node_j"]] += m["end_j"]["moment"]

    for nid in member_moment_at:
        total = member_moment_at[nid] + ext_moment[nid]
        assert abs(total) < TOL_MOMENT, f"节点 {nid} 弯矩不平衡：{total}"


def test_joint_force_equilibrium():
    """每个刚性节点的三维力平衡（水平、竖向、弯矩），含杆端力贡献。"""
    frame, _ = make_mixed_frame()
    result = analyze_frame(frame)
    nodes = {n.id: n for n in frame.nodes}

    balance = {nid: np.zeros(3) for nid in nodes}
    for nl in frame.nodal_loads:
        balance[nl.node_id] += [nl.fx, nl.fy, nl.moment]
    for r in result["reactions"]:
        balance[r["node_id"]] += [r["fx"], r["fy"], r["moment"]]

    for m in result["member_forces"]:
        fi, fj = member_global_end_forces(frame, result, m["member_id"])
        balance[m["node_i"]] += [fi[0], fi[1], m["end_i"]["moment"]]
        balance[m["node_j"]] += [fj[0], fj[1], m["end_j"]["moment"]]

    for nid, v in balance.items():
        assert abs(v[0]) < TOL_FORCE, f"节点{nid} ΣFx={v[0]}"
        assert abs(v[1]) < TOL_FORCE, f"节点{nid} ΣFy={v[1]}"
        assert abs(v[2]) < TOL_MOMENT, f"节点{nid} ΣM={v[2]}"
