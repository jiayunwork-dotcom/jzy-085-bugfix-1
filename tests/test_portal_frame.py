"""门式刚架教材算例 + 梁的经典解手算校核。

门式刚架（两柱底固定，柱、梁等长 L=4m，等截面 EI，轴向刚度 EA 取很大
以使轴向变形可忽略），横梁高度处受水平集中力 P=10 kN。

按位移法（忽略轴向变形）的闭式解（结构力学标准结果）：

    两个独立位移未知量：层间侧移 Δ、节点转角 θ2=θ3=θ

    柱：i = EI/L，一根柱抗侧刚度 12i/L²；节点弯矩平衡给出 θ = -PL²/(28EI)
    层间剪力平分：柱剪力 V = P/2

    柱底弯矩 M_base = 6i|θ|/... → 2PL/7
    柱顶弯矩（梁端）3PL/14，横梁剪力为零、弯矩为常数 3PL/14（上侧受拉）
    水平反力各 -P/2；竖向反力 ±3P/7（由整体取矩）

数值（P=10, L=4）：
    |R_x| = 5；|R_y| = 30/7 ≈ 4.2857；柱底 80/7 ≈ 11.4286；
    梁端弯矩 60/7 ≈ 8.5714，横梁跨中标量弯矩（下侧受拉为正）= -3PL/14。
"""

from __future__ import annotations

import numpy as np

from conftest import (
    TOL_DISP,
    TOL_FORCE,
    TOL_MOMENT,
    TOL_THEORY,
    by_id,
    make_beam,
    make_portal_frame,
    section_sagging_moment,
)
from framesolver.analysis import analyze_frame


def test_portal_reactions_match_textbook():
    frame, p = make_portal_frame()
    result = analyze_frame(frame)
    assert result["success"] is True

    reactions = by_id(result["reactions"])
    p_val, l = p["p"], p["l"]
    # 下列教材值由“忽略轴向变形”的位移法闭式解给出；本模型 A 有限，
    # 用相对容差 TOL_THEORY（0.2%）比较，量级与符号必须对上
    # 水平反力：两柱各承担 P/2，方向与荷载相反
    assert abs(reactions["1"]["fx"] - (-p_val / 2)) < TOL_THEORY * p_val
    assert abs(reactions["4"]["fx"] - (-p_val / 2)) < TOL_THEORY * p_val
    # 竖向反力（教材）：左 -3P/7，右 +3P/7（整体对柱底取矩）
    assert abs(reactions["1"]["fy"] - (-3 * p_val / 7)) < TOL_THEORY * p_val
    assert abs(reactions["4"]["fy"] - 3 * p_val / 7) < TOL_THEORY * p_val
    # 柱底弯矩：2PL/7，支座对结构为逆时针
    m_base = 2 * p_val * l / 7
    assert abs(reactions["1"]["moment"] - m_base) < TOL_THEORY * m_base
    assert abs(reactions["4"]["moment"] - m_base) < TOL_THEORY * m_base


def test_portal_displacements_match_closed_form():
    frame, p = make_portal_frame()
    result = analyze_frame(frame)
    disp = by_id(result["displacements"])
    ei = p["e"] * p["i"]
    p_val, l = p["p"], p["l"]
    u_theory = 5 * p_val * l ** 3 / (84 * ei)
    th_theory = -p_val * l ** 2 / (28 * ei)
    assert abs(disp["2"]["ux"] - u_theory) < TOL_THEORY * abs(u_theory)
    assert abs(disp["3"]["ux"] - u_theory) < TOL_THEORY * abs(u_theory)
    assert abs(disp["2"]["theta"] - th_theory) < TOL_THEORY * abs(th_theory)
    assert abs(disp["3"]["theta"] - th_theory) < TOL_THEORY * abs(th_theory)
    # 固定端无位移
    for nid in ("1", "4"):
        d = disp[nid]
        assert abs(d["ux"]) < TOL_DISP and abs(d["uy"]) < TOL_DISP and abs(d["theta"]) < TOL_DISP


def test_portal_beam_end_and_midspan_moments():
    frame, p = make_portal_frame()
    result = analyze_frame(frame)
    beam = next(m for m in result["member_forces"] if m["member_id"] == "b")
    p_val, l = p["p"], p["l"]
    # 梁端弯矩大小 3PL/14（杆对节点）。侧移刚架横梁为双曲率，
    # 两端局部弯矩同号、剪力为常数，弯矩图在跨中过零（反弯点）。
    m_end = 3 * p_val * l / 14
    assert abs(beam["end_i"]["moment"] - m_end) < TOL_THEORY * m_end
    assert abs(beam["end_j"]["moment"] - m_end) < TOL_THEORY * m_end
    # 跨中截面弯矩为零（教材门式刚架侧移工况的反弯点）。
    # 有限轴向刚度使节点竖向有微小变位，故按内力量级用相对容差
    m_mid = section_sagging_moment(frame, result, "b", l / 2, qy=0.0)
    assert abs(m_mid) < TOL_THEORY * m_end
    # 弯矩图线性：x=L/4 处为跨端的一半（端部上侧受拉）
    m_quarter = section_sagging_moment(frame, result, "b", l / 4, qy=0.0)
    assert abs(m_quarter - m_end / 2) < TOL_THEORY * m_end

def test_portal_column_shears_and_axial():
    frame, p = make_portal_frame()
    result = analyze_frame(frame)
    members = by_id(result["member_forces"], key="member_id")
    p_val = p["p"]
    # 柱剪力（柱为竖直、局部 x̄ 向上）：每根柱剪力大小 P/2（相对容差）
    assert abs(abs(members["c1"]["end_i"]["shear"]) - p_val / 2) < TOL_THEORY * p_val
    assert abs(abs(members["c2"]["end_i"]["shear"]) - p_val / 2) < TOL_THEORY * p_val
    # 轴力大小 3P/7；左柱受拉、右柱受压（轴力标量拉为正）
    assert abs(members["c1"]["axial_force"] - 3 * p_val / 7) < TOL_THEORY * p_val
    assert abs(members["c2"]["axial_force"] + 3 * p_val / 7) < TOL_THEORY * p_val


def test_fixed_beam_uniform_load_textbook():
    """两端固定梁满跨均布荷载 q：固端弯矩 qL²/12，跨中 qL²/24（下侧受拉）。"""
    frame, p = make_beam(q_down=3.0, fixed=True)
    result = analyze_frame(frame)
    q, l = p["q"], p["l"]
    beam = result["member_forces"][0]
    reactions = by_id(result["reactions"])

    # 固端弯矩（杆对节点）：i 端顺时针 -qL²/12，j 端逆时针 +qL²/12
    fe = q * l ** 2 / 12
    assert abs(beam["end_i"]["moment"] - (-fe)) < TOL_MOMENT
    assert abs(beam["end_j"]["moment"] - fe) < TOL_MOMENT
    # 竖向反力各 qL/2，向上
    assert abs(reactions["A"]["fy"] - q * l / 2) < TOL_FORCE
    assert abs(reactions["B"]["fy"] - q * l / 2) < TOL_FORCE
    # 跨中截面标量弯矩：下侧受拉 +qL²/24
    m_mid = section_sagging_moment(frame, result, "beam", l / 2, qy=-q)
    assert abs(m_mid - q * l ** 2 / 24) < TOL_MOMENT
    # 四分之一跨：R_A x - qx²/2 - qL²/12（下侧受拉为正）= qL²/96
    m_q1 = section_sagging_moment(frame, result, "beam", l / 4, qy=-q)
    expected_q1 = q * l ** 2 / 96
    assert abs(m_q1 - expected_q1) < TOL_MOMENT


def test_simply_supported_beam_uniform_load_textbook():
    """简支梁满跨均布荷载：端弯矩为 0，跨中 qL²/8，反力各 qL/2。"""
    frame, p = make_beam(q_down=3.0, fixed=False)
    result = analyze_frame(frame)
    q, l = p["q"], p["l"]
    beam = result["member_forces"][0]
    reactions = by_id(result["reactions"])

    assert abs(beam["end_i"]["moment"]) < TOL_MOMENT
    assert abs(beam["end_j"]["moment"]) < TOL_MOMENT
    assert abs(reactions["A"]["fy"] - q * l / 2) < TOL_FORCE
    assert abs(reactions["B"]["fy"] - q * l / 2) < TOL_FORCE
    m_mid = section_sagging_moment(frame, result, "beam", l / 2, qy=-q)
    assert abs(m_mid - q * l ** 2 / 8) < TOL_MOMENT
