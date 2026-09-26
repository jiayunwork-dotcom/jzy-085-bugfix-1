"""补充教材算例：悬臂梁端部集中力；以及合法支承不被误判。"""

from __future__ import annotations

from framesolver.analysis import analyze_frame
from framesolver.models import FrameInput, Member, Node, NodalLoad
from conftest import TOL_THEORY, by_id


def test_cantilever_end_point_load_textbook():
    """悬臂梁（仅左端固定，3 个约束即锁住全部刚体模态）端部横向集中力 P。

    教材解（忽略剪切变形）：
        自由端挠度  PL³/(3EI)，转角 PL²/(2EI)
        固定端弯矩 PL，剪力恒为 P
    """
    l, p = 3.0, 5.0
    e, a, i_ = 3.0e7, 5.0e-2, 2.0e-4
    frame = FrameInput(
        nodes=[
            Node(id="A", x=0.0, y=0.0, restraints=[True, True, True]),
            Node(id="B", x=l, y=0.0, restraints=[False, False, False]),
        ],
        members=[Member(id="beam", node_i="A", node_j="B",
                        elastic_modulus=e, area=a, inertia=i_)],
        nodal_loads=[NodalLoad(node_id="B", fx=0.0, fy=-p, moment=0.0)],
    )
    result = analyze_frame(frame)
    ei = e * i_
    disp = by_id(result["displacements"])
    assert abs(disp["B"]["uy"] - (-p * l ** 3 / (3 * ei))) < TOL_THEORY * abs(
        disp["B"]["uy"]
    )
    assert abs(disp["B"]["theta"] - (-p * l ** 2 / (2 * ei))) < TOL_THEORY * abs(
        disp["B"]["theta"]
    )

    beam = result["member_forces"][0]
    reactions = by_id(result["reactions"])
    # 固定端：竖向反力 +P、反力矩 +PL（逆时针，支座对结构）
    assert abs(reactions["A"]["fy"] - p) < TOL_THEORY * p
    assert abs(reactions["A"]["moment"] - p * l) < TOL_THEORY * p * l
    # 杆端力（杆对固定端节点）：向下的剪力 -P、顺时针弯矩 -PL
    assert abs(beam["end_i"]["shear"] - (-p)) < TOL_THEORY * p
    assert abs(beam["end_i"]["moment"] - (-p * l)) < TOL_THEORY * p * l


def test_pinned_plus_roller_is_valid():
    """经典静定简支梁支承（固定铰 + 水平滚动铰）必须被接受且能求解。"""
    frame = FrameInput(
        nodes=[
            Node(id="A", x=0.0, y=0.0, restraints=[True, True, False]),
            Node(id="B", x=5.0, y=0.0, restraints=[False, True, False]),
        ],
        members=[Member(id="beam", node_i="A", node_j="B",
                        elastic_modulus=3.0e7, area=5.0e-2, inertia=2.0e-4)],
        nodal_loads=[NodalLoad(node_id="B", fx=2.0, fy=-4.0)],
    )
    result = analyze_frame(frame)
    assert result["success"] is True
    reactions = by_id(result["reactions"])
    # 水平力全部由固定铰承担
    assert abs(reactions["A"]["fx"] + 2.0) < 1e-8
    assert abs(reactions["B"]["fx"]) < 1e-8  # 滚动端无水平反力分量
