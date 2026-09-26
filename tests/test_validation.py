"""非法输入与奇异性测试。

需求列出的每一类非法模型都必须：
- 返回带机器可读 code 的 FrameError（不抛裸异常、不丢杆、不返回 NaN）；
- HTTP 层同样映射为结构化错误响应（见 test_http_api.py）。
"""

from __future__ import annotations

import numpy as np
import pytest

from framesolver.analysis import analyze_frame
from framesolver.errors import (
    DisconnectedStructureError,
    DuplicateNodeIdError,
    FrameError,
    InsufficientSupportError,
    InvalidLoadError,
    InvalidPropertyError,
    MemberNodeNotFoundError,
    SingularMatrixError,
    ZeroLengthMemberError,
)
from framesolver.models import (
    DistributedLoad,
    FrameInput,
    Member,
    Node,
    NodalLoad,
)
from framesolver.solver import solve_system

# 一组合法基准部件，各测试只改动一处使其非法
E, A, I = 2.1e8, 1.0e-2, 1.0e-5


def _member(mid="m1", ni="1", nj="2", e=E, a=A, i_=I):
    return Member(id=mid, node_i=ni, node_j=nj, elastic_modulus=e, area=a, inertia=i_)


def test_duplicate_node_id_rejected():
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, True]),
            Node(id="1", x=1, y=0, restraints=[True, True, False]),
        ],
        members=[_member()],
    )
    with pytest.raises(DuplicateNodeIdError) as exc:
        analyze_frame(frame)
    assert "1" in exc.value.message


def test_member_endpoint_nonexistent_node_rejected():
    frame = FrameInput(
        nodes=[Node(id="1", x=0, y=0, restraints=[True, True, True])],
        members=[_member(ni="1", nj="9")],
    )
    with pytest.raises(MemberNodeNotFoundError) as exc:
        analyze_frame(frame)
    assert "9" in exc.value.message
    assert exc.value.code == "MEMBER_NODE_NOT_FOUND"


def test_zero_length_member_rejected():
    frame = FrameInput(
        nodes=[
            Node(id="1", x=1.0, y=2.0, restraints=[True, True, True]),
            Node(id="2", x=1.0, y=2.0, restraints=[False, False, False]),
        ],
        members=[_member()],
    )
    with pytest.raises(ZeroLengthMemberError):
        analyze_frame(frame)


@pytest.mark.parametrize("field", ["elastic_modulus", "area", "inertia"])
@pytest.mark.parametrize("bad", [0.0, -1.0, -1.0e-9])
def test_nonpositive_section_properties_rejected(field, bad):
    """内部核算路径（绕过 Pydantic 入站校验）必须给出 INVALID_PROPERTY。

    常规 HTTP 请求中的非正数先被 Pydantic 的 gt=0 拦成 INVALID_REQUEST，
    这里用 model_construct 直接构造非法对象，验证 validation.py 的防线。
    """
    values = {"elastic_modulus": E, "area": A, "inertia": I, field: bad}
    bad_member = Member.model_construct(id="m1", node_i="1", node_j="2", **values)
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, False]),
            Node(id="2", x=2, y=0, restraints=[True, True, False]),
        ],
        members=[bad_member],
    )
    with pytest.raises(InvalidPropertyError) as exc:
        analyze_frame(frame)
    assert exc.value.code == "INVALID_PROPERTY"


def test_insufficient_supports_unrestrained_body_rejected():
    """完全没有支座：3 个刚体模态全无约束。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[False, False, False]),
            Node(id="2", x=1, y=0, restraints=[False, False, False]),
        ],
        members=[_member()],
    )
    with pytest.raises(InsufficientSupportError) as exc:
        analyze_frame(frame)
    assert exc.value.code == "INSUFFICIENT_SUPPORT"


def test_roller_beam_rotationally_free_rejected():
    """两个平行竖向链杆：水平移动与转动均未锁住。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[False, True, False]),
            Node(id="2", x=2, y=0, restraints=[False, True, False]),
        ],
        members=[_member()],
    )
    with pytest.raises(InsufficientSupportError):
        analyze_frame(frame)


def test_three_parallel_rollers_still_rotatable_rejected():
    """同一根梁下三个竖向链杆：转动约束方向共线，整体转动依然自由。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[False, True, False]),
            Node(id="2", x=1, y=0, restraints=[False, True, False]),
            Node(id="3", x=2, y=0, restraints=[False, True, False]),
        ],
        members=[_member(ni="1", nj="2"), _member(mid="m2", ni="2", nj="3")],
    )
    with pytest.raises(InsufficientSupportError):
        analyze_frame(frame)


def test_disconnected_two_separate_frames_rejected():
    """结构被分成互不相连的两块，每块各自稳定。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, True]),
            Node(id="2", x=1, y=0, restraints=[True, True, False]),
            Node(id="3", x=5, y=0, restraints=[True, True, True]),
            Node(id="4", x=6, y=0, restraints=[True, True, False]),
        ],
        members=[_member(ni="1", nj="2"), _member(mid="m2", ni="3", nj="4")],
    )
    with pytest.raises(DisconnectedStructureError) as exc:
        analyze_frame(frame)
    # 错误信息要把两块的节点都点出来，让人看得懂
    assert "1" in exc.value.message and "3" in exc.value.message


def test_isolated_node_rejected():
    """一个完全不属于任何杆件的孤立节点（相当于独立的零构件分量）。"""
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, False]),
            Node(id="2", x=1, y=0, restraints=[True, True, False]),
            Node(id="9", x=3, y=3, restraints=[True, True, True]),
        ],
        members=[_member()],
    )
    with pytest.raises(DisconnectedStructureError):
        analyze_frame(frame)


def test_nodal_load_on_missing_node_rejected():
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, False]),
            Node(id="2", x=1, y=0, restraints=[True, True, False]),
        ],
        members=[_member()],
        nodal_loads=[NodalLoad(node_id="404", fx=1.0)],
    )
    with pytest.raises(InvalidLoadError) as exc:
        analyze_frame(frame)
    assert "404" in exc.value.message


def test_distributed_load_on_missing_member_rejected():
    frame = FrameInput(
        nodes=[
            Node(id="1", x=0, y=0, restraints=[True, True, False]),
            Node(id="2", x=1, y=0, restraints=[True, True, False]),
        ],
        members=[_member()],
        distributed_loads=[DistributedLoad(member_id="no-such", qy=-2.0)],
    )
    with pytest.raises(InvalidLoadError):
        analyze_frame(frame)


def test_singular_mechanism_raises_without_nan():
    """求解器层的最后一道防线：缩聚矩阵奇异时必须报 SINGULAR_MATRIX。

    本服务所有节点均为刚性连接、杆件 EA/EI 恒正，因此能通过刚体约束
    检查（秩 3）的正常模型 K_ff 必正定；真正的机构无法由输入表达。
    但 solver 仍须对秩亏损矩阵明确报错，绝不返回 NaN / lstsq 蒙混。
    """
    zero = np.zeros((3, 3))
    with pytest.raises(SingularMatrixError) as exc:
        solve_system(zero, np.zeros(3))
    assert exc.value.code == "SINGULAR_MATRIX"
    assert "机构" in exc.value.message or "奇异" in exc.value.message


def test_rank_deficient_matrix_detected():
    """秩为 2 的 3 阶矩阵必须被判奇异，且 lstsq 也不会被拿来蒙混。"""
    k = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    with pytest.raises(SingularMatrixError):
        solve_system(k, np.array([1.0, 2.0, 3.0]))


def test_all_frame_errors_carry_code_and_message():
    """所有业务错误都带机器可读 code 与非空中文说明。"""
    for err_cls in [
        DuplicateNodeIdError,
        MemberNodeNotFoundError,
        ZeroLengthMemberError,
        InvalidPropertyError,
        InvalidLoadError,
        DisconnectedStructureError,
        InsufficientSupportError,
        SingularMatrixError,
    ]:
        err = err_cls("说明文本")
        assert isinstance(err, FrameError)
        assert err.code and err.code.isupper()
        assert err.message == "说明文本"
        assert 400 <= err.http_status < 500
