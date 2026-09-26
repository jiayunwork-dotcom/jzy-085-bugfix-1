"""错误定义。

所有可预期的业务错误都派生自 :class:`FrameError`，带：

- ``code``：稳定的机器可读代码（测试和上游程序按代码分支处理）
- ``message``：给人看的中文说明
- ``http_status``：对应的 HTTP 状态码
"""

from __future__ import annotations


class FrameError(Exception):
    """刚架核算业务错误的公共基类。"""

    code: str = "FRAME_ERROR"
    http_status: int = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DuplicateNodeIdError(FrameError):
    """节点编号重复。"""

    code = "DUPLICATE_NODE_ID"
    http_status = 400


class DuplicateMemberIdError(FrameError):
    """杆件编号重复。"""

    code = "DUPLICATE_MEMBER_ID"
    http_status = 400


class MemberNodeNotFoundError(FrameError):
    """杆件端点指向了不存在的节点。"""

    code = "MEMBER_NODE_NOT_FOUND"
    http_status = 400


class ZeroLengthMemberError(FrameError):
    """杆长为零。"""

    code = "ZERO_LENGTH_MEMBER"
    http_status = 400


class InvalidPropertyError(FrameError):
    """弹性模量 / 截面积 / 惯性矩取了非正数，或坐标 / 荷载不是有限数。"""

    code = "INVALID_PROPERTY"
    http_status = 400


class InvalidLoadError(FrameError):
    """节点荷载指向不存在的节点，或分布荷载指向不存在的杆件。"""

    code = "INVALID_LOAD_REFERENCE"
    http_status = 400


class DisconnectedStructureError(FrameError):
    """结构被分成互不相连的几块，或存在不属于任何杆件的孤立节点。"""

    code = "DISCONNECTED_STRUCTURE"
    http_status = 400


class InsufficientSupportError(FrameError):
    """约束不足以消除平面刚架的整体刚体运动（3 个：水平、竖向、转动）。"""

    code = "INSUFFICIENT_SUPPORT"
    http_status = 400


class SingularMatrixError(FrameError):
    """缩聚后的刚度矩阵奇异（存在机构 / 零能变形模态）。"""

    code = "SINGULAR_MATRIX"
    http_status = 422
