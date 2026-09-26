"""HTTP 输入 / 输出的数据模型（Pydantic v2）。

约定（输入、输出一致）：

- 节点三个自由度顺序固定为 [水平 u, 竖向 v, 转角 θ]；
- 整体坐标：x 水平向右、y 竖直向上，转角逆时针为正；
- 杆件局部坐标：x̄ 沿 i 端指向 j 端，ȳ 为 x̄ 逆时针转 90°。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 输入模型
# ---------------------------------------------------------------------------


class Node(BaseModel):
    id: str = Field(..., description="节点编号，必须在整个模型内唯一")
    x: float = Field(..., description="整体坐标 x")
    y: float = Field(..., description="整体坐标 y")
    # True = 该自由度被支座锁死；False = 自由。
    # [固定端] = [true, true, true]
    # [铰支座] = [true, true, false]
    # [水平滚动（仅竖向链杆）] = [false, true, false]
    restraints: list[bool] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="[水平, 竖向, 转角] 三个自由度是否被约束",
    )


class Member(BaseModel):
    id: str = Field(..., description="杆件编号，必须唯一")
    node_i: str = Field(..., description="i 端节点编号")
    node_j: str = Field(..., description="j 端节点编号")
    elastic_modulus: float = Field(..., gt=0, description="弹性模量 E，必须为正")
    area: float = Field(..., gt=0, description="截面面积 A，必须为正")
    inertia: float = Field(..., gt=0, description="截面惯性矩 I，必须为正")


class NodalLoad(BaseModel):
    node_id: str = Field(..., description="荷载作用的节点编号")
    fx: float = Field(0.0, description="整体坐标 x 方向集中力")
    fy: float = Field(0.0, description="整体坐标 y 方向集中力")
    moment: float = Field(0.0, description="绕 z 轴集中力矩，逆时针为正")


class DistributedLoad(BaseModel):
    member_id: str = Field(..., description="荷载作用的杆件编号")
    # 目前只支持满跨、垂直于杆轴的均布荷载（按局部坐标 ȳ 方向给出）
    load_type: Literal["uniform_transverse"] = "uniform_transverse"
    qy: float = Field(..., description="局部 ȳ 方向均布荷载集度，沿 +ȳ 为正")


class FrameInput(BaseModel):
    nodes: list[Node] = Field(..., min_length=1)
    members: list[Member] = Field(..., min_length=1)
    nodal_loads: list[NodalLoad] = Field(default_factory=list)
    distributed_loads: list[DistributedLoad] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 输出模型
# ---------------------------------------------------------------------------


class NodeDisplacement(BaseModel):
    node_id: str
    ux: float = Field(..., description="水平位移，沿 +x 为正")
    uy: float = Field(..., description="竖向位移，沿 +y 为正")
    theta: float = Field(..., description="转角，逆时针为正（弧度）")


class MemberEndForces(BaseModel):
    """某一端的局部坐标杆端力（杆件作用于节点的力）。"""

    axial: float = Field(..., description="沿局部 x̄ 的分量")
    shear: float = Field(..., description="沿局部 ȳ 的分量")
    moment: float = Field(..., description="杆端弯矩，逆时针为正")


class MemberForces(BaseModel):
    member_id: str
    node_i: str
    node_j: str
    # 两端分别给出“杆件作用于节点”的局部坐标力
    end_i: MemberEndForces
    end_j: MemberEndForces
    # 轴力标量：拉力为正（两端局部 x̄ 分量在“杆对节点”约定下天然相等）
    axial_force: float = Field(..., description="轴力，拉力为正")


class Reaction(BaseModel):
    node_id: str
    fx: float = Field(..., description="支座作用于结构的水平反力，+x 为正")
    fy: float = Field(..., description="支座作用于结构的竖向反力，+y 为正")
    moment: float = Field(..., description="支座反力矩，逆时针为正")
    # 哪个自由度真正由支座提供（未约束的反力分量恒为 0）
    restrained: list[bool]


class SolveResponse(BaseModel):
    success: Literal[True] = True
    displacements: list[NodeDisplacement]
    member_forces: list[MemberForces]
    reactions: list[Reaction]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
