"""输入校验：几何 / 拓扑 / 材料 / 连通性 / 整体刚体约束。

任何一条不满足都抛出带机器可读代码的 :class:`FrameError`，
不允许程序崩溃，也不允许悄悄丢弃杆件或荷载。
"""

from __future__ import annotations

import math

import numpy as np

from .errors import (
    DisconnectedStructureError,
    DuplicateMemberIdError,
    DuplicateNodeIdError,
    InsufficientSupportError,
    InvalidLoadError,
    InvalidPropertyError,
    MemberNodeNotFoundError,
    ZeroLengthMemberError,
)
from .geometry import frame_geometry


def _is_finite_number(value) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def validate(frame) -> dict[str, dict]:
    """执行全部前置校验，返回 ``{杆件编号: 几何信息}`` 供后续阶段复用。"""
    # ------------------------------------------------------------------
    # 1) 节点：编号唯一、坐标有限
    # ------------------------------------------------------------------
    node_ids: list[str] = []
    for node in frame.nodes:
        if not _is_finite_number(node.x) or not _is_finite_number(node.y):
            raise InvalidPropertyError(f"节点 {node.id} 的坐标必须是有限实数")
        node_ids.append(node.id)
    if len(set(node_ids)) != len(node_ids):
        dupes = sorted({nid for nid in node_ids if node_ids.count(nid) > 1})
        raise DuplicateNodeIdError(f"节点编号重复：{', '.join(dupes)}")

    node_set = set(node_ids)
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # ------------------------------------------------------------------
    # 2) 杆件：编号唯一、端点存在、材料截面为正、杆长非零
    # ------------------------------------------------------------------
    member_ids = [m.id for m in frame.members]
    if len(set(member_ids)) != len(member_ids):
        dupes = sorted({mid for mid in member_ids if member_ids.count(mid) > 1})
        raise InvalidPropertyError(f"杆件编号重复：{', '.join(dupes)}")

    geometries: dict[str, dict] = {}
    coords = {n.id: (n.x, n.y) for n in frame.nodes}
    for member in frame.members:
        for name, value in (
            ("弹性模量", member.elastic_modulus),
            ("截面面积", member.area),
            ("惯性矩", member.inertia),
        ):
            if not _is_finite_number(value) or value <= 0:
                raise InvalidPropertyError(
                    f"杆件 {member.id} 的{name}必须是正数，收到 {value}"
                )
        if member.node_i not in node_set:
            raise MemberNodeNotFoundError(
                f"杆件 {member.id} 的 i 端指向了不存在的节点 {member.node_i}"
            )
        if member.node_j not in node_set:
            raise MemberNodeNotFoundError(
                f"杆件 {member.id} 的 j 端指向了不存在的节点 {member.node_j}"
            )

        xi, yi = coords[member.node_i]
        xj, yj = coords[member.node_j]
        geom = frame_geometry(xi, yi, xj, yj)  # 零杆长在此抛 ZeroLengthMemberError
        geometries[member.id] = geom

    # ------------------------------------------------------------------
    # 3) 荷载：作用对象必须存在、数值有限
    # ------------------------------------------------------------------
    member_set = set(member_ids)
    for nodal in frame.nodal_loads:
        if nodal.node_id not in node_set:
            raise InvalidLoadError(
                f"节点荷载作用在不存在的节点 {nodal.node_id} 上"
            )
        for name, value in (("fx", nodal.fx), ("fy", nodal.fy), ("moment", nodal.moment)):
            if not _is_finite_number(value):
                raise InvalidPropertyError(
                    f"节点 {nodal.node_id} 的荷载分量 {name} 必须是有限实数"
                )
    for dist in frame.distributed_loads:
        if dist.member_id not in member_set:
            raise InvalidLoadError(
                f"分布荷载作用在不存在的杆件 {dist.member_id} 上"
            )
        if not _is_finite_number(dist.qy):
            raise InvalidPropertyError(
                f"杆件 {dist.member_id} 的分布荷载集度必须是有限实数"
            )

    # ------------------------------------------------------------------
    # 4) 拓扑连通性：所有节点必须通过杆件连成一个整体
    #    （孤立节点、互不相连的几块都明确报错）
    # ------------------------------------------------------------------
    _check_connected(frame, node_set)

    # ------------------------------------------------------------------
    # 5) 整体刚体运动：平面刚体有 3 个模态（水平、竖向、转动），
    #    支座约束必须把它们全部消除
    # ------------------------------------------------------------------
    _check_rigid_body_restraint(frame, node_index, geometries)

    return {"geometries": geometries, "node_index": node_index}


def _check_connected(frame, node_set: set[str]) -> None:
    """并查集检查全部节点是否同属一个连通分量。"""
    parent = {nid: nid for nid in node_set}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for member in frame.members:
        union(member.node_i, member.node_j)

    roots = {find(nid) for nid in node_set}
    if len(roots) > 1:
        # 找出各分量里的代表节点，让说明可读
        groups: dict[str, list[str]] = {}
        for nid in sorted(node_set):
            groups.setdefault(find(nid), []).append(nid)
        detail = "；".join(
            f"第 {idx} 块节点 {{{', '.join(members)}}}"
            for idx, members in enumerate(groups.values(), start=1)
        )
        raise DisconnectedStructureError(f"结构被分成互不相连的多个部分：{detail}")


def _check_rigid_body_restraint(frame, node_index: dict[str, int], geometries: dict) -> None:
    """检查支座能否锁住平面刚体的 3 个整体运动模态。

    做法：在每个被约束自由度上取单位约束方向（全局基向量），
    组成 3×nr 的“约束方向矩阵”，其秩必须为 3。
    约束方向（对参考点取矩，矩心取原点）：

    - 水平链杆 (1, 0)，约束功对应力矩臂 -y
    - 竖向链杆 (0, 1)，对应力矩臂  x
    - 转动约束对应纯力矩 (0, 0, 1)

    秩小于 3 即存在未被锁住的整体刚体平动 / 转动。
    """
    coords = {n.id: np.array([n.x, n.y], dtype=float) for n in frame.nodes}
    directions = []
    for node in frame.nodes:
        x, y = node.x, node.y
        if node.restraints[0]:  # 水平
            directions.append(np.array([1.0, 0.0, -y]))
        if node.restraints[1]:  # 竖向
            directions.append(np.array([0.0, 1.0, x]))
        if node.restraints[2]:  # 转角
            directions.append(np.array([0.0, 0.0, 1.0]))

    if not directions:
        raise InsufficientSupportError(
            "结构没有任何支座约束，可整体自由移动和转动（3 个刚体模态均未消除）"
        )

    constraint_matrix = np.column_stack(directions)
    rank = int(np.linalg.matrix_rank(constraint_matrix, tol=1e-10))
    if rank < 3:
        raise InsufficientSupportError(
            f"支座约束不足以消除整体刚体运动：约束方向矩阵的秩为 {rank}（应为 3），"
            "仍存在可自由发生的整体水平移动、竖向移动或转动"
        )
