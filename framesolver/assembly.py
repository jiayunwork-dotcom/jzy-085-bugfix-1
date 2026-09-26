"""总刚组装与荷载向量组装。

每个节点 3 个自由度，整体自由度编号为 ``3*node_index + dof``。
逐杆把整体坐标下的单元刚度叠加（scatter-add）进总刚；
分布荷载的等效节点力、节点集中荷载叠加进总荷载向量。

提供两种精度：

- :func:`assemble`：标准 float64 组装；
- :func:`assemble_compensated`：用 double-double 补偿精度累加
  （见 precision.py），消除 scatter-add 的随机舍入，供 solver
  的混合精度精化使用。输入单元矩阵仍是 float64——其系数对所有
  单元一致，相当于物理参数的整体微扰，不会被条件数放大。
"""

from __future__ import annotations

import numpy as np

from .precision import DD, as_dd, dd_add, dd_zeros


def dof_indices(node_index: int) -> tuple[int, int, int]:
    """节点在整体方程中的三个自由度编号 [水平, 竖向, 转角]。"""
    base = 3 * node_index
    return base, base + 1, base + 2


def element_dof_map(index_i: int, index_j: int) -> np.ndarray:
    """杆件两端共 6 个自由度在总刚中的位置。"""
    return np.array([*dof_indices(index_i), *dof_indices(index_j)], dtype=int)


def assemble(
    ndof: int,
    members: list,
    node_index: dict[str, int],
    k_global_by_member: list[np.ndarray],
    eq_load_global_by_member: list[np.ndarray],
    nodal_loads: list,
) -> tuple[np.ndarray, np.ndarray]:
    """组装总刚度矩阵与总荷载向量。

    参数
    ----
    ndof: 总自由度数（= 3 × 节点数）
    members: 校验后的杆件模型列表
    node_index: 节点编号 -> 紧凑索引
    k_global_by_member / eq_load_global_by_member:
        与 members 等长、且已转换到整体坐标的单元刚度 / 等效节点力
    nodal_loads: 校验后的节点荷载模型列表
    """
    stiffness = np.zeros((ndof, ndof), dtype=float)
    load = np.zeros(ndof, dtype=float)

    for member, k_g, eq_g in zip(
        members, k_global_by_member, eq_load_global_by_member, strict=True
    ):
        dofs = element_dof_map(node_index[member.node_i], node_index[member.node_j])
        # np.ix_ 做 6×6 分块 scatter-add
        stiffness[np.ix_(dofs, dofs)] += k_g
        load[dofs] += eq_g

    for nodal in nodal_loads:
        base = 3 * node_index[nodal.node_id]
        load[base] += nodal.fx
        load[base + 1] += nodal.fy
        load[base + 2] += nodal.moment

    return stiffness, load


def assemble_compensated(
    ndof: int,
    members: list,
    node_index: dict[str, int],
    k_global_by_member: list[np.ndarray],
    eq_load_global_by_member: list[np.ndarray],
    nodal_loads: list,
) -> tuple[DD, DD]:
    """补偿精度版本的组装，返回 (K_dd, P_dd)，见模块 docstring。

    单元矩阵按 float64 计算（系数级误差是系统性的、不被条件数放大），
    但 scatter-add 累加全程用 double-double，使总刚与荷载向量
    对“由 float64 系数组装出的数学结果”精确到 ~1e-32 相对。
    """
    stiffness = dd_zeros((ndof, ndof))
    load = dd_zeros((ndof,))

    for member, k_g, eq_g in zip(
        members, k_global_by_member, eq_load_global_by_member, strict=True
    ):
        dofs = element_dof_map(node_index[member.node_i], node_index[member.node_j])
        block = stiffness.hi[np.ix_(dofs, dofs)], stiffness.lo[np.ix_(dofs, dofs)]
        summed = dd_add(DD(*block), as_dd(k_g))
        stiffness.hi[np.ix_(dofs, dofs)] = summed.hi
        stiffness.lo[np.ix_(dofs, dofs)] = summed.lo

        p_block = dd_add(
            DD(load.hi[dofs], load.lo[dofs]),
            as_dd(eq_g),
        )
        load.hi[dofs] = p_block.hi
        load.lo[dofs] = p_block.lo

    for nodal in nodal_loads:
        base = 3 * node_index[nodal.node_id]
        p_block = dd_add(
            DD(load.hi[base : base + 3], load.lo[base : base + 3]),
            as_dd(np.array([nodal.fx, nodal.fy, nodal.moment], dtype=float)),
        )
        load.hi[base : base + 3] = p_block.hi
        load.lo[base : base + 3] = p_block.lo

    return stiffness, load
