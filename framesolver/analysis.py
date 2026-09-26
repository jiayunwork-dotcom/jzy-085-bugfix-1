"""把各功能段串成一次完整的直接刚度法核算。"""

from __future__ import annotations

import numpy as np

from . import assembly, element, forces as forces_mod
from .constraints import classify_dofs, condense
from .element import local_stiffness, uniform_transverse_equivalent_loads
from .geometry import force_to_global, stiffness_to_global, transformation_matrix
from .solver import solve_system
from .validation import validate


def analyze_frame(frame) -> dict:
    """对一副刚架执行完整核算，返回可直接序列化为 SolveResponse 的字典。"""
    # 1) 校验（含连通性与整体刚体约束检查），顺便复用几何信息
    validation_info = validate(frame)
    geometries = validation_info["geometries"]
    node_index = validation_info["node_index"]
    n_nodes = len(frame.nodes)
    ndof = 3 * n_nodes

    # 分布荷载按杆件编号索引（当前每根杆至多一条，多条时叠加）
    udl_by_member: dict[str, float] = {}
    for dist in frame.distributed_loads:
        udl_by_member[dist.member_id] = udl_by_member.get(dist.member_id, 0.0) + dist.qy

    # 2) 逐杆：局部刚度 -> 坐标变换 -> 等效节点力
    k_global_list: list[np.ndarray] = []
    eq_global_list: list[np.ndarray] = []
    local_data: list[tuple] = []
    for member in frame.members:
        geom = geometries[member.id]
        k_local = local_stiffness(
            geom.length, member.elastic_modulus, member.area, member.inertia
        )
        t = transformation_matrix(geom)
        k_global_list.append(stiffness_to_global(k_local, t))

        qy = udl_by_member.get(member.id, 0.0)
        eq_local = uniform_transverse_equivalent_loads(geom.length, qy)
        eq_global_list.append(force_to_global(eq_local, t))
        local_data.append((member, geom, k_local, t, eq_local))

    # 3) 总刚与总荷载向量组装
    stiffness, load = assembly.assemble(
        ndof,
        frame.members,
        node_index,
        k_global_list,
        eq_global_list,
        frame.nodal_loads,
    )

    # 4) 划行划列施加约束，求解自由自由度
    restraints = [node.restraints for node in frame.nodes]
    free_dofs, restrained_dofs = classify_dofs(restraints)
    k_ff, p_f = condense(stiffness, load, free_dofs)
    d_free = solve_system(k_ff, p_f)  # 奇异矩阵在此抛 SingularMatrixError

    # 5) 还原完整位移向量（被约束自由度位移为零）
    displacement = np.zeros(ndof, dtype=float)
    displacement[free_dofs] = d_free
    _guard_finite(displacement, "节点位移")

    # 6) 各杆杆端内力（局部刚度 × 局部位移）
    member_forces_out: list[dict] = []
    for member, geom, k_local, t, eq_local in local_data:
        ii = node_index[member.node_i]
        jj = node_index[member.node_j]
        member_dofs = np.array([3 * ii, 3 * ii + 1, 3 * ii + 2,
                                3 * jj, 3 * jj + 1, 3 * jj + 2])
        f_local = forces_mod.member_end_forces_local(
            k_local, t, displacement[member_dofs], eq_local
        )
        _guard_finite(f_local, f"杆件 {member.id} 的杆端力")
        # f_local 为“杆作用于节点”：i 端轴向分量即轴力（拉正）；j 端与之反号
        axial = float(f_local[0])

        member_forces_out.append(
            {
                "member_id": member.id,
                "node_i": member.node_i,
                "node_j": member.node_j,
                "end_i": {
                    "axial": float(f_local[0]),
                    "shear": float(f_local[1]),
                    "moment": float(f_local[2]),
                },
                "end_j": {
                    "axial": float(f_local[3]),
                    "shear": float(f_local[4]),
                    "moment": float(f_local[5]),
                },
                "axial_force": axial,
            }
        )

    # 7) 支座反力回代
    reaction_vector = forces_mod.reactions(stiffness, displacement, load, restrained_dofs)
    _guard_finite(reaction_vector, "支座反力")

    reactions_out: list[dict] = []
    for node in frame.nodes:
        idx = node_index[node.id]
        reactions_out.append(
            {
                "node_id": node.id,
                "fx": float(reaction_vector[3 * idx]),
                "fy": float(reaction_vector[3 * idx + 1]),
                "moment": float(reaction_vector[3 * idx + 2]),
                "restrained": [bool(r) for r in node.restraints],
            }
        )

    displacements_out = []
    for node in frame.nodes:
        idx = node_index[node.id]
        displacements_out.append(
            {
                "node_id": node.id,
                "ux": float(displacement[3 * idx]),
                "uy": float(displacement[3 * idx + 1]),
                "theta": float(displacement[3 * idx + 2]),
            }
        )

    return {
        "success": True,
        "displacements": displacements_out,
        "member_forces": member_forces_out,
        "reactions": reactions_out,
    }


def _guard_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        # 正常路径下 solver 已拦截；这里保证任何阶段都不会吐出 NaN
        raise RuntimeError(f"{label}中出现非有限值（NaN 或无穷）")
