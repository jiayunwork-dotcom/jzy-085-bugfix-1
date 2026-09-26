"""约束施加方式：自由度分类（“划行划列”法）。

本服务全程只使用这一种约束处理方式（不使用大数罚函数）：

1. 把所有自由度分成“自由 f”与“被支座约束 r”两组；
2. 被约束自由度的位移直接置零，从控制方程中删去对应的行和列，
   只求解缩聚方程 ``K_ff d_f = P_f``；
3. 位移补零还原成全向量后，再用 ``R = K_rr d_r + K_rf d_f - P_r``
   由被约束行回代求支座反力（见 forces.py）。

这样约束被精确满足，不存在罚系数选取带来的数值病态问题。
"""

from __future__ import annotations

import numpy as np


def classify_dofs(restraints: list[list[bool]]) -> tuple[np.ndarray, np.ndarray]:
    """按节点顺序展开约束标记，返回 (自由自由度, 受约束自由度) 索引数组。"""
    n_nodes = len(restraints)
    restrained_mask = np.zeros(3 * n_nodes, dtype=bool)
    for i, flags in enumerate(restraints):
        restrained_mask[3 * i : 3 * i + 3] = flags
    all_dofs = np.arange(3 * n_nodes)
    free = all_dofs[~restrained_mask]
    restrained = all_dofs[restrained_mask]
    return free, restrained


def condense(stiffness: np.ndarray, load: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """取出缩聚矩阵 K_ff 与右端项 P_f。"""
    return stiffness[np.ix_(free, free)], load[free]
