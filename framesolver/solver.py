"""线性方程求解与奇异性判定。

判定策略（不依赖浮点异常，明确给出可解释的结论）：

1. 对缩聚矩阵 K_ff 做奇异值分解，以
   ``最小奇异值 < 阈值 × 最大奇异值``（阈值取 1e-10）判为秩亏损；
2. 秩正常时优先用 Cholesky 分解（K_ff 理论上对称正定）求解；
   分解失败则退回对称 SVD 求解，再按残差复核；
3. 求解后强制残差复核，残差相对过大一律按奇异 / 病态处理，
   绝不把 NaN 或不可信的数字当结果返回。
"""

from __future__ import annotations

import numpy as np

from .errors import SingularMatrixError

# 秩亏损的相对奇异值阈值
_RANK_TOL = 1e-10
# 残差相对阈值（残差按荷载向量的尺度归一化）
_RESIDUAL_TOL = 1e-8


def _check_finite(matrix: np.ndarray) -> None:
    if not np.all(np.isfinite(matrix)):
        raise SingularMatrixError("缩聚刚度矩阵中出现非有限值（NaN 或无穷），无法求解")


def solve_system(k_ff: np.ndarray, p_f: np.ndarray) -> np.ndarray:
    """求解 K_ff d_f = P_f；奇异时抛 :class:`SingularMatrixError`。"""
    _check_finite(k_ff)
    n = k_ff.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    # 强制对称，消除组装中可能残留的量级 1e-16 的不对称
    k_sym = 0.5 * (k_ff + k_ff.T)

    # ---- 第一步：SVD 判秩 ------------------------------------------------
    singular_values = np.linalg.svd(k_sym, compute_uv=False)
    s_max = singular_values[0]
    s_min = singular_values[-1]
    if not np.isfinite(s_min) or s_max == 0.0 or s_min < _RANK_TOL * s_max:
        effective_rank = int(np.count_nonzero(singular_values >= _RANK_TOL * s_max))
        raise SingularMatrixError(
            f"缩聚刚度矩阵奇异：{n} 个自由自由度中仅 {effective_rank} 个独立，"
            "结构存在可动机构或零能变形模态（约束冗余 / 机构可动）"
        )

    # ---- 第二步：Cholesky 优先，失败退回 SVD -----------------------------
    try:
        chol = np.linalg.cholesky(k_sym)
        d_f = np.linalg.solve(chol.T, np.linalg.solve(chol, p_f))
    except np.linalg.LinAlgError:
        d_f = np.linalg.lstsq(k_sym, p_f, rcond=None)[0]

    # ---- 第三步：残差复核 ------------------------------------------------
    if not np.all(np.isfinite(d_f)):
        raise SingularMatrixError("位移解中出现非有限值（NaN 或无穷），判定为奇异矩阵")

    residual = k_sym @ d_f - p_f
    scale = max(float(np.max(np.abs(p_f))), float(s_max) * float(np.max(np.abs(d_f))), 1.0)
    if float(np.max(np.abs(residual))) > _RESIDUAL_TOL * scale:
        raise SingularMatrixError(
            "方程残差超出容差，刚度矩阵病态或近奇异，结果不可信"
        )

    return d_f
