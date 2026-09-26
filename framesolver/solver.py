"""线性方程求解与奇异性判定。

设计目标：一副结构能不能解，只取决于结构自身——与用户选用哪套
自洽单位制、杆件细分程度无关；真正奇异（机构 / 零能模态）时明确
报错，绝不返回 NaN 或不可信的数字。

步骤：

1. **对称对角缩放**：平动自由度的刚度量纲为 力/长度，转动自由度为
   力·长度，二者数值可相差 长度² 倍（N·mm 下可达 1e7 以上），使
   未缩放矩阵的条件数严重依赖单位制与网格密度。用 D = diag(K)^(-1/2)
   做对称缩放 K̂ = D·K·D（对角线全为 1）后，条件数只反映结构自身
   性态，与单位制无关。
2. **对称特征值判秩**（在缩放矩阵 K̂ 上）：真正的机构产生机器精度
   量级的零特征值（≲1e-15·λ_max），而离散细化造成的合法病态最小
   特征值也在 1e-11·λ_max 之上，二者相差两个数量级以上；阈值取
   n·eps（与 numpy 默认数值秩判据同级），把两类情形干净分开。
3. **Cholesky 求解**缩放系统（失败退回最小二乘），得到初始解。
4. **迭代精化**：在**未缩放的物理坐标**下用扩展精度（np.longdouble）
   计算残差 r = P − K·d，再经缩放 Cholesky 因子回代校正量，残差
   不再下降时停止。条件数 ~1e10 以上的病态系统也能收敛到所装矩阵
   的真实解（缩放后的解向量动态范围被拉大，不宜在缩放坐标下精化）。
5. **残差复核**：按后向误差 |r|/(|K|·|d|+|P|) 归一，超过 1e-8 一律按
   奇异 / 病态处理——归一化尺度同时含 |K|·|d|，因为 float64 解向量
   自身的表示舍入就会产生 ~eps·|K|·|d| 的残差（N·mm 下力的单位上
   可达 1e-4 量级），不能只按 |P| 归一一处理，绝不把 NaN 或不可信的
   数字当结果返回。
"""

from __future__ import annotations

import numpy as np

from .errors import SingularMatrixError

# 数值秩阈值 = n·eps（以最大特征值归一化）：
# 机构零能模态在 ~1e-16 量级，合法病态系统不低于 ~1e-11，间隔两个数量级
_RANK_TOL_FACTOR = 1.0
# 迭代精化最大步数（正常 2~3 步即收敛到机器精度）
_MAX_REFINE_STEPS = 4
# 残差相对阈值（残差按荷载 / 刚度·位移的尺度归一化，不含绝对量纲项）
_RESIDUAL_TOL = 1e-8

# 扩展精度：残差计算用；平台不支持（longdouble 等价 float64）时自动退化，
# 正确性不依赖它，只是病态系统下的精化收益变小
_EXTENDED = np.longdouble


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
    k64 = np.asarray(k_sym, dtype=float)
    p64 = np.asarray(p_f, dtype=float)
    if not np.all(np.isfinite(p64)):
        raise SingularMatrixError("荷载向量中出现非有限值（NaN 或无穷），无法求解")

    # ---- 第一步：对称对角缩放（消除平动 / 转动自由度的量纲差异） ------
    diag = np.diag(k64)
    if np.any(diag <= 0.0):
        raise SingularMatrixError(
            "缩聚刚度矩阵主对角线出现非正元素：存在完全没有刚度的自由度"
            "（可动机构 / 零能变形模态），无法求解"
        )
    scale = 1.0 / np.sqrt(diag)
    k_hat = k64 * np.outer(scale, scale)  # K̂ = D·K·D，主对角线全为 1

    # ---- 第二步：对称特征值判秩（结论与单位制无关） --------------------
    eigenvalues = np.linalg.eigvalsh(k_hat)
    lam_max = float(eigenvalues[-1])
    lam_min = float(eigenvalues[0])
    rank_tol = _RANK_TOL_FACTOR * n * np.finfo(float).eps
    if not np.isfinite(lam_min) or lam_min < rank_tol * lam_max:
        effective_rank = int(np.count_nonzero(eigenvalues >= rank_tol * lam_max))
        raise SingularMatrixError(
            f"缩聚刚度矩阵奇异：{n} 个自由自由度中仅 {effective_rank} 个独立，"
            "结构存在可动机构或零能变形模态（约束冗余 / 机构可动）"
        )

    # ---- 第三步：Cholesky 求解缩放系统（失败退回最小二乘） ------------
    p_hat = p64 * scale
    try:
        chol = np.linalg.cholesky(k_hat)

        def solve_hat(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))

    except np.linalg.LinAlgError:

        def solve_hat(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.lstsq(k_hat, rhs, rcond=None)[0]

    d = solve_hat(p_hat) * scale  # 回到物理坐标

    # ---- 第四步：物理坐标迭代精化（扩展精度残差） ----------------------
    k_ex = np.asarray(k_sym, dtype=_EXTENDED)
    p_ex = np.asarray(p_f, dtype=_EXTENDED)
    prev_residual = np.inf
    for _ in range(_MAX_REFINE_STEPS):
        r_ex = p_ex - k_ex @ d.astype(_EXTENDED)
        residual = float(np.max(np.abs(r_ex)))
        if not np.isfinite(residual) or residual == 0.0 or residual >= prev_residual:
            break
        prev_residual = residual
        r64 = np.asarray(r_ex, dtype=float)
        d = d + solve_hat(r64 * scale) * scale

    # ---- 第五步：残差复核（后向误差准则） ------------------------------
    if not np.all(np.isfinite(d)):
        raise SingularMatrixError("位移解中出现非有限值（NaN 或无穷），判定为奇异矩阵")

    # 残差按 |K|·|d| 与 |P| 的尺度归一化：float64 解向量自身的表示舍入
    # 就会产生 ~eps·|K|·|d| 的残差下限（N·mm 下可达 1e-4 量级的力），
    # 用 |P| 归一会把精确解误判为病态；后向误差 ~eps 才是可解的判据
    r_ex = p_ex - k_ex @ d.astype(_EXTENDED)
    residual = float(np.max(np.abs(r_ex)))
    k_inf = float(np.max(np.sum(np.abs(k_ex), axis=1)))  # |K|_∞
    scale_ref = max(float(np.max(np.abs(p_ex))), k_inf * float(np.max(np.abs(d))))
    if scale_ref > 0.0 and residual > _RESIDUAL_TOL * scale_ref:
        raise SingularMatrixError(
            "方程残差超出容差，刚度矩阵病态或近奇异，结果不可信"
        )

    return d
