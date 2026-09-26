"""线性方程求解与奇异性判定。

目标：一副结构能不能解、解出来是多少，只取决于结构本身——与用户
选择哪套自洽单位（N·mm 还是 kN·m）、一根杆切成几段无关；同时真正
退化的矩阵（机构、零能模态）必须一如既往地报 SINGULAR_MATRIX，
绝不返回 NaN 或不可信的数。

为此求解全程遵循以下管线（各步理由见行内注释）：

0. **对称对角均衡**（Jacobi 均衡）：刚度矩阵混合平动自由度
   （量纲 力/长度）与转动自由度（量纲 力·长度），直接对原矩阵做
   奇异值判定，结论会随单位制与网格疏密漂移。先把矩阵变换为
       K̂ = D⁻¹ K D⁻¹,  p̂ = D⁻¹ p,  D = sqrt(diag(K))
   均衡矩阵在任意对角合同变换（单位换算正是此类变换）下保持不变，
   对角元全为 1，其谱只反映结构自身的力学性态。均衡用补偿精度
   （precision.DD）执行，不引入新的舍入；
1. **判秩**：对均衡矩阵的 float64 主项做奇异值分解，以
   ``最小奇异值 < 1e-12 × 最大奇异值`` 判为秩亏损。均衡后合法结构
   的条件数只取决于力学性态（实测 300 段悬臂约 4×10¹⁰），而精确
   秩亏矩阵的最小奇异值是机器噪声量级（~1e-16 相对），1e-12 位于
   两者之间，两侧都留有数量级余量；
2. **求解**：float64 Cholesky 分解（K̂ 理论上对称正定）得初始解，
   再做**混合精度迭代精化**：残差用 double-double 补偿精度计算
   （与条件数无关的 ~1e-32 精度），修正量用同一组 float64 因子回代。
   细网格下条件数可达 1e10 量级，单纯 float64 三角回代的前向误差
   会被放大到 1e-6 边缘；精化把前向误差压回 1e-12 量级以下，
   且行为与平台无关（不依赖 longdouble 等硬件相关类型）。
   Cholesky 失败则退回最小二乘，由残差复核兜底；
3. **残差复核**：在均衡后的无量纲系统上做**逐行（分量式）后向误差**
   检查。第 i 行的后向误差 |r_i| / (|K̂||ŷ| + |p̂|)_i 回答的是
   “算出的解是否精确满足某个与 K、p 仅差机器精度级扰动的方程”，
   是后向稳定求解的标准判据；后向误差过大一律按奇异 / 病态处理。

   为什么不用“残差最大元 / 荷载最大元”的全局归一化：像端部单点
   加载的长悬臂，内部自由度荷载为零但位移很大，其舍入残差约为
   条件数 × 机器精度，会被全局尺度归一化放大成假阳性（同一副结构
   换单位、多切几段就被误判“不可信”）。
"""

from __future__ import annotations

import numpy as np

from .errors import SingularMatrixError
from .precision import (
    DD,
    as_dd,
    dd_div,
    dd_matvec,
    dd_outer,
    dd_sqrt,
    dd_sub,
)

# 秩亏损的相对奇异值阈值（作用于对角均衡后的矩阵，见模块 docstring）
_RANK_TOL = 1e-12
# 逐行后向误差阈值（均衡后的无量纲系统上计算）
_RESIDUAL_TOL = 1e-8
# 混合精度迭代精化的最大次数（实测 2~3 次即收敛到 float64 表示极限）
_REFINEMENT_MAX_ITERS = 4
# 精化收敛判据：修正量相对位移不超过此值（float64 可表示极限）
_REFINEMENT_CONVERGENCE = 1e-16


def _check_finite(matrix: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(matrix)):
        raise SingularMatrixError(f"{label}中出现非有限值（NaN 或无穷），无法求解")


def solve_system(k_ff, p_f) -> np.ndarray:
    """求解 K_ff d_f = P_f；奇异时抛 :class:`SingularMatrixError`。

    ``k_ff`` / ``p_f`` 可以是普通 float64 数组（直接调用、测试），
    也可以是补偿精度 :class:`~framesolver.precision.DD`（analysis 的
    标准路径，组装误差已被消除）；两种输入走同一条求解管线，
    返回值统一为 float64 位移向量。
    """
    if isinstance(k_ff, DD):
        k_dd, p_dd = k_ff, p_f
    else:
        k_dd = as_dd(np.asarray(k_ff, dtype=float))
        p_dd = as_dd(np.asarray(p_f, dtype=float))
    _check_finite(k_dd.hi, "缩聚刚度矩阵")
    _check_finite(p_dd.hi, "荷载向量")
    n = k_dd.hi.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    # 强制对称，消除组装中可能残留的量级 1e-16 的不对称
    k_dd = DD(
        0.5 * (k_dd.hi + k_dd.hi.T),
        0.5 * (k_dd.lo + k_dd.lo.T),
    )

    # ---- 第零步：补偿精度的对称对角均衡 ---------------------------------
    diag_hi = np.diag(k_dd.hi).copy()
    if np.any(diag_hi <= 0.0):
        # 正定刚度矩阵的对角元必为正；非正对角元意味着该自由度
        # 完全不受任何刚度约束（零能模态），是货真价实的机构
        raise SingularMatrixError(
            "缩聚刚度矩阵存在非正对角元：有自由度完全不受任何刚度约束，"
            "结构存在可动机构或零能变形模态"
        )
    d_dd = dd_sqrt(DD(diag_hi, np.diag(k_dd.lo).copy()))
    k_hat = dd_div(k_dd, dd_outer(d_dd))  # K̂ = D⁻¹ K D⁻¹
    p_hat = dd_div(p_dd, d_dd)            # p̂ = D⁻¹ p

    # ---- 第一步：SVD 判秩（在均衡矩阵的 float64 主项上） -----------------
    k_hat_sym = 0.5 * (k_hat.hi + k_hat.hi.T)
    singular_values = np.linalg.svd(k_hat_sym, compute_uv=False)
    s_max = singular_values[0]
    s_min = singular_values[-1]
    if not np.isfinite(s_min) or s_max == 0.0 or s_min < _RANK_TOL * s_max:
        effective_rank = int(np.count_nonzero(singular_values >= _RANK_TOL * s_max))
        raise SingularMatrixError(
            f"缩聚刚度矩阵奇异：{n} 个自由自由度中仅 {effective_rank} 个独立，"
            "结构存在可动机构或零能变形模态（约束冗余 / 机构可动）"
        )

    # ---- 第二步：float64 Cholesky + 补偿精度残差的迭代精化 ---------------
    try:
        chol = np.linalg.cholesky(k_hat_sym)

        def _chol_solve(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))

        y = _chol_solve(p_hat.hi)
        for _ in range(_REFINEMENT_MAX_ITERS):
            # 残差以 double-double 精度计算（与条件数无关），
            # 修正量用同一组 float64 Cholesky 因子回代
            r = dd_sub(p_hat, dd_matvec(k_hat, y))
            delta = _chol_solve(r.hi + r.lo)
            y = y + delta
            y_scale = float(np.max(np.abs(y)))
            if y_scale == 0.0 or float(np.max(np.abs(delta))) <= (
                _REFINEMENT_CONVERGENCE * y_scale
            ):
                break
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(k_hat_sym, p_hat.hi, rcond=None)[0]

    # ---- 第三步：逐行后向误差复核（均衡系统即无量纲系统） ---------------
    if not np.all(np.isfinite(y)):
        raise SingularMatrixError("位移解中出现非有限值（NaN 或无穷），判定为奇异矩阵")

    # 对最终解重算补偿精度残差（精化循环内的残差对应更新前的解）
    residual_dd = dd_sub(p_hat, dd_matvec(k_hat, y))
    residual = residual_dd.hi + residual_dd.lo
    # 分母为零的行与解完全解耦（无荷载、位移恒为零），残差也必为零，跳过
    row_scale = np.abs(k_hat.hi) @ np.abs(y) + np.abs(p_hat.hi)
    active = row_scale > 0.0
    if bool(np.any(active)) and float(
        np.max(np.abs(residual[active]) / row_scale[active])
    ) > _RESIDUAL_TOL:
        raise SingularMatrixError(
            "方程残差超出容差，刚度矩阵病态或近奇异，结果不可信"
        )

    # 还原到原坐标：d = D⁻¹ ŷ（float64 表示误差 ~1e-16，远小于验收容差）
    d_f = y / d_dd.hi
    if not np.all(np.isfinite(d_f)):
        raise SingularMatrixError("位移解还原后出现非有限值（NaN 或无穷），判定为奇异矩阵")
    return d_f
