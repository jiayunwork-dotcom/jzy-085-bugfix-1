"""平面刚架直接刚度法核算服务（planar rigid-frame solver）。

模块划分（各管一段、互不重叠）：

- geometry.py  方向余弦与坐标变换
- element.py   局部坐标系下带轴向 + 弯曲的六阶单元刚度、分布荷载等效节点力
- precision.py 补偿精度（double-double）运算
- assembly.py  总刚组装（float64 / 补偿精度两版）
- constraints.py  自由度分类（受约束 / 自由），本实现统一采用“划行划列”法
- solver.py    对角均衡、SVD 判秩、Cholesky 与混合精度精化、残差复核
- forces.py    杆端内力与支座反力回代
- validation.py  输入校验（编号、拓扑、几何、材料、连通性、刚体约束）
- models.py    HTTP 输入 / 输出的 Pydantic 数据模型
- analysis.py  把上述各段串成一次完整核算
- main.py      唯一的 HTTP 入口（FastAPI）
"""

from .analysis import analyze_frame

__all__ = ["analyze_frame"]
