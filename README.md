# 平面刚架受力核算服务（framesolver）

一个只做一件事的核算服务：喂一副平面刚架的几何 / 截面 / 荷载 JSON，
用**直接刚度法**解出节点位移，再回代出每根杆的轴力、剪力、弯矩和各支座反力。
不做画图、不做进度台账，不设账户体系；对外只有一个 HTTP 入口 `POST /solve`。

- 语言：Python 3.12（锁定）
- Web 框架：FastAPI
- 线性代数：NumPy

---

## 一条命令跑起来

### 方式一：Docker Compose（推荐）

```bash
docker compose up --build -d
# 服务在 http://localhost:8000
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d @examples/portal_frame.json
```

### 方式二：Docker

```bash
docker build -t frame-solver .
docker run --rm -p 8000:8000 frame-solver
```

### 方式三：本地 Python（3.12）

```bash
pip install -r requirements.txt
uvicorn framesolver.main:app --host 0.0.0.0 --port 8000
```

交互式接口文档（FastAPI 自带，仅用于调试）：`http://localhost:8000/docs`。

### 跑测试

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 接口

### `POST /solve`

请求体：

```json
{
  "nodes": [
    {"id": "1", "x": 0.0, "y": 0.0, "restraints": [true, true, true]}
  ],
  "members": [
    {"id": "m1", "node_i": "1", "node_j": "2",
     "elastic_modulus": 2.1e8, "area": 2.0e-2, "inertia": 8.0e-5}
  ],
  "nodal_loads": [
    {"node_id": "2", "fx": 10.0, "fy": 0.0, "moment": 0.0}
  ],
  "distributed_loads": [
    {"member_id": "m1", "load_type": "uniform_transverse", "qy": -3.0}
  ]
}
```

- 每个节点 3 个自由度：水平 `u`、竖向 `v`、转角 `θ`；
  `restraints` 按此顺序给布尔值，`true` = 锁死。
  固定端 `[t,t,t]`、铰支座 `[t,t,f]`、水平向滚动支座（仅竖向链杆）`[f,t,f]`。
- 杆件带弹性模量、截面积、惯性矩（都必须为正）。
- 荷载两类：
  - 节点集中力 / 力矩（整体坐标，`moment` 逆时针为正）；
  - 杆件满跨、**垂直于杆轴**的均布荷载（局部坐标 `qy`，沿局部 +ȳ 为正，重力向下给负值）。

成功响应 `200`：

```json
{
  "success": true,
  "displacements": [{"node_id": "1", "ux": 0.0, "uy": 0.0, "theta": 0.0}],
  "member_forces": [{
    "member_id": "m1", "node_i": "1", "node_j": "2",
    "end_i": {"axial": 0.0, "shear": 0.0, "moment": 0.0},
    "end_j": {"axial": 0.0, "shear": 0.0, "moment": 0.0},
    "axial_force": 0.0
  }],
  "reactions": [{"node_id": "1", "fx": 0.0, "fy": 0.0, "moment": 0.0,
                 "restrained": [true, true, true]}]
}
```

失败响应（HTTP 400 / 422）：

```json
{"success": false, "error": {"code": "SINGULAR_MATRIX", "message": "……可读说明……"}}
```

| code | HTTP | 含义 |
|---|---|---|
| `DUPLICATE_NODE_ID` | 400 | 节点编号重复 |
| `MEMBER_NODE_NOT_FOUND` | 400 | 杆件端点指向不存在的节点 |
| `ZERO_LENGTH_MEMBER` | 400 | 杆长为零（两端坐标重合） |
| `INVALID_PROPERTY` | 400 | E / A / I 非正数，或坐标、荷载不是有限数 |
| `INVALID_LOAD_REFERENCE` | 400 | 荷载挂在不存在的节点 / 杆件上 |
| `DISCONNECTED_STRUCTURE` | 400 | 结构被分成互不相连的几块，或存在孤立节点 |
| `INSUFFICIENT_SUPPORT` | 400 | 约束不足以消除 3 个整体刚体运动 |
| `SINGULAR_MATRIX` | 422 | 缩聚刚度矩阵奇异（机构 / 零能模态） |
| `INVALID_REQUEST` | 422 | 请求 JSON 不符合 Schema（字段缺失、类型错等） |

任何情况下都不会返回 NaN，也不会悄悄丢弃杆件：出错即结构化报错。

---

## 求解方法（直接刚度法）

1. **单元刚度**（`element.py`）：每根杆在局部坐标下形成同时含轴向与弯曲的
   6 阶刚度矩阵 `K_l`（Hermite 梁单元）。
2. **坐标变换**（`geometry.py`）：局部 x̄ 由 i 端指向 j 端，
   整体刚度 `K_g = Tᵀ K_l T`，`T = diag(R,R)`，
   `R = [[c,s,0],[-s,c,0],[0,0,1]]`。
3. **总刚组装**（`assembly.py`）：按自由度 scatter-add 进总刚，
   节点荷载与均布荷载等效节点力（`[0, qL/2, qL²/12, 0, qL/2, −qL²/12]`）
   叠加进荷载向量。
4. **约束施加**（`constraints.py`）：全程只使用**划行划列法**一种——
   被约束自由度位移置零并从方程中删去，解 `K_ff d_f = P_f`；
   不使用大数罚函数，因而没有罚系数导致的病态问题。
5. **方程求解**（`solver.py`）：SVD 判秩（最小奇异值 < 1e-10 × 最大奇异值
   即判奇异），Cholesky 求解（失败退对称 SVD），并强制残差复核（1e-8）。
6. **内力回代**（`forces.py`）：杆端力严格用**本杆自己的局部刚度乘本杆的
   局部位移**恢复：`f = P_eq − K_l (T d_g)`；不另起弯矩查表。
7. **反力回代**：取受约束行 `R = (K d − P)_r`。

### 符号约定（测试与输出一致）

- 整体坐标 x 向右、y 向上，转角 / 弯矩逆时针为正；
- 杆端力为**杆件作用于节点**的局部坐标六维力；
- `axial_force` 取 i 端轴向分量，**拉力为正**；j 端轴向分量与之等值反号；
- 标量截面弯矩（测试里用于和教材对照）以**下侧受拉为正**。

---

## 手算校核算例（门式刚架）

见 `examples/portal_frame.json`，也是 `tests/test_portal_frame.py` 的算例：

- 两柱底**固定**，柱高 = 梁跨 = L = 4 m，柱梁等截面（E = 2.1×10⁸ kN/m²、
  I = 8×10⁻³ m⁴、A 取 2×10⁴ m² 使轴向变形可忽略）；
- 横梁高度处（左柱顶节点 2）作用水平集中力 P = 10 kN（+x 向）。

按位移法（忽略轴向变形，2 个未知量：层间侧移 Δ、节点转角 θ）的闭式解：

| 量 | 公式 | 数值 | 服务输出（有限 A） |
|---|---|---|---|
| 柱顶侧移 | 5PL³/(84EI) | 2.2676×10⁻⁵ m | 2.2676×10⁻⁵ m |
| 节点转角 | −PL²/(28EI) | −3.4014×10⁻⁶ rad | −3.4014×10⁻⁶ rad |
| 水平反力 | 各 −P/2 | −5.0 kN | −5.0 kN |
| 竖向反力 | ∓3P/7 | ∓4.2857 kN | −4.2857 / +4.2857 |
| 柱底弯矩 | 2PL/7 | 11.4286 kN·m | 11.4286 kN·m |
| 梁端弯矩 | 3PL/14 | 8.5714 kN·m | 8.5714 kN·m |
| 横梁跨中弯矩 | 反弯点，0 | 0 | 0 |

另附梁的经典解测试：两端固定梁均布荷载 q 的固端弯矩 qL²/12、跨中 qL²/24；
简支梁跨中 qL²/8、反力各 qL/2。

---

## 被自动化测试守死的力学恒等式

容差写成具体数值（力 / 弯矩 1×10⁻⁸；与忽略轴向变形的教材解比较用 0.2%
相对容差），见 `tests/`：

1. 全部支座反力 + 全部外荷载（含杆件均布荷载合力），整体坐标
   ΣFx = 0、ΣFy = 0；
2. 对**任一固定点**的合力矩为 0（测试取原点和两个任意点）；
3. 同一根杆两端轴力等值反号（含杆件隔离体六维自平衡）；
4. 每个刚性节点处，交汇各杆杆端弯矩 + 外加节点力矩 + 支座反力矩 = 0
   （水平、竖向力平衡同样逐节点检查）。

测试共 58 个，分布在：

- `test_elements.py`：方向余弦、变换正交性、单元刚度系数、刚体零空间、
  等效荷载合力、组装 scatter-add；
- `test_equilibrium.py`：上述四条力学恒等式；
- `test_portal_frame.py`：门式刚架教材反力 / 位移 / 弯矩与梁经典解；
- `test_validation.py`：全部非法输入类型与奇异性；
- `test_http_api.py`：HTTP 成功 / 各类错误 / 坏 JSON / OpenAPI。

---

## 模块划分

```
framesolver/
├── geometry.py     # 方向余弦与坐标变换
├── element.py      # 六阶局部单元刚度 + 均布荷载等效节点力
├── assembly.py     # 总刚与荷载向量组装
├── constraints.py  # 自由度分类（划行划列法）
├── solver.py       # SVD 判秩 + Cholesky 求解 + 残差复核
├── forces.py       # 杆端内力与支座反力回代
├── validation.py   # 全部输入校验（含连通性、刚体约束）
├── models.py       # Pydantic 输入 / 输出模型
├── analysis.py     # 串联各段完成一次核算
└── main.py         # 唯一 HTTP 入口 /solve
```

各文件互相独立、各管一段，便于阅读与维护。
