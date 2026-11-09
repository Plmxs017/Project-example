# AI 最优路线：设计与实现说明

本文说明「AI 最优路线」按钮背后的算法、数据结构、接口与执行流程，方便答辩时讲清楚每个数字是怎么来的。

相关代码：

| 文件 | 作用 |
| --- | --- |
| `backend/algo/route_planner.py` | 纯算法：最短路矩阵、访问顺序优化、结果组装 |
| `backend/sim/world.py` | 目标点收集、路线转车道折线、AI 任务状态机 |
| `backend/main.py` | `/api/ai/route`、`/api/ai/execute`、`/api/ai/cancel` |
| `backend/static/index.html` | 路线光带、编号站点、AI 面板、小助手播报 |
| `backend/tests/test_route_planner.py` | 算法测试（含与暴力枚举对拍） |
| `backend/tests/test_ai_route.py` | 仿真内执行测试 |

## 1. 问题定义

给定：

- 车辆当前位置（取最近的路网节点作为起点）；
- 一组待处理目标点（每个火点的道路停靠节点；没有火情时是 12 个巡检点）；
- 城市车行图 `adj`（无向加权图，边权是欧氏长度）；

求：一条从当前位置出发、**经过所有目标点一次**、总行驶距离最短的路线。

这是**开放路径**的旅行商问题（Open-TSP）：不要求回到起点，因此比经典 TSP 少一条边。

## 2. 三层求解

### 2.1 最短路层：A* + 距离矩阵

- 单源最短路使用 Dijkstra（`shortest_distances`）一次算出起点到全图所有节点的距离；
- 目标点之间用 `graph_astar`（欧氏距离启发）重建具体路径，用于前端画线和车辆行驶；
- 结果缓存成 `matrix[(a, b)]`，避免重复搜索。目标点规模 ≤ 13，矩阵构建是一次性的毫秒级操作。

### 2.2 顺序层：四种策略对比

| 策略 | 思路 | 复杂度 | 说明 |
| --- | --- | --- | --- |
| 原始顺序 | 按火情出现先后 | O(n) | 人的直觉，通常绕路 |
| 最近邻贪心 | 每次去最近的未访问点 | O(n²) | 快，但会「先近后远」把自己绕进去 |
| 最近邻 + 2-opt | 贪心后用「翻转一段」改良 | O(n²) ~ O(n³) | 近似解，通常已经很接近最优 |
| 状态压缩 DP | Held-Karp，枚举子集与最后访问点 | O(n²·2ⁿ) | n ≤ 10 时求**精确最优** |

2-opt 的直觉：如果路线里存在「A → B … C → D」两段交叉或绕远，把中间一段反转成「A → C … B → D」往往更短；反复尝试直到没有改进。

DP 的状态：`dp[(mask, last)]` = 已访问集合 `mask`、最后停在 `last` 的最小里程；转移时枚举下一个点。最后从满集合里取最小值即可回溯出顺序。

### 2.3 解释层：把结果讲清楚

返回结构里同时包含：

- `stops`：每一站的序号、坐标、名称、距今累计里程；
- `legs`：每一段的距离与预计用时；
- `algorithm.candidates`：四种方案的总里程，前端会显示成「原始 1650 · 贪心 1190 · 2-opt 1190 · DP 1190」；
- `saved_distance` / `saved_percent`：相对原始顺序节省的里程和百分比；
- `polyline`：把节点路径按「靠右车道」偏移后的折线，直接用于 Canvas 绘制。

> 目标点超过 10 个时自动退化为 2-opt 近似解，`algorithm.exact` 会变成 `false`。

## 3. 接口

### `POST /api/ai/route`

只规划、不改车辆状态。

```json
{
  "ok": true,
  "mode": "fire",
  "mode_label": "多火情最优处置路线",
  "stops": [
    {"order": 1, "node": 11, "x": 170.0, "y": 250.0, "kind": "fire",
     "label": "火情 #2", "fire_id": 2, "intensity": 100,
     "leg_distance": 280.0, "distance_so_far": 280.0}
  ],
  "legs": [{"from": 4, "to": 11, "distance": 280.0, "eta": 3.4}],
  "path_nodes": [4, 5, 11],
  "total_distance": 1190.0,
  "baseline_distance": 1650.0,
  "saved_distance": 460.0,
  "saved_percent": 27.9,
  "eta": 22.4,
  "algorithm": {
    "name": "状态压缩 DP（精确最优）",
    "shortest_path": "A*（欧氏距离启发）",
    "optimizer": "状态压缩 DP",
    "exact": true,
    "targets": 3,
    "candidates": [
      {"name": "原始顺序", "distance": 1650.0, "order": [11, 16, 20]},
      {"name": "最近邻贪心", "distance": 1190.0, "order": [11, 20, 16]},
      {"name": "最近邻 + 2-opt", "distance": 1190.0, "order": [11, 20, 16]},
      {"name": "状态压缩 DP（精确最优）", "distance": 1190.0, "order": [11, 20, 16]}
    ]
  },
  "polyline": [[154.0, 300.0], [170.0, 316.0]],
  "blocked_nodes": [5],
  "dropped": [],
  "explanation": [
    "识别到 3 个目标点，先用 A* 算出两两最短路，再用 状态压缩 DP 决定访问顺序。",
    "最优总里程 1190，按发现顺序要跑 1650，节省 460（27.9%）。",
    "全程 3 站，预计 22 秒跑完。"
  ]
}
```

失败时返回 `{"ok": false, "reason": "..."}`，例如自检进行中、没有目标点、目标点全部不可达。

> `blocked_nodes` 是被不可移开的重障封锁、规划时直接剔除的路口；`dropped` 是被完全封死、本次无法前往的目标名称。
> 规划使用裁剪后的路网邻接表（`_nav_adj`），因此被封锁的路口不会出现在 `path_nodes` 与 `polyline` 中。

### `POST /api/ai/execute`

重新规划一次并接管车辆：

```json
{"ok": true, "mode": "fire", "mode_label": "多火情最优处置路线",
 "stops": 3, "total_distance": 1190.0, "eta": 22.4,
 "algorithm": "状态压缩 DP（精确最优）"}
```

### `POST /api/ai/cancel`

```json
{"ok": true}
```

## 4. 执行状态机

AI 接管后车辆进入 `ai_mission`：

```text
ai_mission ──到达火点──> extinguishing ──扑灭──> ai_mission
     │                                        │
     └──到达巡检点──> ai_scan（1.4s）──────────┘
     └──全部完成──> idle
```

- 每一站都用 A* 重新算一次「当前位置 → 站点」的路线，所以中途被行人、路障影响后仍能正确到达；
- 与原有逻辑共用：行人降速、机械臂清障、到达判定；
- **FOV 反馈 / 新火情**：视野发现重障、行人占道或路口拥堵且命中剩余路线时，从车辆当前位置重排剩余站点（2 秒冷却），任务不中断；重排次数写在 `ai.replans`，原因写在 `ai.replan_reason`；
- 被重障完全封死、暂时到不了的目标会挂起并在结束时报告（`ai.blocked_targets` / `ai.deferred`），不谎报完成；
- 中止条件：水量耗尽（转补水）、注入故障（转维修）、手动自检、用户点「停止」；
- `get_state()["ai"]` 会返回 `active / index / total / visited / completed / replans / blocked_targets / stops / polyline`，前端据此画进度与重规划提示。

## 5. 小助手 MOMO

- 立绘 `backend/static/assets/ai-assistant.png` 由原始 PNG 去白底生成（透明背景）；
- CSS 动画：`float` 漂浮、`breathe` 呼吸光晕、`hop` 点击跳跃、`thinking` 规划时加速摆动 + 星星闪烁；
- 气泡采用打字机效果逐字显示，规划 / 执行 / 完成 / 被点击时展示不同台词；
- 点击人物会随机说一条「路径规划小知识」，是纯前端交互，不消耗后端资源。

## 6. 测试与复现

```bash
cd backend
python -m pytest tests -q
```

- `test_route_planner.py`：DP 与全排列暴力枚举对拍、2-opt 不劣于贪心、不可达目标剔除、重复目标去重；
- `test_ai_route.py`：规划覆盖全部火点、无火情时转巡检、执行后火情清零、取消恢复待命、自检中拒绝接管、心跳字段完整；
- `test_fov_ai.py`：重障不被机械臂清理、规划避开重障节点、FOV 重障反馈触发 AI 重排且任务不中断、新火情并入路线、人行道行人不逼停、拥堵反馈与自行疏解。

## 7. 已知限制

- 目标点数量 ≤ 10 才能保证精确最优，超过后退化为 2-opt 近似解；
- 暂未把「水量 / 故障」纳入代价函数，执行中会中断并交回原有调度（安全策略，不属于 FOV 冲突）；
- FOV 反馈会把「重障 / 行人 / 拥堵」近似成道路节点封锁，不做精确的边分段；重障封死的是整个路口。
- ETA 是按有效均速 82 单位/秒 + 每站 2.5 秒的粗略估计，未考虑红绿灯等待。
