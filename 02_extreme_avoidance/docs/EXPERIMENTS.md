# 实验记录：2026-10-15

以下保留初始平台的历史记录；当前在线基线结果见 [ONLINE_RESULTS.md](ONLINE_RESULTS.md)。旧 `joint` 在当前代码中对应 `joint_legacy`，本文件不是新控制器的结果报告。

本轮从额度中断时保留的 Python＋OCD 代码继续。重新构建原生绑定，补齐避障与恢复的独立指标、reset 执行器状态清理、结果来源哈希、物理检查、回归测试、候选库与离线回放。所有结果均来自本机实际执行。

## 1. 代表场景与完整结果

入口：[实验总览](../artifacts/suite_20261015/index.html)。六类场景共 14 个代表工况，10 个达到各自完成条件，4 个失败；这是固定参数的功能覆盖，不是统计成功率估计。没有删去失败工况。

| 场景 | 控制器 | 终止状态 | 曾完成障碍处理 | 稳定恢复 | 最小道路余量 m |
|---|---|---|---|---|---|
| straight_popup | aeb | stopped_safe | 是 | 是 | 4.0400 |
| straight_popup | steer | recovered | 是 | 是 | 0.9877 |
| straight_popup | joint | recovered | 是 | 是 | 1.0798 |
| straight_critical | aeb | collision | 否 | 否 | 4.0400 |
| straight_critical | steer | offroad | 否 | 否 | -0.0002 |
| straight_critical | joint | recovered | 是 | 是 | 0.6198 |
| straight_low_grip | aeb | collision | 否 | 否 | 4.0400 |
| straight_low_grip | steer | recovered | 是 | 是 | 0.5623 |
| straight_low_grip | joint | offroad | 是 | 否 | -0.0001 |
| curve_grip_drop | cruise | recovered | 不适用 | 是 | 3.9779 |
| curve_obstacle_low_grip | aeb | stopped_safe | 是 | 是 | 3.9995 |
| curve_obstacle_low_grip | steer | recovered | 是 | 是 | 1.1094 |
| curve_obstacle_low_grip | joint | recovered | 是 | 是 | 1.2087 |
| split_grip | cruise | recovered | 不适用 | 是 | 4.0425 |

`straight_critical` 冻结为初速 20 m/s、10 m 路宽、障碍中心 s=45 m、揭示距离 30 m、轮胎力缩放 0.6。纯制动发生碰撞；单独转向越界；转向＋3000 N 制动能够绕行并稳定恢复。这只是该参数组的对照，不意味着联合控制在所有条件下更优。

`straight_low_grip` 冻结为初速 20 m/s、障碍 s=50 m、揭示距离 40 m、缩放 0.5。单独转向成功；联合控制虽然通过障碍，但在回正/加速阶段失稳并越界。轨迹显示，后轮纵向滑移在恢复巡航后显著增大，侧偏与横摆随之增长；这与当前控制器没有牵引力/恢复稳定性约束相符，但尚未用消融实验证明唯一因果。该工况验证了“避开障碍≠全程成功”的独立评价能力。

`curve_grip_drop` 与 `split_grip` 无障碍，检查道路跟踪与逐轮附着区域查询；它们的 `avoidance_success=null`。左右分离附着下的强制动偏航另在独立物理实验中检查，不能把轻负载巡航当作极限制动验证。

## 2. 基本物理与数值检查

[报告](../artifacts/physics_20261015/report.json)：13/13 项通过。原始 CSV 和轮胎台架曲线一并保存。检查阈值见 `scripts/validate_physics.py`；所有阈值都在本轮运行前写入脚本。

| 检查 | 实测结果 |
|---|---|
| 上游源码 | 192 个文件均符合记录的 Git blob SHA |
| 静态轮荷 | 总轮荷 7848 N，与 800×9.81 一致；静止速度为 0 |
| 滑行 | 15 m/s 滑行 6 s 后为 12.767 m/s，横向偏移为 0 |
| 20 m/s 制动，缩放 1.0 | 首次低于 0.1 m/s 的距离 23.561 m |
| 20 m/s 制动，缩放 0.5 | 首次低于 0.1 m/s 的距离 41.008 m |
| 停车保持 | 末段残余速度约 0.030 m/s，未人为置零 |
| 左右镜像转向 | 镜像状态误差约 2.22e-16 |
| 稳态转弯 | 末 1 s 最大 ay−vx·r 误差 3.82e-5 m/s² |
| 转向叠加制动 | 状态有限，最终 vx=8.837 m/s，存在正常横摆响应 |
| 1 ms → 0.5 ms 步长 | 转弯最大位置差 8.85e-5 m，速度差 0.002734 m/s，航向差 2.39e-6 rad |
| 前后轴进入附着区 | 前轴 1.24 s、后轴 1.48 s；左右同步 |
| 左右分离附着制动 | 最大横摆角速度约 0.02399 rad/s |
| 联合滑移与缩放 | 联合滑移降低纵横向力；缩放 0.5 将相同输入力减半 |

轮胎缩放能力图：[PNG](../artifacts/physics_20261015/physics_checks.png) / [PDF](../artifacts/physics_20261015/physics_checks.pdf)。这些是模型内部一致性与数值检查，不是实车精度证据。

临界闭环场景另以 0.5 ms 步长重跑三种控制器：[step_comparison.json](../artifacts/suite_20261015/step_comparison.json)。碰撞、越界、成功三个结论均未改变。终点 s 差异分别约 0.00118、0.03665、0.00180 m；事件终止时刻略有差异，不能把这些终点差直接解释为同一时刻状态误差。

## 3. 多分支候选数据

[候选总结](../artifacts/candidates_20261015/summary.json)：对固定临界场景枚举左右方向 × 2.6/3.0 m 偏移 × 0/1500/3000 N 制动力，共 12 条；其中 4 条通过完整避障与恢复检查，左右方向都有成功分支。

有效候选为 `trial_002`、`trial_005`、`trial_008`、`trial_011`；其余 8 条越界，全部保存。数据为反馈模板执行轨迹，不是开环最优控制解，也不是在线规划性能。场景标为 `verified_feasible` 只表示该模型下已找到可行候选，未成功的场景只能标为 `unknown`，不能据有限搜索断言物理不可行。

## 4. 耗时与验证范围

本机代表工况的纯仿真循环墙钟时间为 0.362–2.774 s/条；观察组装＋简单反馈控制的 P99 为 0.0060–0.0340 ms。候选库 12 条累计仿真循环约 7.656 s。这些时间不包含构建、绘图与写盘，且不是生成规划器或在线优化器的耗时。

- 原生 C++ 绑定重新构建成功，未修改上游方程。
- `python3 -m unittest discover -s tests -v`：9 项通过，覆盖矩形接触/旋转、圆弧投影、扫掠穿越、未揭示障碍碰撞、reset 重现、终止状态、恢复失败、输入合法性与预测器批次一致性。
- `node scripts/check_replays.cjs artifacts/suite_20261015`：14 份回放通过链接、初始化、播放到终点、暂停、拖动和倍速逻辑检查。
- 已查看实际生成的轨迹 PNG。回放 JavaScript 在 Node 的最小 DOM/canvas 环境中执行；当前沙箱无法启动 Snap Firefox，未完成真实浏览器渲染截图验证。

## 5. 局限和下一步

当前阶段已具备可重复运行的 Python＋OCD 最小建模平台。研究上仍须补齐目标车辆参数标定、CarSim/实测对照、停车低速模型、恢复阶段控制约束、预测器误差标定、在线滚动采样/优化及更多工况统计。现有研究车辆与简单控制器的结果不能直接外推到真实乘用车。

建议下一步先围绕低附着恢复失败建立速度/驱动力约束和稳定性反馈，再比较独立预测器与 OCD 的偏差；完成这些之后再扩大候选数据与引入 Transformer＋CFM。
