# 实验记录（2026-10-13）

本轮不进行训练；所有推理使用同一份作者发布的权重。前半部分记录合成接口实验；2026-10-13晚间数据就绪后，新增了3个真实nuPlan mini非反应式闭环场景，结果见文末。仍不构成完整论文结果复现。

## 已完成

- 官方模型14274384个参数严格加载，未忽略缺失或多余键。
- 官方nuPlan规划适配器构建及权重加载成功，见 `artifacts/environment.json`。
- 仿真配置组合成功，见 `artifacts/nuplan_config_check.log`；这不是实际场景仿真。
- 合成直路接口场景：有限数值、输出尺寸 `[1,1,80,4]` 正确、各设置固定种子重复误差为0。
- 无反向传播、无优化器步骤、无训练检查点产生。

## 机制与成本检查

RTX 4090；Python 3.9.25；PyTorch 2.3.0+cu121；FP32；batch=1；CPU线程数4。每项预热5次、测量30次；前后同步GPU；序列运行。输入只有少量有效目标。

| 推理设置 | 解码器调用/次 | 均值 ms | P95 ms | PyTorch峰值分配 MiB |
|---|---:|---:|---:|---:|
| 发布设置：4步、CFG=1.0、双分支 | 8 | 39.51 | 39.73 | 67.96 |
| 4步、仅条件分支 | 8 | 38.50 | 38.67 | 65.61 |
| 2步、CFG=1.0、双分支 | 4 | 22.04 | 22.10 | 67.96 |
| 4步、CFG=1.8、双分支 | 8 | 39.82 | 39.93 | 67.96 |

每项场景编码器调用均为1次；midpoint每个积分步调用解码器2次。双分支CFG在每次解码中拼接条件与掩码条件，不是把调用计数直接翻倍。

计时包含结构化输入的归一化、编码、ODE生成、片段组装和反归一化；不含真实数据/地图处理、CPU-GPU输入传输、车辆控制与仿真。PyTorch分配统计不包括CUDA上下文和驱动等总占用。30次样本不足以建立严格实时保证。

4步到2步的对照验证了计算量变化，尚不能判断真实场景轨迹质量是否受损。关闭CFG的耗时变化较小，不能据此宣称普遍或显著加速。

## CFG=1 时单分支一致性

同一合成场景、5个相同种子对比双分支与单条件分支，最大绝对差约5.34e-5，满足`atol=1e-3, rtol=1e-4`。这是不同batch运算数值差异范围内的接口检查；后续仍需真实数据验证。记录见 `artifacts/cfg_equivalence.json`。

## 可查看的产物

- `artifacts/synthetic_smoke/report.json`：发布设置结果及逐次耗时。
- `artifacts/synthetic_smoke/trajectories.npz`：5个随机种子的输出轨迹。
- `artifacts/synthetic_smoke/trajectories.png`：同场景不同种子轨迹，明确标注合成场景。
- 其他设置分别保存在`synthetic_no_cfg`、`synthetic_two_steps`、`synthetic_cfg_1p8`目录。
- `checkpoints/SHA256SUMS`：原始权重和配置校验值。
- `artifacts/bootstrap.log`：环境重建脚本与依赖一致性检查结果。

## 真实闭环准备（已于本日晚间完成首轮）

依照`DATA_SETUP.md`准备nuPlan mini数据库和地图后，先运行3个非反应式场景并固定token，检查完整运行报告、碰撞/可行驶区域/进度等指标和轨迹，再按相同场景比较采样设置。需确认真实地图和数据读取链路后，才能把状态更新为“少量场景闭环运行复现完成”。

## 首轮真实闭环结果

使用mini数据、单卡顺序执行、`closed_loop_nonreactive_agents`、官方发布权重、CFG=1.0、4步midpoint、随机种子0。3个场景全部运行完成，无异常；评价使用nuPlan原生指标、LQR跟踪与运动学自行车模型。

| 场景token | 类型 | 综合分数（0–1） | 无责任碰撞 | 可行驶区域合规 | 平均规划调用 ms |
|---|---|---:|---:|---:|---:|
| 55950ed834da5061 | high_magnitude_speed | 0.985075 | 1 | 1 | 238.8 |
| ab08bf421e915319 | stationary_in_traffic | 1.000000 | 1 | 1 | 244.2 |
| 00009841879a5bb9 | stationary | 0.930779 | 1 | 1 | 261.3 |

3场景聚合分数为0.971951。无责任碰撞与可行驶区域合规值1表示相应指标通过。运行成功不自动等于全部驾驶指标满分，应同时查看进度、限速等结果。样本仅3个且为mini子集，不能与论文Val14/Test14分数直接比较，也不是极限避障安全证明。

平均规划调用约239–261 ms，包含真实场景特征/地图处理和模型调用，不是前述合成输入约39.5 ms的同一计时范围；没有据此满足严格实时预算的结论。未开展进一步性能优化。

- 数据校验：`artifacts/data_preparation.json`。
- 可读汇总：`artifacts/nuplan/mini_summary.json`。
- 原始运行目录：`artifacts/nuplan/exp/simulation/closed_loop_nonreactive_agents/flow_planner_mini/2026-10-13_21-31-26`。
- 原始目录中保留`runner_report.parquet`、`aggregator_metric/`、`simulation_log/`、`summary/summary.pdf`和Hydra配置。
- 指标中的规划器名`diffusion_planner`来自Flow Planner官方适配器的`name()`返回值，本次实际使用的是Flow Planner权重。

复跑相同场景：

```bash
cd /home/ljt23/research_ca/01_flow_planner
.venv/bin/python scripts/run_nuplan.py --scenarios 3 --tokens 00009841879a5bb9 55950ed834da5061 ab08bf421e915319
```
