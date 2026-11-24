# 最优控制专家与原 Flow Planner 闭环对比

本目录只实施第一阶段：在真实 nuPlan `81f400b414215325-s0` 场景中，将自车初速度设为
15 m/s 进行压力测试。15 m/s 是实验设定，不是原日志速度。

专家使用 CasADi/Ipopt 直接多重射击求解带道路、完整车身避障近似、转向、转向速率、
纵向加速度和合加速度包络约束的最优控制问题。优化控制只用于生成带时间信息的后轴状态
参考轨迹；实际执行必须经过与原 Flow Planner 相同的 nuPlan LQR 和 OCD。优化器内的运动学
自行车可行性不等同于 OCD 双轨 MF5.2 闭环可行性。

实施顺序：

1. `benchmark_solver.py` 对少量状态做冷启动求解，统计平均、P95、最大耗时和失败率。
2. 在相同场景中比较 0.1、0.5、1.0 s 专家重规划周期；LQR 和 OCD 始终以 0.1 s 执行。
3. 冻结重规划周期后，再与官方 Flow Planner 组做正式闭环对比和可视化。

本目录不包含动力学条件注入、训练或监督微调。

## 当前冻结结果

冷启动 5 次求解的平均、P95、最大耗时约为 0.599、0.599、0.599 s，均成功。
本阶段用于离线专家数据生成，不要求实时截止；0.1 s 组相对 1.0 s 组的总墙钟时间只增加
约 1.73 s，因此冻结为 0.1 s。LQR 与 OCD 同样按 0.1 s 执行。停车点进入承诺区后保持
最近一次有效参考，不继续移动停车目标。

正式对比中，原 Flow Planner 在约 2.70 s 碰撞；0.1 s 最优控制专家通过强制动安全停车，
18 s 内无碰撞、无道路越界。高速右绕参考的早期尝试在 LQR/OCD 中出现跟踪滞后和边界过冲，
因此没有作为有效专家标签。

随后在速度—附着筛选中找到 `12 m/s、内部峰值缩放 1.0` 的绕行区分工况：专家参考经
相同 LQR + OCD 后完成绕行，最小车身间隙约 0.319 m；原 Flow Planner 在完全相同条件下
约 3.30 s 碰撞。专家按每个 0.1 s 控制周期尝试求解，共 181 次，其中 121 次成功、60 次失败；
失败周期继续跟踪最近有效参考。累计求解约 100.0 s，完整闭环墙钟时间约 109.2 s。
正式结果见 [`results_scan/comparison_v12_s1p0/REPORT.md`](results_scan/comparison_v12_s1p0/REPORT.md)。

结果入口：[`results/comparison/REPORT.md`](results/comparison/REPORT.md)，动画为
`results/comparison/flow_vs_expert.gif`。

```bash
cd /home/ljt23/research_ca
01_flow_planner/.venv/bin/python 06_optimal_control_expert/benchmark_solver.py
01_flow_planner/.venv/bin/python 06_optimal_control_expert/run_expert_closed_loop.py --force
01_flow_planner/.venv/bin/python 06_optimal_control_expert/analyze_and_render.py
01_flow_planner/.venv/bin/python 06_optimal_control_expert/verify_results.py
```
