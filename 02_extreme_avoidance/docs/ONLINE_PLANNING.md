# 在线规划基线与实验集

## 实现

`rolling` 是固定格点候选的滚动时域搜索。每 100 ms 从当前观测重新预测 5 s，筛选转向—制动反馈机动，只执行所选机动的下一个 20 ms 控制请求；两次搜索之间也根据最新状态计算反馈控制。一次候选筛选积分步长为 10 ms，预测反馈与实际反馈都在 20 ms 更新，使用 C++ 批量循环降低调用开销。

候选参数依次为 `[signed_offset_m, outward_ramp_m, brake_N, brake_end_s, return_length_m, stop_flag]`。首版枚举左右方向、2.6/3.0/3.4 m 偏移、两种建立距离、0/1200/2400 N 制动，共 36 个绕行候选，另加一个中心线制动候选和上一周期的候选。因此第一次搜索 37 个，随后最多 38 个。制动释放位置和回归距离当前固定为障碍中心与 35 m；它们不是本版额外搜索维度。没有宣称连续空间最优性。

与旧离线模板采样的关键差别是：每次搜索从实测新状态、当前执行器状态和待执行命令开始，重新比较候选，而不是从场景起点运行一次并固定到结束。当前规划器也会筛查所有已揭示静态障碍，但机动参考仅围绕首次揭示的最近障碍构造，因此实验范围限定为单障碍。曲路输入会明确报错。

## 预测、筛选与恢复

原生三自由度自行车包含纵横向与横摆、转向和纵向执行器，8 维状态为 `[x,y,yaw,vx,vy,r,delta,force]`。质量、惯量、轴距、制动分配及执行器参数来自同一车辆配置。前后轴小侧偏刚度由冻结 MF52 在静态轴荷下的零点斜率推导；有效峰值系数由参考轮胎台架的约 1.46/1.18 倍率映射。模型采用 tanh 横向力、摩擦椭圆、纵向载荷转移和简化空气动力学，不包含轮速、轮胎松弛、侧倾和 ABS 周期。制动饱和系数 0.9 是工程近似，未通过数据拟合。

候选按车身尺寸与预测步间保守扫掠检查道路/障碍，另限制预测侧偏和航向；所有控制请求共用幅值、速率和延迟。筛选代价包括横向偏移、侧偏、横摆、终端车道偏差和航向、制动成本及进度。终端恢复是软目标，最终是否连续恢复仍由 OCD 闭环评价判定，预测通过不是物理安全证书。

转向/制动反馈完成障碍前后机动后，只在车道偏差、航向、侧偏和横摆同时较小时允许正驱动力，且限为 1000 N。修复后的 `joint` 同样使用恢复门控；旧控制器通过 `joint_legacy` 保留。此门控不保证任何初始状态都能恢复，低速滞留仍可能超时。

无预测可行候选时立即请求 AEB；100 ms 搜索预算超时则丢弃结果并请求 AEB。AEB 是固定应急动作，不保证低附着或侧偏状态下安全。无解和回退次数均记录。

## 时序和计时边界

规划周期 100 ms、反馈周期 20 ms、模型执行延迟 40 ms。仿真采用同步控制调用，不模拟计算期间车辆继续运动，也没有硬实时调度。`planner_*` 计时包括候选构造、展开、碰撞检查、排序和动作生成；`controller_*` 还包括观测组装和候选日志整理。两者均不包括 OCD 推进、图形生成或写盘。

必须同时查看规划 100 ms 截止时间与控制调用的 20 ms 超时率，不能只用包含大量廉价反馈调用的总体 P95 掩盖搜索开销。若计算超时则记录并回退；当前测试时间通过也不能证明硬实时保证。

## 数据与复现

```bash
cd /home/ljt23/research_ca/02_extreme_avoidance
bash scripts/build.sh
.venv/bin/python -m extreme_avoidance run --scenario straight_online --controller rolling
.venv/bin/python scripts/run_benchmark.py --output artifacts/my_online_benchmark
.venv/bin/python scripts/validate_planning.py --output artifacts/my_planning_validation
.venv/bin/python -m unittest discover -s tests -v
node scripts/check_replays.cjs artifacts/my_online_benchmark
```

场景矩阵由 `configs/benchmark.json` 冻结：15/20/25 m/s × 内部倍率 0.5/0.8 × 揭示距离 20/30/45 m，共 18 个基础场景。AEB、修复的 joint 各 18 条，rolling 左右优先各 18 条，共 72 条闭环记录。镜像优先仅改变等价候选排序，允许两侧搜索，不代表方向被强制锁定。

每条记录含 `trace.csv`、`metrics.json`、`config.json`、`manifest.json` 和 `replay.html`。rolling 另保存 `plans.jsonl`，每次规划记录全部候选参数、预测可行性、代价、间隙和终端状态摘要、选择与回退。每个候选的 prediction 列依次为 `[feasible, score, minimum_gap_m, road_margin_m, max_beta_rad, final_d_m, final_yaw_rad, final_vx_mps, final_x_m]`。预测失败候选可能提前终止，其终值是终止时摘要，不是完整 5 s 结果。

这些预测候选没有全部用 OCD 逐条复核，不能直接标为合格训练示范。闭环执行轨迹中 `success=true` 且无约束违反的记录，才是当前模型和恢复时域内的有效机动。所有控制器、镜像分支、邻近扰动须按基础场景 `group_id` 分组；目前未建立训练/测试划分。未来可以用 Transformer＋CFM 替换 `_candidates()`，保留同一执行器和检验接口。

有限搜索未成功标为 `unknown`；AEB 失败但其他方法成功只表明找到非 AEB 解，不证明物理上必须转向。该实验集有意包含容易与困难场景，成功率仅描述这个固定矩阵。

## 当前验证范围

内部缩放物理检查、1/2/5 s 预测误差及原始轨迹见 `artifacts/planning_validation_v3/`。12 组开环输入中，最大位置误差分别约 0.098/0.405/3.520 m。5 s 长时域误差明显，0.25 m 几何余量是工程启发式，不能覆盖所有误差。滚动反馈可以减少实际偏离，但不消除模型失配。

后续优先工作是分析临界失败与预测假阳性、扩展短时域误差统计、增加更强采样/局部优化对照，再生成经过 OCD 独立检验的离线数据。当前阶段没有生成模型训练、真实路面辨识、CarSim 或实车验证。
