# nuPlan 静态障碍物最小闭环实验

本目录只包含一个真实 nuPlan 场景的两组对照：原始场景，以及在同一初态前方加入固定 `BARRIER`。两组都使用 `01_flow_planner` 中的官方 Flow Planner 权重、发布采样设置、nuPlan LQR、运动学自行车模型和非反应式日志回放交通。

## 一条命令复现

需要在能访问 RTX 4090 的宿主机环境运行：

```bash
cd /home/ljt23/research_ca/04_nuplan_static_obstacle
bash run_all.sh
```

该命令不会训练或修改上游源码；它依次重跑两组闭环并从新日志生成动画和对比图。场景与障碍物参数集中在 `config.yaml`。

## 文件

- `config.yaml`：场景 token、官方权重路径、随机种子、采样和障碍物几何。
- `run_experiment.py`：场景加载、障碍物注入、Flow Planner 重规划、LQR 跟踪、车辆状态传播和完整日志。
- `render_results.py`：从保存的实际日志生成 GIF、轨迹图和速度图。
- `results/*/ego_states.csv`：实际执行状态和碰撞状态。
- `results/*/reference_trajectories_global.npz`：每次重规划的完整全局参考轨迹。
- `results/*/lqr_controls.csv`：LQR 加速度和前轮转角变化率。
- `results/obstacle/model_input_audit.json`：每次调用实际送入模型的 5 个静态物体槽位。

实验结果和解释见 `REPORT.md`。
# Wide-lane replacement experiment

The replacement requested after the first narrow-lane run is kept separately and
does not overwrite the original artifacts.  It uses real nuPlan token
`00513119374c5db4` on a straight three-lane Las Vegas road and records the scene
edit in an interPlan-style manifest.

Run everything with:

```bash
./run_all_wide_lane.sh
```

Main inputs and report:

- `config_wide_lane.yaml`: executable scene, planner, controller, and obstacle settings.
- `interplan_modification_wide_lane.yaml`: declarative interPlan-style modification record.
- `scenario_selection_wide_lane.json`: pre-run geometry and traffic audit.
- `REPORT_WIDE_LANE.md`: outcome and explanation.

The original `config.yaml`, `REPORT.md`, and `results/` are retained unchanged.

## mini stopped-vehicle nudge experiment

The small interPlan-style nudge experiment uses a zero-speed dynamic `VEHICLE`
on the same local mini road network.  It retains both failed placements and the
final collision-free closed-loop result.

Run the final baseline/obstacle pair with:

```bash
bash /home/ljt23/research_ca/04_nuplan_static_obstacle/run_all_interplan_nudge.sh
```

Inputs and outputs:

- `config_interplan_nudge_final.yaml`: final executable configuration.
- `scenario_selection_interplan_nudge.json`: scene geometry and adjustment record.
- `results_interplan_nudge_final/`: logs, verification, animations, and plots.
- `REPORT_INTERPLAN_NUDGE.md`: concise outcome and interpretation.

