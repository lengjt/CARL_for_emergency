# 查看实验结果与动态回放

本轮使用官方权重、不重新训练，完成3个nuPlan mini非反应式闭环场景。综合分数为 **0.971951**（0到1，非成功率）；三个运行均完成，无自车责任碰撞，可行驶区域合规指标均为1。这是最小运行验证，不是论文完整基准复现，也不是极限避障验证。

## 动态画面

动画来自保存的实际仿真日志，叠加nuPlan地图，不是示意动画或相机录像。每个约15秒，以约5帧/秒按仿真时间播放、循环播放；不能据此判断模型实际计算速度。

| 场景 | 综合分数 | 动画 |
| --- | ---: | --- |
| 较高车速 high_magnitude_speed | 0.985075 | [打开GIF](../artifacts/nuplan/replays/high_magnitude_speed.gif) |
| 交通中静止 stationary_in_traffic | 1.000000 | [打开GIF](../artifacts/nuplan/replays/stationary_in_traffic.gif) |
| 静止 stationary | 0.930779 | [打开GIF](../artifacts/nuplan/replays/stationary.gif) |

场景名称是nuPlan数据标签，不代表整个15秒内车辆始终保持该状态。

- 蓝色车辆及轨迹：仿真自车及已经行驶的路径。
- 青色曲线：该帧模型规划的未来轨迹，由闭环控制器跟踪。
- 橙色框：其他车辆；紫色框：行人、自行车；灰色框：其他目标。
- 灰色虚线：数据集里人类驾驶的参考轨迹；路面浅灰虚线为车道中心线。
- 画面随自车平移，坐标轴方向固定为初始车头方向。左上角显示仿真时间和自车速度。

周围交通参与者按照数据集记录运动，不会对模型自车作出反应。这些场景尚不能说明突发障碍、低附着路面或转向—制动联合避障性能。动画当前未绘制信号灯状态，不能仅凭画面评价信号灯遵守情况。

通过SSH工作时，可以用本地编辑器的远程文件浏览器下载GIF后在浏览器中打开，无需在服务器安装桌面或浏览器。上述GIF可直接播放；nuPlan原生交互回放入口为运行目录内的`.nuboard`文件，需要通过NuBoard服务打开，并非独立网页。

## 指标与原始文件

- [实验说明](EXPERIMENTS.md)：实验设置、指标解释、局限性。
- [可读结果汇总](../artifacts/nuplan/mini_summary.json)：每个场景的分数、安全指标和规划耗时。
- [官方指标报告PDF](../artifacts/nuplan/exp/simulation/closed_loop_nonreactive_agents/flow_planner_mini/2026-10-13_21-31-26/summary/summary.pdf)。
- [完整运行目录](../artifacts/nuplan/exp/simulation/closed_loop_nonreactive_agents/flow_planner_mini/2026-10-13_21-31-26)：`runner_report.parquet`为运行状态；`aggregator_metric/`为指标汇总；`simulation_log/`为完整回放记录；`code/hydra/`为实际配置。
- [回放来源与时间记录](../artifacts/nuplan/replays/metadata.json)。每个GIF旁还有一张中间帧PNG预览。

如需重新生成动画，在服务器终端执行：

```bash
cd /home/ljt23/research_ca/01_flow_planner
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/render_replays.py
```

该命令只读取本次可信日志并绘图，不运行训练或新的模型推理。日志保存了CUDA对象，因此当前读取方式需要宿主机GPU可访问。
