# Python + Open Car Dynamics 极限避障最小平台

2026-10-15：新增直路在线滚动采样规划基线、逐轮内部 MF52 峰值缩放、恢复阶段驱动力约束及冻结参数实验集。本阶段不训练生成模型。

**先看结果：[在线规划实验集与动态回放](artifacts/online_benchmark_v2/index.html)。** 每次实验保存完整轨迹、结果、参数来源和在线候选决策。回放支持播放、暂停、进度拖动及倍速，无网络依赖。实现与限制见 [ONLINE_PLANNING.md](docs/ONLINE_PLANNING.md)。

旧版固定模板结果保留在 [初始平台实验](artifacts/suite_20261015/index.html)，其中 `joint` 对应当前 `joint_legacy`；历史结果使用后置力缩放。

| 内容 | 入口 |
|---|---|
| 在线基线实测结果、失败解释、验证记录 | [docs/ONLINE_RESULTS.md](docs/ONLINE_RESULTS.md) |
| 初始平台历史实验 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| 模型、坐标、参数、判定与后续训练接口 | [docs/MODEL_AND_INTERFACE.md](docs/MODEL_AND_INTERFACE.md) |
| 物理检查图 | [PNG](artifacts/physics_20261015/physics_checks.png) / [PDF](artifacts/physics_20261015/physics_checks.pdf) |
| 13 项物理与源码检查 | [report.json](artifacts/physics_20261015/report.json) |
| 12 条候选机动，含全部失败 | [index.json](artifacts/candidates_20261015/index.json) / [summary.json](artifacts/candidates_20261015/summary.json) |

## 运行

使用项目本地 `.venv`，基于系统 Python 3.10，并复用 Ubuntu 22.04 已安装且经检查的 NumPy 1.21.5、Matplotlib 3.5.1 二进制包。CMake、C++20 编译器、pybind11、Eigen 和 Boost 仍由系统提供。运行不需要 ROS 进程、GPU 或联网。首次运行或环境变化后执行：

```bash
cd /home/ljt23/research_ca/02_extreme_avoidance
bash scripts/bootstrap.sh
.venv/bin/python -m extreme_avoidance run --scenario straight_critical --controller joint
```

运行在线规划或重建整个实验集：

```bash
.venv/bin/python -m extreme_avoidance run --scenario straight_online --controller rolling
.venv/bin/python scripts/run_benchmark.py --output artifacts/my_online_benchmark
.venv/bin/python scripts/validate_planning.py --output artifacts/my_planning_validation
```

修改规划参数用 `configs/planner.json` 或 `run --planner-config <file>`；实验矩阵在 `configs/benchmark.json`。新场景显式配置 `friction_mode: internal_mu`，旧场景未声明时保留 `post_force`。

环境选择及重建边界见 [环境说明](docs/ENVIRONMENT.md)。不要复用 `01_flow_planner` 的虚拟环境；原生模块必须用本项目 `.venv` 对应的 Python 重新构建。

命令会打印输出目录，包含 `replay.html`、`trajectory.png`、`trace.csv`、`metrics.json`、`config.json` 和 `manifest.json`。不指定 `--output` 时使用时间戳目录。显式输出路径必须不存在，以免覆盖实验。

```bash
# 其他基线：aeb / steer / joint / cruise
.venv/bin/python -m extreme_avoidance run --scenario straight_low_grip --controller steer --side -1

# 全部代表工况、图像及临界场景步长复核
.venv/bin/python scripts/run_suite.py --output artifacts/my_suite

# 独立物理检查与轮胎缩放标定
.venv/bin/python scripts/validate_physics.py --output artifacts/my_physics
.venv/bin/python scripts/plot_validation.py artifacts/my_physics

# 比较主附着定义与内部 LMUX/LMUY 缩放
.venv/bin/python scripts/calibrate_friction.py --output artifacts/my_friction_calibration.json

# 离线反馈机动模板采样，12 条候选全部保留
.venv/bin/python -m extreme_avoidance sample --scenario straight_critical --output artifacts/my_candidates

# 回归检查与回放脚本检查
.venv/bin/python -m unittest discover -s tests -v
node scripts/check_replays.cjs artifacts/my_suite
```

若服务器没有图形桌面，可通过 VS Code Remote SSH 下载整个 `artifacts/suite_20261015` 文件夹到本机，再用浏览器打开 `index.html`；单个 `replay.html` 本身也可独立下载播放。

## 当前适用范围

这是**使用合成研究车辆参数的最小实验平台**，已经过数值与基本物理一致性检查，尚无实车或 CarSim 对照。车辆为 800 kg 通用研究配置，不能称为已标定的乘用车。

- Open Car Dynamics 固定提交 `94f8fb187fb0ed22bba1d809bd74f66d1ff75af4`，使用双轨、MF52、PT1 转向和 FX 请求接口；192 个所需上游文件的 Git blob 校验全部通过。
- TAMP 仅作为突现障碍、道路分区和附着变化的场景语义参考，未复现 SAA-RTI 算法或其原始实验。
- 新实验采用逐轮 `LMUX/LMUY` 相对基准值的内部倍率，外部 `lambda_mue=1`；旧后置力缩放保留为对照。两者都不是绝对路面摩擦系数，见 [附着定义](docs/FRICTION_DEFINITION.md)。
- `rolling` 每 100 ms 用当前观测重新展开 37–38 个反馈机动候选，反馈控制每 20 ms 更新；采用独立三自由度自行车预测器和扫掠车身检查，OCD 执行后再评价真实仿真结果。它是确定性采样格点滚动规划，不是 CEM/MPPI 或 TAMP 复现。
- 预测器已测量开环误差，5 s 位置误差在测试中可达约 3.52 m；几何余量不构成鲁棒安全保证。在线基线当前仅支持直路静态障碍，未达到可迁移至真实车辆的验证程度。
- 目前仅支持静态障碍和直路/定曲率道路，基础避让控制器面向单障碍。没有移动障碍、感知网络、自动微分或生成模型训练。
