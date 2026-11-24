# Python 环境约定

本项目使用仓库内 `.venv` 作为唯一 Python 入口。它由 Ubuntu 22.04 的 `/usr/bin/python3` 创建，并启用 `--system-site-packages`，以离线复用本机已经验证的 NumPy 1.21.5 和 Matplotlib 3.5.1 二进制包。`requirements-lock.txt` 与 `scripts/check_environment.py` 对这两个直接依赖做精确版本检查。

这是一种面向当前 OCD 原生绑定的折中：解释器入口、构建产物和其他研究项目相互隔离，但 NumPy、Matplotlib 及其传递依赖仍由操作系统包管理，不是完全自包含的 pip 环境。CMake、编译器、pybind11、Eigen 和 Boost 也属于系统级构建依赖。

统一执行：

```bash
bash scripts/bootstrap.sh
.venv/bin/python -m extreme_avoidance run --scenario straight_critical --controller joint
```

`bootstrap.sh` 创建环境、核查版本、用同一解释器编译 `_ocd_native`，然后运行回归检查。`build.sh` 默认拒绝缺少 `.venv` 的运行；如需明确测试另一个兼容解释器，可设置 `EXTREME_AVOIDANCE_PYTHON`，但实验记录会保存实际解释器路径。

如果以后升级 Python、Ubuntu 或进入 PyTorch/CFM 训练，应新建独立训练环境和锁文件，保留本环境作为 OCD 评价环境。两个环境通过版本化数据文件或明确的服务接口连接，不应把大型训练依赖直接加入当前最小仿真环境。
