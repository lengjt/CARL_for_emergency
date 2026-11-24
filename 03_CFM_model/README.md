# 轻量 CFM 极限避障概念验证

本目录在 `02_extreme_avoidance` 的 Open Car Dynamics（OCD）闭环环境上验证一个小型条件流匹配（CFM）候选生成器。研究问题限定为：在直路、单个静态障碍和均匀附着的简单场景中，学习生成的反馈机动参数能否在相同候选预算下帮助避障并稳定恢复。

本阶段不声称真实车辆有效性、全局最优性或全面优于传统规划。车辆仍为合成 800 kg 研究配置；候选的放行继续依赖现有近似预测器，最终结论只取 OCD 闭环结果。

当前正式结果入口为 [概念验证报告](artifacts/report_v4/report.md) 和 [动态回放索引](artifacts/report_v4/index.html)。最终数据使用 `dataset_v1`、宽度 64 的 `training_v3`、`predictor_validation_v2` 和含混合候选对照的 `benchmark_v2`；更早的同类目录是开发审计记录，不纳入最终结论。

## 目录与环境

- `cfm_model/`：场景划分、数据接口、CFM 网络、NumPy 推理和在线规划器。
- `scripts/generate_dataset.py`：扩充简单场景，运行 OCD 闭环示范并保留失败。
- `scripts/train.py`：用 PyTorch 训练三个随机种子并导出 NumPy 权重。
- `scripts/evaluate.py`：在冻结测试组比较 AEB、联合控制、传统滚动格点、检索和 CFM。
- `scripts/build_report.py`：汇总数据覆盖、训练误差、闭环结果和代表性回放。
- `configs/`：冻结的数据、模型和评价设置。
- `artifacts/`：脚本生成的可复查实验产物；脚本拒绝静默覆盖已有实验。

OCD 数据生成与闭环评价使用 `../02_extreme_avoidance/.venv/bin/python`。训练使用已有 `../01_flow_planner/.venv/bin/python`，其中包含 PyTorch 2.3.0。在线评价通过导出的 NumPy 权重运行，因此不把 PyTorch 加入 OCD 环境。

## 运行顺序

```bash
cd /home/ljt23/research_ca/03_CFM_model
../02_extreme_avoidance/.venv/bin/python scripts/generate_dataset.py \
  --output artifacts/dataset_v1
../01_flow_planner/.venv/bin/python scripts/train.py \
  --dataset artifacts/dataset_v1 --output artifacts/training_v1
../02_extreme_avoidance/.venv/bin/python scripts/evaluate.py \
  --dataset artifacts/dataset_v1 --training artifacts/training_v1 \
  --output artifacts/benchmark_v1
../02_extreme_avoidance/.venv/bin/python scripts/build_report.py \
  --dataset artifacts/dataset_v1 --training artifacts/training_v1 \
  --benchmark artifacts/benchmark_v1 --output artifacts/report_v1
```

快速检查可分别传入 `--max-scenarios`、缩短训练步数或只运行一个模型种子。正式结果必须使用配置文件中的完整冻结设置。

## 数据语义

90 个场景由原有 18 个速度—附着—揭示距离基础组合各派生 5 个近邻变体。每个场景运行左右优先的传统滚动规划示范；所有成功、碰撞、出界、超时和回退记录均写入 `episodes.jsonl`。只有整个 OCD 闭环最终满足稳定恢复且无约束违反的 episode，其实际选中决策才进入 `demonstrations.jsonl`。

划分以 18 个基础场景族为单位，近邻变体、左右优先分支及时间切片始终位于同一 split。训练归一化只使用训练组。预测器判为可行但未经 OCD 闭环执行的候选不作为正示范。

## 模型与在线接口

目标为 `[signed_offset_m, ramp_length_m, brake_kN]`。制动释放位置固定为障碍中心，回归长度固定为 35 m；停车候选由公共回退集合提供。CFM 使用标准线性概率路径：

`x_t = (1-t) x_0 + t x_1`，监督速度为 `x_1 - x_0`。

向量场是宽度 64、三个残差块的 MLP。在线用 8 步 Euler 积分一次生成 36 个候选，再通过与传统格点完全相同的预测、约束检查、排名、执行器和 AEB 回退接口。网络结构将条件编码和速度场输入分开记录，后续可以把条件编码器替换成 Transformer，而不改变生成目标与评价协议。
