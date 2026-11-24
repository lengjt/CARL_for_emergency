# Flow Planner 最小运行复现（不重新训练）

本目录使用作者发布的模型权重，学习原方法的条件编码、轨迹分段、多模态 Transformer、流匹配采样和 nuPlan 闭环接口。按照本次任务约定，不进行任何训练或微调。

## 当前状态

- 已固定官方源码、下载官方权重与配套配置，并记录SHA256。
- 已建立项目内 Python 3.9 / PyTorch 2.3.0 CUDA 环境。
- 已在 RTX 4090 上严格加载全部模型参数、完成推理和固定种子一致性检查。
- 已完成合成接口场景上的采样步数/CFG机制检查，保存轨迹图及原始耗时。
- 已验证官方 nuPlan 规划适配器的构建与权重加载；仿真配置已成功组合。
- 已整理64个nuPlan mini数据库及四城市地图，完整性检查通过；已完成3个真实非反应式闭环场景，运行报告均成功。详见实验记录；此小样本不构成完整论文成绩复现。

模型参数量为14,274,384。发布模型生成未来8秒的80个点，输出每点的 `(x,y,cosθ,sinθ)`，不是转角和制动力。

## 先读什么

1. [架构与流程讲解](docs/ARCHITECTURE.md)：流程图、输入尺寸、片段和CFM原理、代码阅读顺序、与极限避障课题的关系。
2. [实验记录](docs/EXPERIMENTS.md)：已执行的实验、指标含义和证据边界。
3. [数据准备](docs/DATA_SETUP.md)：nuPlan mini数据库与地图的官方获取方式、目录结构、运行命令。
4. [兼容性与适配记录](docs/COMPATIBILITY.md)：发布权重格式、配置缺项、依赖固定、CFG行为。
5. [查看结果与动态回放](docs/VIEW_RESULTS.md)：三个真实闭环场景的GIF动画、结果位置及图例。

## 运行方式

从具有GPU访问权限的宿主机终端执行，不需要sudo。当前聊天的受限沙箱默认不暴露GPU，不能把沙箱中的CUDA不可见误判为服务器驱动故障。

```bash
cd /home/ljt23/research_ca/01_flow_planner

# 当前环境已安装；新环境重建时执行。
bash scripts/bootstrap.sh

# 官方规划适配器及权重加载检查（无需数据）。
.venv/bin/python scripts/check_environment.py

# 合成场景接口检查；不是nuPlan性能结果。
.venv/bin/python scripts/run_inference.py

# 实际数据就绪后：默认3个场景、单卡顺序执行。
.venv/bin/python scripts/run_nuplan.py --data-root /absolute/path/to/dataset --scenarios 3
```

支持从官方预处理流程导出的单场景NPZ进行推理：

```bash
.venv/bin/python scripts/run_inference.py --npz /absolute/path/to/sample.npz --output artifacts/real_sample
```

NPZ推理仍是单次开环模型调用，不是闭环安全验证。当前固定batch=1；多个候选顺序采样。闭环运行增加 `--reactive` 可选择反应式周围交通。正式对照需固定场景token、随机种子、采样参数和评价协议。

## 目录

```text
upstream/             固定版本官方源码，未直接修改
checkpoints/          官方权重、配套配置、SHA256
.venv/                项目独立环境
requirements-lock.txt 完整依赖版本
scripts/              环境准备、模型推理、配置检查与闭环入口
artifacts/            环境记录、推理数据/图表、真实闭环输出及动态回放
data/                 用户下载的nuPlan数据与地图
docs/                 原理、实验、数据和适配说明
```

源码版本：Flow Planner `366df7654d7b7fa77f9d2577279da63127c6d2f8`；nuPlan-devkit `e9241677997dd86bfc0bcd44817ab04fe631405b`；权重库 `b627934a8db3021892afd13f5278ba0149532efc`。

## 后续研究原则

复现重点是理解和验证原方法。未来迁移到低附着、转向—制动联合避障时，先设计4090可承担的紧凑模型与数据方案，再依据实测训练显存、吞吐和总实验量比较租用服务器。当前推理测量不能推断训练成本，也不证明原模型满足轮胎联合力、执行器或避障恢复约束。

项目总路线文档中原有的“小规模训练检查”不适用于本轮；以本目录的“不重新训练”约定为准。
