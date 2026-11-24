# 复现适配与边界

本目录保留官方 Flow Planner 与 nuPlan-devkit 源码，不直接修改上游文件。不执行训练、不生成替代权重。

## 已核实的发布材料差异

1. **权重格式**：下载的 `model.pth` 是338个张量组成的平铺 `OrderedDict`，键带 `module.` 前缀。官方 `planner.py` 默认按包含 `ema_state_dict` 的训练检查点读取，会触发 `KeyError`。独立推理直接去前缀并严格加载；闭环入口设置 `enable_ema=false`，使用官方已有的平铺权重加载分支。此开关只改变文件解析方式，不重新计算或改变已发布权重；无法仅从文件格式判定作者导出前是否应用EMA。
2. **配置缺项**：发布的 `model_config.yaml` 引用了 `data.dataset.train.future_downsampling_method` 与 `predicted_neighbor_num`，但没有该节。外围脚本补入官方数据配置的 `uniform` 和 `0`。不读取训练数据，也不初始化训练器。
3. **引导参数**：发布配置 CFG=1.0，仓库模型默认CFG=1.8。基线采用发布配置；实验改动在报告中记录。
4. **运行时CFG覆盖**：模型传入的 `cfg_weight` 未覆盖 `FlowODE` 保存的权重，所以在实例化前修改 `model.cfg_weight`。保留上游源码。
5. **多场景batch**：输入复制与噪声复制的排列不同，本轮限制batch=1。顺序生成多个样本不触及这一问题。

## 软件环境

- 项目内 Python 3.9.25，PyTorch 2.3.0+cu121、torchvision 0.18.0，与模型官方主版本一致。
- Flow Planner 未固定 `flow-matching` 版本，本轮固定1.0.10。
- nuPlan原依赖的Hydra 1.1.0rc1与Flow Planner的1.3.2冲突，统一采用后者。
- 仅安装模型推理与nuPlan仿真所需依赖，未执行完整训练工具链。nuPlan仿真入口间接导入Lightning；这不表示启动训练。
- setuptools固定69.5.1，以保留Lightning导入所需的`pkg_resources`。
- 仿真日志使用TensorBoard 2.15.2，以兼容当前protobuf/gRPC依赖组合；它不参与模型权重或生成计算。
- 采用无界面OpenCV 4.8.1.78，避免旧版OpenCV安装及GUI依赖；地图/渲染等版本见锁文件。
- nuPlan的运行时点云模块直接导入pytest，因此环境也包含pytest。
- `requirements-lock.txt`记录解析后的完整版本；`artifacts/environment.json`记录设备和实际运行环境。

## 测量范围

当前合成场景仅检验张量接口、权重加载、采样和数值输出。推理计时从GPU上的结构化输入开始，包含归一化、编码、采样与拼接，并在前后同步GPU；不包含数据库访问、地图查询、真实场景特征构造、控制和物理仿真。

`peak_allocated_mib`是PyTorch分配器统计，不包含整个CUDA上下文、驱动和其他进程。不能把该值当作服务器总显存需求。合成场景只有少数有效目标，也不能代表高密度交通时的开销。

小样本P95/P99仅描述本次30次运行，不是稳健的实时截止时间保证。低延迟不说明极限避障安全、动力学可行或训练资源足够。

## 真实数据阶段的运行适配

nuPlan顺序工作器要求`number_of_cpus_allocated_per_simulation=1`，首次真实运行暴露了此前设为4的配置错误，已修正。PyTorch内部CPU线程数仍为4，GPU模型运行不变。只改变外围启动参数，未修改上游源码或权重。
