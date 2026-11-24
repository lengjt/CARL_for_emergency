# nuPlan 数据准备

本阶段只使用官方预训练权重，不训练，不需要百万样本训练集。

官方入口：https://www.nuscenes.org/nuplan#download

官方 devkit 的 `docs/dataset_setup.md` 要求创建账户、登录并同意数据使用条款，随后才能看到下载包。本地现已使用用户提供的压缩包完成解压和校验，未使用第三方镜像替代数据。

## 最小需要什么

- nuPlan v1.1 **mini 的数据库包**（解压后为 `.db`）。
- **地图包**：地图索引 `nuplan-maps-v1.0.json` 与对应城市版本的 `map.gpkg`。
- 不需要相机图像、激光雷达点云等 sensor blobs：Flow Planner 使用跟踪目标、车辆状态和向量地图；本入口关闭相机加载。

按以下结构解压，或保留已有位置并用参数指定：

```text
01_flow_planner/data/
├── maps/
│   ├── nuplan-maps-v1.0.json
│   ├── us-nv-las-vegas-strip/9.15.1915/map.gpkg
│   ├── us-ma-boston/9.12.1817/map.gpkg
│   ├── us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg
│   └── sg-one-north/9.17.1964/map.gpkg
└── nuplan-v1.1/splits/mini/
    └── *.db
```

上述版本路径来自当前固定的 devkit 文档；以实际官方地图索引为准，不重命名地图版本。

## 运行

```bash
cd /home/ljt23/research_ca/01_flow_planner
.venv/bin/python scripts/run_nuplan.py --data-root /absolute/path/to/dataset --scenarios 3
```

如地图独立存放，增加 `--maps-root /absolute/path/to/maps`。默认使用单卡、顺序执行、非反应式周围交通参与者，避免多进程复制模型。`--reactive` 切换到 nuPlan IDM 反应式交通。`--tokens TOKEN1 TOKEN2` 固定场景；首次小样本完成后，应从输出配置/日志记录具体 token，再用于后续对照。

结果位于 `artifacts/nuplan/`；nuPlan 保存场景指标、仿真轨迹日志和运行报告。应检查实际成功场景数、失败场景、无责任碰撞、可行驶区域、进度、舒适性及综合分数，不能仅以进程退出作为算法成功。

mini 子集是运行检查，不能与论文 Val14/Test14 的完整分数直接比较。缺数据时入口立即报错，不会生成伪闭环结果。

## 本机数据检查结果（2026-10-13）

- 数据库压缩包：`data/nuplan-v1.1_mini.zip`，8,550,100,030字节。
- ZIP中64个数据库和LICENSE全部通过CRC校验；数据库合计14,351,183,872字节（约13.37 GiB）。
- ZIP内的`data/cache/mini/*.db`已整理到`data/nuplan-v1.1/splits/mini/*.db`。
- 64个数据库全部通过SQLite `quick_check`，并具有nuPlan必要数据表。
- 四个城市地图版本与索引一致，`map.gpkg`全部通过SQLite `quick_check`，覆盖数据库引用的所有城市。地图目录沿用用户整理的结构。
- 原有同名数据库内容不同，替换前曾备份到`data/pre_extraction_backup/20261013_212941/`。该备份已于2026-10-13按用户要求删除（115,769,344字节）；现用64个数据库及原ZIP保留。
- 完整文件清单、SHA256及检查结果见`artifacts/data_preparation.json`。

本机直接运行：

```bash
cd /home/ljt23/research_ca/01_flow_planner
.venv/bin/python scripts/run_nuplan.py --scenarios 3
```
