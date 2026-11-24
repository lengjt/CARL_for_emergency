# 官方 interPlan 数据交接

## 需要下载的内容

这些场景属于 nuPlan **test split**，不属于 train split。因此 Las Vegas train
part 1–6 均不需要下载。需要下载的官方归档是：

```text
nuplan-v1.1_test.zip
```

官方 S3 ZIP 目录已确认以下 7 个文件全部位于该归档的 `data/cache/test/` 中：

1. `2021.05.25.12.30.39_veh-25_00321_01196.db`
2. `2021.05.25.14.16.10_veh-35_00083_00485.db`
3. `2021.10.06.04.07.24_veh-49_00776_01719.db`
4. `2021.06.28.21.29.28_veh-16_01912_03183.db`
5. `2021.10.06.02.32.50_veh-53_00633_00800.db`
6. `2021.06.03.12.02.06_veh-35_04692_04763.db`
7. `2021.09.22.03.46.15_veh-51_01522_02013.db`

统一放入：

```text
/home/ljt23/research_ca/01_flow_planner/data/nuplan-v1.1/splits/test/
```

地图 `nuplan-maps-v1.0` 已经存在于
`/home/ljt23/research_ca/01_flow_planner/data/maps`，无需重复下载。本实验使用
数据库中的矢量地图关联、ego 状态和 tracked objects，不启用相机，也不读取
原始 lidar 点云，因此不需要下载 `sensor_blobs`。

不要改名、解压后嵌套一层目录，或放入现有的 `splits/mini` 目录。interPlan
上游场景构造器的默认路径会由本项目运行入口显式覆盖到上述 `splits/test`。

## 放置后检查

```bash
/home/ljt23/research_ca/01_flow_planner/.venv/bin/python \
  /home/ljt23/research_ca/04_nuplan_static_obstacle/check_official_interplan_data.py
```

检查器会验证：7 个文件全部存在、SQLite 完整性、nuPlan 必要数据表，以及
`interplan10` 的全部基础 token。只有输出 `"passed": true` 后才开始仿真。

## 数据齐全后由 Codex 完成

1. 安装并验证本地 interPlan 插件与当前 nuPlan/Flow Planner 环境兼容。
2. 使用官方 `ModifiedNuPlanScenarioBuilder`、`default_interplan_benchmark`、
   `idm_modified_agents` 和 `interplan10` 修改配置。
3. 先运行 10 个 `nudge_around_parked_vehicle`，再运行 10 个
   `overtake_parked_vehicle` 和 10 个 `avoid_crashed_cars`。
4. 使用现有官方 Flow Planner checkpoint、相同采样设置、nuPlan LQR 和
   运动学自行车模型；不训练、不微调。
5. 保存每场输入审计、参考/实际轨迹、控制量、碰撞与场景指标，并生成动画和
   成功/失败汇总。所有失败都会保留。
