#!/usr/bin/env python3
"""Build a concise Chinese report, plots, and replay index from frozen artifacts."""

import argparse
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfm_model.io import manifest, read_json, read_jsonl, write_json


def percent(value):
    return f"{100.0 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--predictor-validation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = read_json(args.dataset / "audit.json")
    training = read_json(args.training / "summary.json")
    benchmark = read_json(args.benchmark / "summary.json")
    predictor = (read_json(args.predictor_validation / "report.json")
                 if args.predictor_validation else None)
    episodes = read_jsonl(args.benchmark / "episodes.jsonl")
    results = benchmark["results"]
    candidate_audit = {}
    for method in ("cfm", "hybrid"):
        selected = 0
        outside_lattice = 0
        unique = set()
        for episode in episodes:
            if episode["controller"] != method or not episode["success"]:
                continue
            plan_path = args.benchmark / episode["path"] / "plans.jsonl"
            for event in read_jsonl(plan_path):
                if event["selected_index"] is None:
                    continue
                parameters = event["candidates"][event["selected_index"]]["parameters"]
                selected += 1
                unique.add(tuple(round(float(value), 4) for value in parameters[:3]))
                on_lattice = ((round(abs(parameters[0]), 6) in (2.6, 3.0, 3.4) and
                               round(parameters[2], 6) in (0.0, 1200.0, 2400.0)) or
                              parameters[5] > 0.5)
                outside_lattice += not on_lattice
        candidate_audit[method] = {"selected": selected, "outside_lattice": outside_lattice,
                                   "unique_selected": len(unique)}
    keys = ["aeb", "joint", "rolling", "retrieval"] + [
        key for key in results if key.startswith("cfm_seed_")] + [
        key for key in results if key.startswith("hybrid_seed_")]
    labels = {"aeb": "AEB", "joint": "联合控制", "rolling": "传统滚动格点",
              "retrieval": "训练集检索"}
    report = [
        "# CFM 极限避障小规模概念验证报告",
        "",
        "本实验在直路、单静态障碍、均匀附着和合成研究车辆范围内，验证轻量 CFM 是否能生成有助于避障与稳定恢复的候选机动。所有最终成败均由 OCD 闭环判定；预测器通过不等于成功。",
        "",
        "## 数据与划分",
        "",
        f"共 {data['scenario_families']} 个基础场景族、{data['scenario_groups']} 个近邻场景、{data['episodes']} 条左右优先示范 episode。成功 {data['successful_episodes']} 条，失败 {data['failed_episodes_retained']} 条均保留；从完整 OCD 成功 episode 中提取 {data['demonstration_decisions']} 个决策示范，含 {data['unique_target_modes_rounded']} 种按 0.001 舍入的目标模式。",
        "",
        "划分按基础场景族完成；近邻变体、左右分支和同一 episode 的时间切片不跨集合。只有完整闭环稳定恢复且没有约束违反的决策序列进入正示范。",
        "",
        "| 划分 | 场景族 | 场景 | episode | 成功 episode | 失败 episode | 决策示范 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        row = data["by_split"][split]
        report.append(f"| {split} | {row['families']} | {row['scenarios']} | {row['episodes']} | "
                      f"{row['successful_episodes']} | {row['failed_episodes']} | {row['demonstration_decisions']} |")
    report.extend([
        "",
        "## 训练记录与模型误差",
        "",
        f"训练设备为 `{training['device']}`，三个随机种子总训练时间 {training['training_wall_time_s_total']:.1f} s。CFM 验证损失是标准化空间中的速度场均方误差，只衡量所学流场拟合，不是车辆位置误差或安全界。",
        "",
        "| 种子 | 最佳步数 / 运行至 | 最佳验证 CFM MSE | 末次训练 MSE | 时间 (s) | 峰值显存 (MiB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in training["runs"]:
        best_step = row.get("best_step", min(row["history"], key=lambda point: point["val_cfm_mse"])["step"])
        report.append(f"| {row['seed']} | {best_step} / {row['steps_completed']} | {row['best_val_cfm_mse']:.5f} | "
                      f"{row['final_train_cfm_mse']:.5f} | {row['wall_time_s']:.1f} | "
                      f"{row['peak_gpu_memory_MiB']:.1f} |")
    report.extend(["", "每个测试条件生成 32 个候选后，与对应未见示范标签的最近距离如下。该误差衡量候选集合覆盖，不是动力学或安全误差。", "",
                   "| 种子 | 标准化 best-of-32 L2 | 偏移 MAE (m) | 建立距离 MAE (m) | 制动力 MAE (kN) |",
                   "|---:|---:|---:|---:|---:|"])
    for row in training["runs"]:
        error = row["test_set_generation_error"]
        mae = error["best_of_32_raw_mae"]
        report.append(f"| {row['seed']} | {error['best_of_32_normalized_l2_mean']:.3f} | "
                      f"{mae['signed_offset_m']:.3f} | {mae['ramp_length_m']:.3f} | {mae['brake_kN']:.3f} |")
    if predictor:
        report.extend(["", "## 筛选模型误差补测", "",
                       f"针对原平台缺少 25 m/s 误差扫描的问题，新增 {predictor['conditions']} 个开环条件，包含纯制动和更强联合转向制动。下表为有限扫描中的最大误差，不是置信界或安全余量。", "",
                       "| 时域 (s) | 位置 (m) | 横向 (m) | 速度 (m/s) | 航向 (rad) |",
                       "|---:|---:|---:|---:|---:|"])
        for horizon in ("1", "2", "5"):
            row = predictor["worst_by_horizon"][horizon]
            report.append(f"| {horizon} | {row['max_position_error_m']:.3f} | "
                          f"{row['max_lateral_error_m']:.3f} | {row['max_speed_error_mps']:.3f} | "
                          f"{row['max_yaw_error_rad']:.4f} |")
    report.extend([
        "",
        "## 冻结测试组闭环比较",
        "",
        f"主评价包含 {benchmark['test_families']} 个未见基础场景族、{benchmark['test_scenarios']} 个近邻场景。CFM 分别报告三个训练种子；候选数量、预测筛选、执行器、恢复条件、100 ms 截止时间和 AEB 回退与传统滚动基线一致。",
        "",
        "| 方法 | episode | 避障 | 稳定恢复/有效成功 | 约束违反 | 规划 P95 均值 (ms) | 最大 P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in keys:
        row = results[key]
        label = labels.get(key, key.replace("cfm_seed_", "CFM 种子 ").replace(
            "hybrid_seed_", "混合种子 "))
        report.append(f"| {label} | {row['episodes']} | {row['avoidance_successes']}/{row['episodes']} "
                      f"({percent(row['avoidance_rate'])}) | {row['valid_successes']}/{row['episodes']} "
                      f"({percent(row['valid_success_rate'])}) | {row['constraint_violations']}/{row['episodes']} | "
                      f"{row['planner_p95_ms_mean']:.2f} | {row['planner_p99_ms_max']:.2f} |")
    all_cfm = results["cfm_all_seeds"]
    all_hybrid = results["hybrid_all_seeds"]
    report.extend([
        "",
        f"三个 CFM 种子合计有效成功率为 {percent(all_cfm['valid_success_rate'])}，约束违反率为 {percent(all_cfm['constraint_violation_rate'])}；混合方案分别为 {percent(all_hybrid['valid_success_rate'])} 和 {percent(all_hybrid['constraint_violation_rate'])}。这些比例来自固定有限测试集，不是总体成功概率。",
        "",
        "## 候选贡献诊断",
        "",
        f"在纯 CFM 的成功 episode 中，在线规划共选择 {candidate_audit['cfm']['selected']} 次候选，其中 {candidate_audit['cfm']['outside_lattice']} 次的偏移或制动力不在原固定格点上，共有 {candidate_audit['cfm']['unique_selected']} 种连续参数组合。混合方案相应为 {candidate_audit['hybrid']['outside_lattice']}/{candidate_audit['hybrid']['selected']} 次和 {candidate_audit['hybrid']['unique_selected']} 种。这表明闭环成功确实执行了生成候选；它仍未改善冻结测试集的成功边界。",
        "",
        "## 失败解释与适用边界",
        "",
        "- CFM 只学习传统滚动格点成功 episode 中实际执行的参数，因而可验证的是对已有简单机动分布的压缩与重组，尚不能证明发现了传统候选之外的新可行模式。",
        "- 三个模型均在 100 步达到最佳验证损失，之后训练损失下降而验证损失上升；已通过缩小网络和早停保留最佳检查点，但独立场景仍少，过拟合是当前主要模型误差。",
        "- 训练目标的流场 MSE 与闭环安全没有一一对应关系；生成候选仍可能被近似预测器误判，最终以 OCD 的碰撞、出界、恢复和执行器指标为准。",
        "- 25 m/s 补测的 5 s 最大位置误差达到 15.611 m，说明当前近似筛选器在强联合输入和长时域下误差很大；滚动重规划降低长期依赖，但该筛选器不能作为安全证书。",
        "- 当前能力输入是带可选偏差/延迟的真值代理，车辆参数未经目标乘用车、CarSim 或实车标定。",
        "- 场景不含曲路、多障碍、移动障碍、附着突变或感知误差。本结果只支持是否继续扩大研究的工程决策。",
        "- 规划计时包含候选生成、预测展开、检查和排名，不包含 OCD 推进、绘图和写盘，也不构成硬实时保证。",
        "",
        "## 代表性动态回放",
        "",
    ])
    by_group = defaultdict(dict)
    for row in episodes:
        key = (row["controller"] if row["controller"] not in ("cfm", "hybrid")
               else f"{row['controller']}_seed_{row['model_seed']}")
        by_group[row["group_id"]][key] = row
    ranked = []
    for group, values in by_group.items():
        cfm_rows = [row for key, row in values.items() if key.startswith("cfm_seed_")]
        cfm_successes = sum(row["success"] for row in cfm_rows)
        rolling_success = values.get("rolling", {}).get("success", False)
        ranked.append((not rolling_success and cfm_successes > 0,
                       rolling_success and cfm_successes == 0,
                       cfm_successes, group, values))
    ranked.sort(reverse=True)
    failures = [row for row in ranked if row[2] == 0]
    successes = [row for row in ranked if row[2] > 0]
    selected = failures[:2] + successes[:6]
    replay_rows = []
    for _, _, _, group, values in selected:
        report.append(f"- `{group}`：")
        for key in ("rolling", "retrieval", "cfm_seed_17", "cfm_seed_29", "cfm_seed_43",
                    "hybrid_seed_17", "hybrid_seed_29", "hybrid_seed_43"):
            if key in values:
                row = values[key]
                relative = Path("..") / Path(args.benchmark.name) / row["path"] / "replay.html"
                label = labels.get(key, key.replace("cfm_seed_", "CFM 种子 ").replace(
                    "hybrid_seed_", "混合种子 "))
                report.append(f"  [{label} — {row['status']}]({relative.as_posix()})")
                replay_rows.append((group, label, row["status"], relative.as_posix()))
    (args.output / "report.md").write_text("\n".join(report) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_keys = keys
    rates = [results[key]["valid_success_rate"] for key in plot_keys]
    english = {"aeb": "AEB", "joint": "Joint", "rolling": "Rolling lattice",
               "retrieval": "Retrieval"}
    plot_labels = [english.get(key, key.replace("cfm_seed_", "CFM ").replace(
        "hybrid_seed_", "Hybrid ")) for key in plot_keys]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(rates)), rates, color=["#777777", "#8093b0", "#3c78a8", "#5a9b78"] +
           ["#b36958"] * 3 + ["#8d6ca8"] * 3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valid stable-recovery rate")
    ax.set_xticks(np.arange(len(rates)), plot_labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(rates):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / "closed_loop_success.png", dpi=170)
    plt.close(fig)

    rows_html = "".join(
        f"<tr><td>{html.escape(group)}</td><td>{html.escape(label)}</td>"
        f"<td>{html.escape(status)}</td><td><a href='{html.escape(path)}'>播放</a></td></tr>"
        for group, label, status, path in replay_rows)
    (args.output / "index.html").write_text(f"""<!doctype html><html lang='zh'><meta charset='utf-8'>
<title>CFM 极限避障概念验证</title><style>body{{font:15px system-ui;max-width:1100px;margin:35px auto;color:#24364b}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d8e0e8;text-align:left}}
img{{max-width:100%}}a{{color:#1766a6}}</style><h1>CFM 极限避障概念验证</h1>
<p><a href='report.md'>完整简明报告</a> · <a href='../{html.escape(args.benchmark.name)}/summary.json'>闭环汇总 JSON</a></p>
<img src='closed_loop_success.png' alt='闭环有效成功率'><h2>代表性动态回放</h2>
<table><tr><th>场景</th><th>方法</th><th>状态</th><th>回放</th></tr>{rows_html}</table></html>""")
    write_json(args.output / "summary.json", {
        "dataset": data, "training": training, "benchmark": benchmark,
        "candidate_contribution_audit": candidate_audit,
        "report": "report.md", "replay_index": "index.html",
    })
    write_json(args.output / "manifest.json", manifest(args.output))


if __name__ == "__main__":
    main()
