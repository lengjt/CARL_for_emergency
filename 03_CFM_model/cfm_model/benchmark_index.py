"""Standalone benchmark index used by browsers and replay validation."""

import html
from pathlib import Path


def write_benchmark_index(output, episodes):
    output = Path(output)
    rows = []
    for row in episodes:
        seed = "" if row.get("model_seed") is None else f" / seed {row['model_seed']}"
        status_class = "ok" if row["success"] else "bad"
        rows.append(
            f"<tr><td>{html.escape(row['group_id'])}</td>"
            f"<td>{html.escape(row['controller'] + seed)}</td>"
            f"<td class=\"{status_class}\">{html.escape(row['status'])}</td>"
            f"<td>{row.get('planner_p95_ms', 0.0):.2f}</td>"
            f"<td><a href=\"{html.escape(row['path'])}/replay.html\">播放</a></td></tr>")
    (output / "index.html").write_text("""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>CFM 闭环评价回放</title><style>body{font:15px system-ui;max-width:1200px;margin:30px auto;color:#24364b}
table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #d8e0e8;text-align:left}
.ok{color:#17714b}.bad{color:#b23a3a}a{color:#1766a6}</style><h1>CFM 闭环评价回放</h1>
<p>固定测试场景；失败记录全部保留。<a href="summary.json">汇总 JSON</a></p>
<table><tr><th>场景</th><th>方法</th><th>状态</th><th>规划 P95 (ms)</th><th>回放</th></tr>""" +
        "\n".join(rows) + "</table></html>")
