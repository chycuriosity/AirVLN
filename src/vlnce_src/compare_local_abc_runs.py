import argparse
import json
from pathlib import Path


KEY_METRICS = [
    "success",
    "ndtw",
    "sdtw",
    "oracle_success",
    "distance_to_goal",
    "path_length",
    "steps_taken",
]

INTERSECTION_METRICS = [
    "intersection_candidate_events",
    "intersection_wrong_decision_candidates",
    "intersection_cloud_checked_candidates",
    "intersection_cloud_positive_events",
    "intersection_cloud_positive_rate",
    "intersection_event_count",
    "intersection_episodes_with_events",
    "intersection_episode_event_rate",
    "intersection_corrections_applied",
    "intersection_corrections_suppressed",
    "intersection_cloud_latency_sec_avg",
    "intersection_cloud_latency_sec_max",
]


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_stats(path_value):
    path = Path(path_value).expanduser()
    if path.is_file():
        return path

    candidates = sorted(
        path.glob("**/stats_ckpt_*_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No stats_ckpt_*_*.json found under {}".format(path))
    return candidates[0]


def eval_root_from_stats(stats_path):
    parts = list(stats_path.parts)
    if "eval" not in parts:
        return None
    eval_index = len(parts) - 1 - parts[::-1].index("eval")
    return Path(*parts[: eval_index + 1])


def run_time_from_stats(stats_path):
    # DATA/output/{name}/eval/results/{time}/stats...
    return stats_path.parent.name


def resolve_intermediate(stats_path):
    eval_root = eval_root_from_stats(stats_path)
    if eval_root is None:
        return None
    path = eval_root / "intermediate_results" / run_time_from_stats(stats_path) / stats_path.name
    return path if path.exists() else None


def resolve_intersection_summary(stats_path):
    eval_root = eval_root_from_stats(stats_path)
    if eval_root is None:
        return None
    path = eval_root / "intersection_events" / run_time_from_stats(stats_path) / stats_path.name.replace("stats_", "summary_")
    return path if path.exists() else None


def resolve_intersection_events(stats_path):
    eval_root = eval_root_from_stats(stats_path)
    if eval_root is None:
        return None
    path = eval_root / "intersection_events" / run_time_from_stats(stats_path) / stats_path.name.replace("stats_", "events_")
    return path if path.exists() else None


def load_run(label, path_value):
    stats_path = resolve_stats(path_value)
    stats = read_json(stats_path)
    intermediate_path = resolve_intermediate(stats_path)
    intermediate = read_json(intermediate_path) if intermediate_path else {}
    summary_path = resolve_intersection_summary(stats_path)
    intersection_summary = read_json(summary_path) if summary_path else {}
    events_path = resolve_intersection_events(stats_path)
    events = read_json(events_path) if events_path else []
    return {
        "label": label,
        "stats_path": stats_path,
        "stats": stats,
        "intermediate_path": intermediate_path,
        "intermediate": intermediate,
        "intersection_summary_path": summary_path,
        "intersection_summary": intersection_summary,
        "events_path": events_path,
        "events": events,
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "{:.6f}".format(value)
    return str(value)


def metric_table(runs):
    lines = [
        "| metric | A | B | C | C-A |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in KEY_METRICS:
        values = [runs[label]["stats"].get(key) for label in ["A", "B", "C"]]
        delta = None
        if isinstance(values[0], (int, float)) and isinstance(values[2], (int, float)):
            delta = values[2] - values[0]
        lines.append("| {} | {} | {} | {} | {} |".format(
            key,
            fmt(values[0]),
            fmt(values[1]),
            fmt(values[2]),
            fmt(delta),
        ))
    return lines


def intersection_table(runs):
    lines = [
        "| metric | B | C |",
        "| --- | ---: | ---: |",
    ]
    for key in INTERSECTION_METRICS:
        b_value = runs["B"]["stats"].get(key)
        c_value = runs["C"]["stats"].get(key)
        if b_value is None:
            b_value = runs["B"]["intersection_summary"].get(key.replace("intersection_", ""))
        if c_value is None:
            c_value = runs["C"]["intersection_summary"].get(key.replace("intersection_", ""))
        lines.append("| {} | {} | {} |".format(key, fmt(b_value), fmt(c_value)))
    return lines


def episode_outcomes(runs):
    a_eps = runs["A"]["intermediate"]
    c_eps = runs["C"]["intermediate"]
    common = sorted(set(a_eps.keys()) & set(c_eps.keys()))
    helped = []
    hurt = []
    ndtw_gain = []
    for episode_id in common:
        a_info = a_eps[episode_id]
        c_info = c_eps[episode_id]
        a_success = float(a_info.get("success", 0.0) or 0.0)
        c_success = float(c_info.get("success", 0.0) or 0.0)
        a_ndtw = float(a_info.get("ndtw", 0.0) or 0.0)
        c_ndtw = float(c_info.get("ndtw", 0.0) or 0.0)
        item = {
            "episode_id": episode_id,
            "a_success": a_success,
            "c_success": c_success,
            "a_ndtw": a_ndtw,
            "c_ndtw": c_ndtw,
            "ndtw_delta": c_ndtw - a_ndtw,
        }
        if a_success < 1.0 and c_success >= 1.0:
            helped.append(item)
        if a_success >= 1.0 and c_success < 1.0:
            hurt.append(item)
        ndtw_gain.append(item)

    ndtw_gain.sort(key=lambda item: item["ndtw_delta"], reverse=True)
    return {
        "common_episode_count": len(common),
        "helped_success": helped,
        "hurt_success": hurt,
        "top_ndtw_gain": ndtw_gain[:10],
        "top_ndtw_drop": list(reversed(ndtw_gain[-10:])),
    }


def events_by_episode(events):
    counts = {}
    for event in events:
        episode_id = str(event.get("episode_id"))
        counts[episode_id] = counts.get(episode_id, 0) + 1
    return counts


def event_table(runs):
    b_counts = events_by_episode(runs["B"]["events"])
    c_counts = events_by_episode(runs["C"]["events"])
    episode_ids = sorted(set(b_counts) | set(c_counts), key=lambda item: (-(b_counts.get(item, 0) + c_counts.get(item, 0)), item))
    lines = [
        "| episode_id | B events | C events |",
        "| --- | ---: | ---: |",
    ]
    for episode_id in episode_ids[:20]:
        lines.append("| {} | {} | {} |".format(
            episode_id,
            b_counts.get(episode_id, 0),
            c_counts.get(episode_id, 0),
        ))
    if len(lines) == 2:
        lines.append("| none | 0 | 0 |")
    return lines


def build_report(runs):
    outcomes = episode_outcomes(runs)
    lines = [
        "# Local A/B/C Evaluation Comparison",
        "",
        "## Inputs",
        "",
    ]
    for label in ["A", "B", "C"]:
        lines.append("- {} stats: `{}`".format(label, runs[label]["stats_path"]))
        if runs[label]["intermediate_path"]:
            lines.append("- {} episodes: `{}`".format(label, runs[label]["intermediate_path"]))
        if runs[label]["intersection_summary_path"]:
            lines.append("- {} intersection summary: `{}`".format(label, runs[label]["intersection_summary_path"]))

    lines.extend([
        "",
        "## Core Metrics",
        "",
        *metric_table(runs),
        "",
        "## Intersection Metrics",
        "",
        *intersection_table(runs),
        "",
        "## Episode-Level C vs A",
        "",
        "- common_episode_count: `{}`".format(outcomes["common_episode_count"]),
        "- A failed, C succeeded: `{}`".format(len(outcomes["helped_success"])),
        "- A succeeded, C failed: `{}`".format(len(outcomes["hurt_success"])),
        "",
        "### Top nDTW Gains",
        "",
        "| episode_id | A nDTW | C nDTW | delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    for item in outcomes["top_ndtw_gain"]:
        lines.append("| {} | {} | {} | {} |".format(
            item["episode_id"],
            fmt(item["a_ndtw"]),
            fmt(item["c_ndtw"]),
            fmt(item["ndtw_delta"]),
        ))

    lines.extend([
        "",
        "### Most Intersection Events",
        "",
        *event_table(runs),
        "",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare local A/B/C AirVLN evaluation runs.")
    parser.add_argument("--a", required=True, help="A run stats file or directory")
    parser.add_argument("--b", required=True, help="B run stats file or directory")
    parser.add_argument("--c", required=True, help="C run stats file or directory")
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    runs = {
        "A": load_run("A", args.a),
        "B": load_run("B", args.b),
        "C": load_run("C", args.c),
    }
    report = build_report(runs)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
