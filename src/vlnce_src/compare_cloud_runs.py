import argparse
import csv
import json
import os
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_one(run_dir, pattern):
    matches = sorted(Path(run_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError("No file matching {} in {}".format(pattern, run_dir))
    return matches[0]


def load_run(run_dir):
    run_dir = Path(run_dir).resolve()
    stats_file = find_one(run_dir, "stats_cloud_*.json")
    failure_file = find_one(run_dir, "failure_analysis_cloud_*.json")
    manifest_file = run_dir / "run_manifest.json"
    episode_file = run_dir / "episode_list_evaluated.json"
    return {
        "run_dir": str(run_dir),
        "stats_file": str(stats_file),
        "failure_file": str(failure_file),
        "manifest_file": str(manifest_file) if manifest_file.exists() else None,
        "episode_file": str(episode_file) if episode_file.exists() else None,
        "stats": read_json(stats_file),
        "failure": read_json(failure_file),
        "manifest": read_json(manifest_file) if manifest_file.exists() else {},
        "episodes": read_json(episode_file).get("episode_ids", []) if episode_file.exists() else [],
    }


def numeric_delta(a, b, key):
    a_value = a["stats"].get(key)
    b_value = b["stats"].get(key)
    if isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
        return {
            "a": a_value,
            "b": b_value,
            "delta_b_minus_a": b_value - a_value,
        }
    return None


def episode_outcomes(run):
    return run["failure"].get("episodes", {})


def compare_episodes(a, b):
    a_eps = episode_outcomes(a)
    b_eps = episode_outcomes(b)
    common_ids = sorted(set(a_eps.keys()) & set(b_eps.keys()))
    only_a = sorted(set(a_eps.keys()) - set(b_eps.keys()))
    only_b = sorted(set(b_eps.keys()) - set(a_eps.keys()))

    improved = []
    regressed = []
    same = []
    for episode_id in common_ids:
        a_item = a_eps[episode_id]
        b_item = b_eps[episode_id]
        a_score = float(a_item.get("ndtw") or 0.0)
        b_score = float(b_item.get("ndtw") or 0.0)
        row = {
            "episode_id": episode_id,
            "a_success": a_item.get("success"),
            "b_success": b_item.get("success"),
            "a_ndtw": a_score,
            "b_ndtw": b_score,
            "delta_ndtw": b_score - a_score,
            "a_reasons": a_item.get("reasons", []),
            "b_reasons": b_item.get("reasons", []),
        }
        if b_score > a_score:
            improved.append(row)
        elif b_score < a_score:
            regressed.append(row)
        else:
            same.append(row)

    improved.sort(key=lambda item: item["delta_ndtw"], reverse=True)
    regressed.sort(key=lambda item: item["delta_ndtw"])
    return {
        "common_episode_count": len(common_ids),
        "only_a_episode_ids": only_a,
        "only_b_episode_ids": only_b,
        "improved": improved,
        "regressed": regressed,
        "same_count": len(same),
    }


def compare_behavior_labels(a, b):
    a_labels = a["failure"].get("summary", {}).get("behavior_labels", {})
    b_labels = b["failure"].get("summary", {}).get("behavior_labels", {})
    labels = sorted(set(a_labels.keys()) | set(b_labels.keys()))
    return {
        label: {
            "a": a_labels.get(label, 0),
            "b": b_labels.get(label, 0),
            "delta_b_minus_a": b_labels.get(label, 0) - a_labels.get(label, 0),
        }
        for label in labels
    }


def compare_configs(a, b):
    keys = [
        "model",
        "base_url",
        "split",
        "eval_num",
        "max_action",
        "depth_mode",
        "use_depth_summary",
        "use_memory",
        "use_pose",
    ]
    return {
        key: {
            "a": a["manifest"].get(key),
            "b": b["manifest"].get(key),
            "same": a["manifest"].get(key) == b["manifest"].get(key),
        }
        for key in keys
        if a["manifest"].get(key) is not None or b["manifest"].get(key) is not None
    }


def write_summary_csv(path, metric_deltas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "a", "b", "delta_b_minus_a"])
        for key, item in metric_deltas.items():
            writer.writerow([key, item["a"], item["b"], item["delta_b_minus_a"]])


def write_markdown(path, payload):
    lines = [
        "# Cloud Run Comparison",
        "",
        "## Runs",
        "",
        "- A: `{}`".format(payload["run_a"]["run_dir"]),
        "- B: `{}`".format(payload["run_b"]["run_dir"]),
        "",
        "## Metric Deltas",
        "",
        "| metric | A | B | B - A |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, item in payload["metric_deltas"].items():
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} |".format(
            key,
            float(item["a"]),
            float(item["b"]),
            float(item["delta_b_minus_a"]),
        ))

    lines.extend([
        "",
        "## Episode Coverage",
        "",
        "- common_episode_count: `{}`".format(payload["episode_comparison"]["common_episode_count"]),
        "- only_a_episode_count: `{}`".format(len(payload["episode_comparison"]["only_a_episode_ids"])),
        "- only_b_episode_count: `{}`".format(len(payload["episode_comparison"]["only_b_episode_ids"])),
        "- improved_count: `{}`".format(len(payload["episode_comparison"]["improved"])),
        "- regressed_count: `{}`".format(len(payload["episode_comparison"]["regressed"])),
        "",
        "## Behavior Label Deltas",
        "",
    ])
    if payload["behavior_label_deltas"]:
        for key, item in payload["behavior_label_deltas"].items():
            lines.append("- {}: {} -> {} ({:+d})".format(
                key,
                item["a"],
                item["b"],
                int(item["delta_b_minus_a"]),
            ))
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Most Improved Episodes",
        "",
        "| episode_id | A nDTW | B nDTW | delta | B reasons |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for item in payload["episode_comparison"]["improved"][:10]:
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
            item["episode_id"],
            item["a_ndtw"],
            item["b_ndtw"],
            item["delta_ndtw"],
            ", ".join(item["b_reasons"]),
        ))

    lines.extend([
        "",
        "## Most Regressed Episodes",
        "",
        "| episode_id | A nDTW | B nDTW | delta | B reasons |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for item in payload["episode_comparison"]["regressed"][:10]:
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
            item["episode_id"],
            item["a_ndtw"],
            item["b_ndtw"],
            item["delta_ndtw"],
            ", ".join(item["b_reasons"]),
        ))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two cloud evaluation result directories.")
    parser.add_argument("--run_a", required=True, help="First DATA/output/.../eval/results/<time> directory")
    parser.add_argument("--run_b", required=True, help="Second DATA/output/.../eval/results/<time> directory")
    parser.add_argument("--output_dir", default=None, help="Directory to save comparison files")
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    metrics = [
        "success",
        "ndtw",
        "sdtw",
        "oracle_success",
        "distance_to_goal",
        "path_length",
        "steps_taken",
        "cloud_requested_steps",
        "cloud_fallback_steps",
        "cloud_error_steps",
        "cloud_latency_sec_avg",
        "cloud_latency_sec_max",
    ]
    metric_deltas = {}
    for key in metrics:
        item = numeric_delta(run_a, run_b, key)
        if item is not None:
            metric_deltas[key] = item

    payload = {
        "run_a": {key: run_a[key] for key in ["run_dir", "stats_file", "failure_file", "manifest_file", "episode_file"]},
        "run_b": {key: run_b[key] for key in ["run_dir", "stats_file", "failure_file", "manifest_file", "episode_file"]},
        "config_deltas": compare_configs(run_a, run_b),
        "metric_deltas": metric_deltas,
        "behavior_label_deltas": compare_behavior_labels(run_a, run_b),
        "episode_comparison": compare_episodes(run_a, run_b),
    }

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(run_b["run_dir"]) / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / "cloud_run_comparison.json"
    csv_file = output_dir / "cloud_run_metric_deltas.csv"
    report_file = output_dir / "cloud_run_comparison.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    write_summary_csv(csv_file, metric_deltas)
    write_markdown(report_file, payload)

    print("comparison_json: {}".format(json_file))
    print("comparison_csv: {}".format(csv_file))
    print("comparison_report: {}".format(report_file))


if __name__ == "__main__":
    main()
