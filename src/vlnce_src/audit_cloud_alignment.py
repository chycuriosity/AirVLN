import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(str(os.getcwd())).resolve()))


CHECKS = [
    {
        "name": "dataset_split_mapping",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["args.EVAL_DATASET == 'val_unseen'", "AirVLNENV(batch_size=args.batchSize"],
        "cloud_snippets": ["split_map = {", "AirVLNENV(batch_size=args.batchSize"],
    },
    {
        "name": "episode_limit",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["if args.EVAL_NUM != -1", "cnt * train_env.batch_size >= args.EVAL_NUM"],
        "cloud_snippets": ["if args.EVAL_NUM != -1", "cnt * train_env.batch_size >= args.EVAL_NUM"],
    },
    {
        "name": "environment_reset",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["outputs = train_env.reset()", "observations, _, dones"],
        "cloud_snippets": ["outputs = train_env.reset()", "observations, _, dones, infos"],
    },
    {
        "name": "action_execution",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["train_env.makeActions(actions)", "outputs = train_env.get_obs()"],
        "cloud_snippets": ["train_env.makeActions(actions)", "outputs = train_env.get_obs()"],
    },
    {
        "name": "video_generation",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["observations_to_image", "append_text_to_image", "generate_video("],
        "cloud_snippets": ["observations_to_image", "append_text_to_image", "generate_video("],
    },
    {
        "name": "metric_aggregation",
        "local_file": "src/vlnce_src/train.py",
        "cloud_file": "src/vlnce_src/cloud_eval.py",
        "local_snippets": ["stats_episodes", "sum(v[stat_key] for v in stats_episodes.values())"],
        "cloud_snippets": ["stats_episodes", "sum(v[stat_key] for v in numeric_stats_episodes.values())"],
    },
]


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_snippets(text, snippets):
    return {snippet: snippet in text for snippet in snippets}


def main():
    repo_root = Path(str(os.getcwd())).resolve()
    results = []
    for check in CHECKS:
        local_text = read_text(repo_root / check["local_file"])
        cloud_text = read_text(repo_root / check["cloud_file"])
        local = check_snippets(local_text, check["local_snippets"])
        cloud = check_snippets(cloud_text, check["cloud_snippets"])
        results.append(
            {
                "name": check["name"],
                "ok": all(local.values()) and all(cloud.values()),
                "local_file": check["local_file"],
                "cloud_file": check["cloud_file"],
                "local_snippets": local,
                "cloud_snippets": cloud,
            }
        )

    payload = {
        "all_ok": all(item["ok"] for item in results),
        "checks": results,
        "note": (
            "This audit verifies that cloud_eval keeps the original AirVLN evaluation "
            "surface for dataset iteration, environment stepping, video generation, and metric aggregation. "
            "The policy action source is intentionally different: local checkpoint policy vs cloud model."
        ),
    }

    output_file = Path(
        os.environ.get(
            "AIRVLN_AUDIT_OUTPUT",
            str(repo_root.parent / "DATA/output/cloud_alignment_audit.json"),
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print("all_ok: {}".format(payload["all_ok"]))
    print("result_file: {}".format(output_file))
    if not payload["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
