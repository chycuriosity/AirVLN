import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


GROUPS = ["A", "B", "C"]


def read_yaml(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def bool_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value, base_dir):
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def latest_file(pattern):
    files = list(pattern)
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def latest_episode_list(project_prefix, run_name, split):
    root = Path(project_prefix) / "DATA/output/{}/eval/episode_lists".format(run_name)
    return latest_file(root.glob("*/episode_list_{}.json".format(split)))


def latest_stats(project_prefix, run_name, split):
    root = Path(project_prefix) / "DATA/output/{}/eval/results".format(run_name)
    return latest_file(root.glob("*/stats_ckpt_*_{}.json".format(split)))


def group_script(group):
    if group == "A":
        return "./AirVLN/scripts/eval.sh"
    if group == "B":
        return "./AirVLN/scripts/eval_intersection_detect.sh"
    if group == "C":
        return "./AirVLN/scripts/eval_intersection_correct.sh"
    raise ValueError("Unsupported group: {}".format(group))


def build_env(spec, group, episode_list_path=None):
    run_names = spec.get("run_names") or {}
    ports = spec.get("simulator_ports") or {}
    cuda_devices = spec.get("cuda_visible_devices") or {}
    resume_times = spec.get("resume_times") or {}
    intersection = spec.get("intersection") or {}
    correction = spec.get("correction") or {}

    env = os.environ.copy()
    env.update({
        "LOCAL_EVAL_NAME": str(run_names.get(group) or "AirVLN-seq2seq"),
        "LOCAL_EVAL_CKPT_PATH": str(spec["checkpoint_path"]),
        "LOCAL_EVAL_DATASET": str(spec.get("dataset", "val_unseen")),
        "LOCAL_EVAL_NUM": str(spec.get("eval_num", -1)),
        "LOCAL_EVAL_MAX_ACTION": str(spec.get("max_action", 500)),
        "LOCAL_EVAL_BATCH_SIZE": str(spec.get("batch_size", 1)),
        "LOCAL_EVAL_CUDA_VISIBLE_DEVICES": str(cuda_devices.get(group, cuda_devices.get("default", "0"))),
        "LOCAL_EVAL_GPU_DEVICE": str(spec.get("gpu_device", 0)),
        "LOCAL_SIMULATOR_TOOL_PORT": str(ports.get(group, ports.get("default", 30000))),
    })

    cloud_config = spec.get("cloud_config")
    if cloud_config:
        env["LOCAL_EVAL_CLOUD_CONFIG"] = str(cloud_config)

    if bool_value(spec.get("resume")):
        env["LOCAL_EVAL_RESUME"] = "1"
        resume_time = resume_times.get(group)
        if resume_time:
            env["LOCAL_EVAL_RESUME_TIME"] = str(resume_time)

    if group == "A" and bool_value(spec.get("save_episode_list"), True):
        env["LOCAL_EVAL_SAVE_EPISODE_LIST"] = "1"

    if group in {"B", "C"}:
        env.update({
            "LOCAL_INTERSECTION_WRONG_POLICY": str(intersection.get("wrong_policy", "branch_mismatch")),
            "LOCAL_INTERSECTION_CANDIDATE_MODE": str(intersection.get("candidate_mode", "strict")),
            "LOCAL_INTERSECTION_TURN_WINDOW": str(intersection.get("turn_window", 4)),
            "LOCAL_INTERSECTION_MAX_EVENTS_PER_EPISODE": str(intersection.get("max_events_per_episode", -1)),
            "LOCAL_INTERSECTION_CLOUD_CONFIDENCE_THRESHOLD": str(intersection.get("cloud_confidence_threshold", 0.5)),
        })
        if bool_value(intersection.get("save_inputs")):
            env["LOCAL_INTERSECTION_SAVE_INPUTS"] = "1"
        if episode_list_path:
            env["LOCAL_EVAL_EPISODE_LIST_PATH"] = str(episode_list_path)

    if group == "C":
        env.update({
            "LOCAL_INTERSECTION_COOLDOWN_STEPS": str(correction.get("cooldown_steps", 8)),
            "LOCAL_INTERSECTION_MAX_CORRECTIONS_PER_EPISODE": str(correction.get("max_corrections_per_episode", 3)),
            "LOCAL_INTERSECTION_MAX_CORRECTIONS_PER_CLUSTER": str(correction.get("max_corrections_per_cluster", 1)),
        })

    return env


def public_env(env):
    keys = sorted(key for key in env if key.startswith("LOCAL_"))
    return {key: env[key] for key in keys}


def run_group(project_prefix, experiment_dir, group, spec, episode_list_path, execute):
    script = group_script(group)
    env = build_env(spec, group, episode_list_path=episode_list_path)
    command = ["bash", script]
    step = {
        "group": group,
        "script": script,
        "cwd": str(project_prefix),
        "env": public_env(env),
        "command": " ".join(command),
        "executed": bool(execute),
    }
    if not execute:
        return step

    log_path = experiment_dir / "logs" / "{}.log".format(group)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(log_path), "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            command,
            cwd=str(project_prefix),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    step["returncode"] = proc.returncode
    step["log_path"] = str(log_path)
    if proc.returncode != 0:
        raise RuntimeError("{} group failed with code {}; see {}".format(group, proc.returncode, log_path))
    return step


def maybe_compare(project_prefix, experiment_dir, spec):
    compare = spec.get("compare") or {}
    if not bool_value(compare.get("enabled"), True):
        return None

    run_names = spec.get("run_names") or {}
    split = str(spec.get("dataset", "val_unseen"))
    stats = {
        group: latest_stats(project_prefix, run_names.get(group), split)
        for group in GROUPS
    }
    if not all(stats.values()):
        return {
            "skipped": True,
            "reason": "missing stats file",
            "stats": {group: str(path) if path else None for group, path in stats.items()},
        }

    output = experiment_dir / "reports" / "local_abc_report.md"
    command = [
        sys.executable,
        str(repo_root() / "src/vlnce_src/compare_local_abc_runs.py"),
        "--a",
        str(stats["A"]),
        "--b",
        str(stats["B"]),
        "--c",
        str(stats["C"]),
        "--output",
        str(output),
    ]
    if bool_value(compare.get("strict"), True):
        command.append("--strict")
    proc = subprocess.run(command, cwd=str(repo_root()), text=True)
    return {
        "skipped": False,
        "returncode": proc.returncode,
        "output": str(output),
        "command": " ".join(command),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare or execute a local AirVLN A/B/C experiment.")
    parser.add_argument("config", help="Path to local_abc_experiment.yaml")
    parser.add_argument("--execute", action="store_true", help="Run A/B/C sequentially. Without this flag only writes a plan.")
    args = parser.parse_args()

    config_path = resolve_path(args.config, repo_root())
    spec = read_yaml(config_path)
    project_prefix = resolve_path(spec.get("project_prefix") or repo_root().parent, repo_root())
    spec["project_prefix"] = str(project_prefix)
    spec["checkpoint_path"] = str(resolve_path(spec.get("checkpoint_path"), project_prefix))
    if args.execute and not Path(spec["checkpoint_path"]).exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(spec["checkpoint_path"]))

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = str(spec.get("experiment_name", "local_abc"))
    experiment_dir = Path(project_prefix) / "DATA/output/experiments/{}_{}".format(experiment_name, run_id)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(config_path), str(experiment_dir / "experiment_spec.yaml"))

    enabled_groups = spec.get("groups") or {}
    steps = []
    episode_list_path = resolve_path(spec.get("episode_list_path"), project_prefix)

    for group in GROUPS:
        if not bool_value(enabled_groups.get(group), True):
            continue
        if group in {"B", "C"} and episode_list_path is None:
            episode_list_path = latest_episode_list(
                project_prefix,
                (spec.get("run_names") or {}).get("A", "AirVLN-seq2seq"),
                str(spec.get("dataset", "val_unseen")),
            )
            if episode_list_path is None and not args.execute:
                episode_list_path = "<auto: generated by A group during --execute>"
            if episode_list_path is None and args.execute:
                raise FileNotFoundError("No episode list found for {} group; run A first or set episode_list_path".format(group))
        step = run_group(project_prefix, experiment_dir, group, spec, episode_list_path, args.execute)
        steps.append(step)
        if args.execute and group == "A" and episode_list_path is None:
            episode_list_path = latest_episode_list(
                project_prefix,
                (spec.get("run_names") or {}).get("A", "AirVLN-seq2seq"),
                str(spec.get("dataset", "val_unseen")),
            )
            if episode_list_path is None:
                raise FileNotFoundError("A group finished but no episode list was generated")

    compare_result = None
    if args.execute:
        compare_result = maybe_compare(project_prefix, experiment_dir, spec)

    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "config": str(config_path),
        "experiment_dir": str(experiment_dir),
        "episode_list_path": str(episode_list_path) if episode_list_path else None,
        "steps": steps,
        "compare": compare_result,
    }
    write_json(experiment_dir / "experiment_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
    if compare_result and int(compare_result.get("returncode", 0)) != 0:
        sys.exit(int(compare_result["returncode"]))


if __name__ == "__main__":
    main()
