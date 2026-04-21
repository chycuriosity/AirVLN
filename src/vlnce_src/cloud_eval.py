import gc
import csv
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(str(os.getcwd())).resolve()))

import numpy as np
import tqdm

from Model.utils.common import append_text_to_image, generate_video, observations_to_image
from Model.utils.tensorboard_utils import TensorboardWriter
from airsim_plugin.airsim_settings import AirsimActions
from src.common.param import args
from src.vlnce_src.cloud_model import ACTION_ID_TO_NAME, PROMPT_VERSION, CloudActionClient
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import Tokenizer, read_vocab
from utils.logger import logger


def args_for_logging():
    display_args = vars(args).copy()
    if display_args.get("cloud_api_key"):
        display_args["cloud_api_key"] = "***"
    return display_args


def args_snapshot():
    snapshot = {}
    for key, value in vars(args).items():
        if key == "cloud_api_key" and value:
            snapshot[key] = "***"
        elif isinstance(value, Path):
            snapshot[key] = str(value)
        else:
            snapshot[key] = value
    return snapshot


def setup():
    seed = 100
    random.seed(seed)
    np.random.seed(seed)


def apply_smoke_overrides():
    if not args.cloud_smoke_test:
        return
    args.EVAL_NUM = int(args.cloud_smoke_eval_num)
    args.maxAction = int(args.cloud_smoke_max_action)
    args.EVAL_GENERATE_VIDEO = True
    args.cloud_save_input_images = True
    args.cloud_save_request_json = True
    logger.warning(
        "cloud_smoke_test enabled: EVAL_NUM={}, maxAction={}, save_input_images=True".format(
            args.EVAL_NUM,
            args.maxAction,
        )
    )


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained("bert-base-uncased")

    vocab = read_vocab(args.TRAIN_VOCAB)
    return Tokenizer(vocab=vocab, encoding_length=args.maxInput)


def initialize_env(tok):
    split_map = {
        "train": "train",
        "val_seen": "val_seen",
        "val_unseen": "val_unseen",
        "test": "test",
    }
    if args.EVAL_DATASET not in split_map:
        raise KeyError("Unsupported EVAL_DATASET: {}".format(args.EVAL_DATASET))
    return AirVLNENV(batch_size=args.batchSize, split=split_map[args.EVAL_DATASET], tokenizer=tok)


def initialize_eval_env(tok):
    original_eval_num = args.EVAL_NUM
    if args.cloud_episode_list_path:
        args.EVAL_NUM = -1
    try:
        train_env = initialize_env(tok)
    finally:
        args.EVAL_NUM = original_eval_num

    if args.cloud_episode_list_path:
        episode_ids = load_episode_list(args.cloud_episode_list_path)
        filter_env_episodes(train_env, episode_ids)
    return train_env


def safe_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def write_json(path, payload, indent=None):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path, payload):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def resolve_path(path_value):
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(str(os.getcwd())).resolve() / path
    return path


def validate_config():
    errors = []
    warnings = []

    if int(args.batchSize) != 1:
        warnings.append("云端评测建议 batchSize=1，当前为 {}".format(args.batchSize))
    if not args.cloud_model:
        errors.append("cloud_model 不能为空")
    if not args.cloud_base_url:
        errors.append("cloud_base_url 不能为空")
    if not args.cloud_api_key and not os.getenv(args.cloud_api_key_env):
        errors.append("cloud_api_key 为空，且环境变量 {} 未设置".format(args.cloud_api_key_env))
    if args.cloud_depth_mode not in {"none", "summary", "image", "both"}:
        errors.append("cloud_depth_mode 必须是 none/summary/image/both")
    if int(args.cloud_depth_grid_size) < 1:
        errors.append("cloud_depth_grid_size 必须 >= 1")
    if float(args.cloud_depth_near_percentile) >= float(args.cloud_depth_far_percentile):
        errors.append("cloud_depth_near_percentile 必须小于 cloud_depth_far_percentile")
    if int(args.maxAction) <= 0:
        errors.append("maxAction 必须 > 0")
    if int(args.EVAL_NUM) == 0:
        errors.append("EVAL_NUM 不能为 0；使用 -1 表示全量")
    if args.cloud_use_memory and int(args.cloud_max_tokens) < 96:
        warnings.append("cloud_use_memory=true 时建议 cloud_max_tokens >= 96")
    if not args.cloud_verify_ssl:
        warnings.append("cloud_verify_ssl=false，当前会跳过 HTTPS 证书校验")

    for key in ["cloud_prompt_system_path", "cloud_prompt_user_path", "cloud_episode_list_path"]:
        path = resolve_path(getattr(args, key, None))
        if path is not None and not path.exists():
            errors.append("{} 指向的文件不存在: {}".format(key, path))

    if errors:
        for item in errors:
            logger.error("config error: {}".format(item))
        raise ValueError("Invalid cloud evaluation config: {}".format("; ".join(errors)))
    for item in warnings:
        logger.warning("config warning: {}".format(item))


def load_episode_list(path_value):
    path = resolve_path(path_value)
    payload = read_json(path)
    if isinstance(payload, dict):
        episode_ids = payload.get("episode_ids", [])
    else:
        episode_ids = payload
    if not isinstance(episode_ids, list) or not episode_ids:
        raise ValueError("Episode list must be a non-empty JSON list or contain episode_ids: {}".format(path))
    return [str(item) for item in episode_ids]


def filter_env_episodes(train_env, episode_ids):
    order = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    filtered = [item for item in train_env.data if str(item["episode_id"]) in order]
    filtered.sort(key=lambda item: order[str(item["episode_id"])])
    missing = [episode_id for episode_id in episode_ids if episode_id not in {str(item["episode_id"]) for item in filtered}]
    if missing:
        raise ValueError("Episode list contains {} ids not found in split {}. First missing id: {}".format(
            len(missing),
            train_env.split,
            missing[0],
        ))
    train_env.data = filtered
    train_env.index_data = 0
    train_env.scenes = set(item["scene_id"] for item in train_env.data)
    logger.info("Filtered evaluation data to {} fixed episode ids.".format(len(train_env.data)))


def episode_list_payload(train_env, evaluated_ids=None):
    ids = [str(item["episode_id"]) for item in train_env.data]
    if evaluated_ids is not None:
        ids = [str(item) for item in evaluated_ids]
    return {
        "split": train_env.split,
        "eval_num": args.EVAL_NUM,
        "episode_count": len(ids),
        "episode_ids": ids,
    }


def log_episode_result(index, info):
    logger.info((
        "result-{} \t"
        "distance_to_goal: {} \t"
        "success: {} \t"
        "ndtw: {} \t"
        "sdtw: {} \t"
        "path_length: {} \t"
        "oracle_success: {} \t"
        "steps_taken: {}"
    ).format(
        index,
        info["distance_to_goal"],
        info["success"],
        info["ndtw"],
        info["sdtw"],
        info["path_length"],
        info["oracle_success"],
        info["steps_taken"],
    ))


def load_completed_episode_ids(model_tag):
    if not args.cloud_resume:
        return set()

    root = Path(args.project_prefix) / "DATA/output/{}/eval/intermediate_results_every".format(args.name)
    completed = set()
    if not root.exists():
        return completed

    for result_file in root.glob("*/cloud_{}/*.json".format(model_tag)):
        completed.add(result_file.stem)

    logger.info("Loaded {} completed episodes for resume.".format(len(completed)))
    return completed


def update_api_stats(api_stats, meta):
    api_stats["requested_steps"] += 1
    api_stats["attempts"] += int(meta.get("attempts") or 0)
    latency = meta.get("latency_sec")
    if latency is not None:
        api_stats["latency_sec_total"] += float(latency)
        api_stats["latency_sec_max"] = max(api_stats["latency_sec_max"], float(latency))
    if meta.get("error"):
        api_stats["error_steps"] += 1
    if meta.get("fallback_used"):
        api_stats["fallback_steps"] += 1


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def compact_info(info):
    keep_keys = [
        "episode_id",
        "trajectory_id",
        "done",
        "is_collisioned",
        "distance_to_goal",
        "success",
        "ndtw",
        "sdtw",
        "path_length",
        "oracle_success",
        "steps_taken",
    ]
    return {key: to_jsonable(info.get(key)) for key in keep_keys if key in info}


def pose_list(observation):
    pose = observation.get("pose", [])
    try:
        return [round(float(item), 4) for item in np.asarray(pose).flatten().tolist()]
    except Exception:
        return []


def observation_summary(cloud_client, observation):
    summary = {
        "pose": pose_list(observation),
        "has_rgb": bool("rgb" in observation and observation["rgb"] is not None),
        "has_depth": bool("depth" in observation and observation["depth"] is not None),
    }
    if summary["has_depth"]:
        try:
            summary["depth_summary"] = to_jsonable(cloud_client._depth_summary(observation["depth"]))
        except Exception as exc:
            summary["depth_summary_error"] = repr(exc)
    return summary


def write_trajectory_trace(path, episode, actions, final_info):
    payload = {
        "episode_id": str(episode["episode_id"]),
        "trajectory_id": episode.get("trajectory_id"),
        "instruction": episode["instruction"]["instruction_text"],
        "final_info": to_jsonable(final_info),
        "steps": to_jsonable(actions),
    }
    write_json(path, payload, indent=2)


def max_consecutive_action(actions):
    max_run = 0
    current_run = 0
    previous = None
    for item in actions:
        action_id = item.get("action_id")
        if action_id == previous:
            current_run += 1
        else:
            current_run = 1
            previous = action_id
        max_run = max(max_run, current_run)
    return max_run


def behavior_diagnostics(info, actions):
    action_ids = [item.get("action_id") for item in actions]
    action_names = [item.get("action_name") for item in actions]
    counts = {}
    for name in action_names:
        counts[name] = counts.get(name, 0) + 1

    stop_steps = [item.get("step") for item in actions if item.get("action_id") == AirsimActions.STOP]
    turn_ids = {AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT}
    turn_count = sum(1 for action_id in action_ids if action_id in turn_ids)
    forward_count = sum(1 for action_id in action_ids if action_id == AirsimActions.MOVE_FORWARD)
    side_count = sum(1 for action_id in action_ids if action_id in {AirsimActions.MOVE_LEFT, AirsimActions.MOVE_RIGHT})

    labels = []
    if stop_steps and stop_steps[0] <= 2 and float(info.get("success", 0.0)) < 1.0:
        labels.append("early_stop")
    if int(info.get("is_collisioned", 0)) == 1 and action_ids and action_ids[-1] == AirsimActions.MOVE_FORWARD:
        labels.append("collision_after_forward")
    if len(actions) >= 4 and max_consecutive_action(actions) >= 4:
        labels.append("repeated_same_action")
    if len(actions) >= 6 and turn_count / float(len(actions)) >= 0.7:
        labels.append("turning_loop")
    if len(actions) >= 6 and forward_count == 0 and float(info.get("success", 0.0)) < 1.0:
        labels.append("no_forward_progress")

    return {
        "labels": labels,
        "action_counts": counts,
        "max_consecutive_same_action": max_consecutive_action(actions),
        "stop_step": stop_steps[0] if stop_steps else None,
        "turn_ratio": turn_count / float(len(actions)) if actions else 0.0,
        "forward_ratio": forward_count / float(len(actions)) if actions else 0.0,
        "side_move_ratio": side_count / float(len(actions)) if actions else 0.0,
    }


def get_git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def write_run_snapshot(results_dir, cloud_client):
    write_json(
        results_dir / "config_snapshot.json",
        {
            "args": args_snapshot(),
            "prompt": cloud_client.get_prompt_metadata(),
        },
        indent=2,
    )


def write_summary_csv(path, aggregated_stats):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(aggregated_stats.keys()):
            writer.writerow([key, aggregated_stats[key]])


def write_failure_analysis(path, stats_episodes, episode_action_summaries, api_stats):
    payload = {
        "summary": {
            "episodes": len(stats_episodes),
            "failed_episodes": 0,
            "collision_episodes": 0,
            "timeout_like_episodes": 0,
            "fallback_steps": api_stats["fallback_steps"],
            "error_steps": api_stats["error_steps"],
            "behavior_labels": {},
        },
        "episodes": {},
    }

    for episode_id, info in stats_episodes.items():
        actions = episode_action_summaries.get(str(episode_id), [])
        behavior = behavior_diagnostics(info, actions)
        reasons = []
        if float(info.get("success", 0.0)) < 1.0:
            reasons.append("not_success")
            payload["summary"]["failed_episodes"] += 1
        if int(info.get("is_collisioned", 0)) == 1:
            reasons.append("collision")
            payload["summary"]["collision_episodes"] += 1
        if int(info.get("steps_taken", 0)) >= int(args.maxAction):
            reasons.append("max_action_reached")
            payload["summary"]["timeout_like_episodes"] += 1
        if any(item.get("fallback_used") for item in actions):
            reasons.append("cloud_fallback_used")
        if float(info.get("ndtw", 0.0)) < 0.2:
            reasons.append("very_low_ndtw")
        reasons.extend(behavior["labels"])

        for label in behavior["labels"]:
            payload["summary"]["behavior_labels"][label] = (
                payload["summary"]["behavior_labels"].get(label, 0) + 1
            )

        payload["episodes"][str(episode_id)] = {
            "reasons": reasons,
            "behavior": behavior,
            "distance_to_goal": info.get("distance_to_goal"),
            "success": info.get("success"),
            "ndtw": info.get("ndtw"),
            "sdtw": info.get("sdtw"),
            "steps_taken": info.get("steps_taken"),
            "path_length": info.get("path_length"),
            "actions": [
                {
                    "step": item.get("step"),
                    "action_id": item.get("action_id"),
                    "action_name": item.get("action_name"),
                    "fallback_used": item.get("fallback_used"),
                    "error": item.get("error"),
                    "latency_sec": item.get("latency_sec"),
                    "memory": item.get("memory"),
                }
                for item in actions
            ],
        }

    write_json(path, payload, indent=2)
    return payload


def format_metric(value):
    if isinstance(value, float):
        return "{:.6f}".format(value)
    return str(value)


def write_markdown_report(path, aggregated_stats, api_stats, failure_payload, manifest_path, episode_list_path):
    key_metrics = [
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
    lines = [
        "# Cloud Evaluation Report",
        "",
        "## Run",
        "",
        "- name: `{}`".format(args.name),
        "- model: `{}`".format(args.cloud_model),
        "- dataset: `{}`".format(args.EVAL_DATASET),
        "- maxAction: `{}`".format(args.maxAction),
        "- depth_mode: `{}`".format(args.cloud_depth_mode),
        "- memory: `{}`".format(args.cloud_use_memory),
        "- manifest: `{}`".format(manifest_path),
        "- episode_list: `{}`".format(episode_list_path),
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in key_metrics:
        if key in aggregated_stats:
            lines.append("| {} | {} |".format(key, format_metric(aggregated_stats[key])))

    failure_summary = failure_payload["summary"]
    lines.extend([
        "",
        "## Failures",
        "",
        "- failed_episodes: `{}`".format(failure_summary.get("failed_episodes")),
        "- collision_episodes: `{}`".format(failure_summary.get("collision_episodes")),
        "- timeout_like_episodes: `{}`".format(failure_summary.get("timeout_like_episodes")),
        "- fallback_steps: `{}`".format(api_stats.get("fallback_steps")),
        "- error_steps: `{}`".format(api_stats.get("error_steps")),
        "",
        "## Behavior Labels",
        "",
    ])
    behavior_labels = failure_summary.get("behavior_labels", {})
    if behavior_labels:
        for key, value in sorted(behavior_labels.items()):
            lines.append("- {}: `{}`".format(key, value))
    else:
        lines.append("- none")

    worst = sorted(
        failure_payload["episodes"].items(),
        key=lambda item: float(item[1].get("ndtw") or 0.0),
    )[:10]
    lines.extend([
        "",
        "## Lowest nDTW Episodes",
        "",
        "| episode_id | success | ndtw | reasons |",
        "| --- | ---: | ---: | --- |",
    ])
    for episode_id, item in worst:
        lines.append("| {} | {} | {} | {} |".format(
            episode_id,
            item.get("success"),
            format_metric(float(item.get("ndtw") or 0.0)),
            ", ".join(item.get("reasons") or []),
        ))

    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_run_manifest(path, cloud_client, train_env, model_tag, run_started_at, run_finished_at, output_paths):
    write_json(
        path,
        {
            "git_commit": get_git_commit(),
            "start_time": datetime.datetime.fromtimestamp(run_started_at).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(run_finished_at).isoformat(),
            "duration_sec": run_finished_at - run_started_at,
            "config_path": args.cloud_config,
            "name": args.name,
            "model": args.cloud_model,
            "model_tag": model_tag,
            "base_url": args.cloud_base_url,
            "split": train_env.split,
            "eval_num": args.EVAL_NUM,
            "max_action": args.maxAction,
            "depth_mode": args.cloud_depth_mode,
            "use_depth_summary": args.cloud_use_depth_summary,
            "use_memory": args.cloud_use_memory,
            "use_pose": args.cloud_use_pose,
            "generate_video": args.EVAL_GENERATE_VIDEO,
            "prompt": cloud_client.get_prompt_metadata(),
            "outputs": output_paths,
        },
        indent=2,
    )


def cloud_eval():
    validate_config()
    apply_smoke_overrides()
    if args.batchSize != 1:
        logger.warning("Cloud evaluation is most stable with --batchSize 1; current batchSize is {}".format(args.batchSize))

    logger.info(args_for_logging())
    writer = TensorboardWriter(
        str(Path(args.project_prefix) / "DATA/output/{}/eval/TensorBoard/{}".format(args.name, args.make_dir_time)),
        flush_secs=30,
    )
    run_started_at = time.time()
    cloud_client = CloudActionClient(args, logger)
    model_tag = safe_model_name(args.cloud_model)
    tok = initialize_tokenizer()
    train_env = initialize_eval_env(tok)
    completed_episode_ids = load_completed_episode_ids(model_tag)
    api_stats = {
        "requested_steps": 0,
        "attempts": 0,
        "error_steps": 0,
        "fallback_steps": 0,
        "latency_sec_total": 0.0,
        "latency_sec_max": 0.0,
    }

    results_dir = Path(args.project_prefix) / "DATA/output/{}/eval/results/{}".format(args.name, args.make_dir_time)
    result_file = results_dir / "stats_cloud_{}_{}.json".format(model_tag, train_env.split)
    if os.path.exists(str(result_file)):
        print("skipping -- evaluation exists.")
        return
    write_run_snapshot(results_dir, cloud_client)
    selected_episode_list_file = results_dir / "episode_list_selected.json"
    write_json(selected_episode_list_file, episode_list_payload(train_env), indent=2)

    stats_episodes = {}
    episode_action_summaries = {}
    evaluated_episode_ids = []
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    try:
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch()
            if train_env.batch is None:
                logger.warning("train_env.batch is None, going to break and stop eval")
                break

            batch_episode_ids = [str(item["episode_id"]) for item in train_env.batch]
            if args.cloud_resume and all(episode_id in completed_episode_ids for episode_id in batch_episode_ids):
                logger.info("skip completed episodes: {}".format(batch_episode_ids))
                pbar.update(len(batch_episode_ids))
                continue

            rgb_frames = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            infos = [{} for _ in range(train_env.batch_size)]
            action_histories = [[] for _ in range(train_env.batch_size)]

            outputs = train_env.reset()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            for step in range(int(args.maxAction)):
                logger.info("cloud_model:{} \t {} - {} / {} \t {}".format(
                    args.cloud_model,
                    idx,
                    step,
                    end_iter,
                    [int(not done) for done in dones],
                ))

                actions = []
                for env_index in range(train_env.batch_size):
                    if dones[env_index] or skips[env_index]:
                        actions.append(AirsimActions.STOP)
                        continue

                    if args.cloud_max_api_calls != -1 and api_stats["requested_steps"] >= int(args.cloud_max_api_calls):
                        logger.warning("Reached cloud_max_api_calls={}, stopping evaluation.".format(args.cloud_max_api_calls))
                        dones = [True for _ in dones]
                        actions.append(AirsimActions.STOP)
                        break

                    input_files = {}
                    if args.cloud_save_input_images:
                        input_dir = Path(args.project_prefix) / "DATA/output/{}/eval/cloud_inputs/{}/{}".format(
                            args.name,
                            args.make_dir_time,
                            train_env.batch[env_index]["episode_id"],
                        )
                        input_files = cloud_client.save_inputs(
                            observation=observations[env_index],
                            episode=train_env.batch[env_index],
                            step=step,
                            history=action_histories[env_index],
                            output_dir=input_dir,
                        )

                    action, meta = cloud_client.predict_action(
                        observation=observations[env_index],
                        episode=train_env.batch[env_index],
                        step=step,
                        history=action_histories[env_index],
                        info=infos[env_index],
                    )
                    update_api_stats(api_stats, meta)
                    actions.append(action)

                    action_record = {
                        "episode_id": train_env.batch[env_index]["episode_id"],
                        "trajectory_id": train_env.batch[env_index]["trajectory_id"],
                        "step": step,
                        "pre_observation": observation_summary(cloud_client, observations[env_index]),
                        "action_id": int(action),
                        "action_name": ACTION_ID_TO_NAME.get(int(action), "UNKNOWN"),
                        "latency_sec": meta.get("latency_sec"),
                        "attempts": meta.get("attempts"),
                        "error": meta.get("error"),
                        "fallback_used": meta.get("fallback_used"),
                        "memory": meta.get("memory"),
                        "prompt_version": meta.get("prompt_version"),
                        "prompt_hash": meta.get("prompt_hash"),
                    }
                    action_histories[env_index].append(action_record)

                    log_dir = Path(args.project_prefix) / "DATA/output/{}/eval/cloud_logs/{}".format(args.name, args.make_dir_time)
                    log_payload = {
                        **action_record,
                        "raw_response": meta.get("raw_response"),
                    }
                    if input_files:
                        log_payload["input_files"] = input_files
                    if args.cloud_save_prompts:
                        log_payload["pose"] = observations[env_index].get("pose", []).tolist()
                    append_jsonl(log_dir / "{}.jsonl".format(train_env.batch[env_index]["episode_id"]), log_payload)

                if len(actions) < train_env.batch_size:
                    actions.extend([AirsimActions.STOP for _ in range(train_env.batch_size - len(actions))])

                train_env.makeActions(actions)

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                logger.info("action: {}".format(actions))

                for env_index in range(train_env.batch_size):
                    if action_histories[env_index]:
                        action_histories[env_index][-1]["done_after_action"] = bool(dones[env_index])
                        action_histories[env_index][-1]["post_observation"] = observation_summary(
                            cloud_client,
                            observations[env_index],
                        )
                        action_histories[env_index][-1]["post_info"] = compact_info(infos[env_index])

                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[env_index], infos[env_index])
                        frame = append_text_to_image(
                            frame, train_env.batch[env_index]["instruction"]["instruction_text"]
                        )
                        rgb_frames[env_index].append(frame)

                    if not dones[env_index] or skips[env_index]:
                        continue

                    skips[env_index] = True
                    pbar.update()

                if np.array(dones).all():
                    break

            for env_index in range(int(train_env.batch_size)):
                episode_id = str(train_env.batch[env_index]["episode_id"])
                stats_episodes[episode_id] = infos[env_index]
                episode_action_summaries[episode_id] = action_histories[env_index]
                evaluated_episode_ids.append(episode_id)

                every_results_dir = Path(args.project_prefix) / "DATA/output/{}/eval/intermediate_results_every/{}/cloud_{}".format(
                    args.name,
                    args.make_dir_time,
                    model_tag,
                )
                intermediate_file = every_results_dir / "{}.json".format(train_env.batch[env_index]["episode_id"])
                write_json(intermediate_file, {**infos[env_index]})

                if args.EVAL_GENERATE_VIDEO:
                    video_dir = Path(args.project_prefix) / "DATA/output/{}/eval/videos/{}".format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(video_dir),
                        images=rgb_frames[env_index],
                        episode_id=train_env.batch[env_index]["episode_id"],
                        checkpoint_idx="cloud_{}".format(model_tag),
                        metrics={"ndtw": infos[env_index]["ndtw"]},
                        tb_writer=writer,
                    )

                log_episode_result(env_index, infos[env_index])

                trajectory_dir = Path(args.project_prefix) / "DATA/output/{}/eval/trajectories/{}/cloud_{}".format(
                    args.name,
                    args.make_dir_time,
                    model_tag,
                )
                trajectory_file = trajectory_dir / "{}.json".format(train_env.batch[env_index]["episode_id"])
                write_trajectory_trace(
                    trajectory_file,
                    train_env.batch[env_index],
                    action_histories[env_index],
                    infos[env_index],
                )

        pbar.close()

        intermediate_dir = Path(args.project_prefix) / "DATA/output/{}/eval/intermediate_results/{}".format(args.name, args.make_dir_time)
        intermediate_file = intermediate_dir / "stats_cloud_{}_{}.json".format(model_tag, train_env.split)
        write_json(intermediate_file, stats_episodes)

        numeric_stats_episodes = {}
        for episode_id, info in stats_episodes.items():
            numeric_info = info.copy()
            for key, value in list(numeric_info.items()):
                if type(value) == str or type(value) == list or type(value) == dict:
                    del numeric_info[key]
            numeric_stats_episodes[episode_id] = numeric_info

        if len(numeric_stats_episodes) == 0:
            raise RuntimeError("No episodes were evaluated.")

        aggregated_stats = {}
        num_episodes = len(numeric_stats_episodes)
        for stat_key in next(iter(numeric_stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in numeric_stats_episodes.values())
                / num_episodes
            )

        if api_stats["requested_steps"] > 0:
            api_stats["latency_sec_avg"] = api_stats["latency_sec_total"] / api_stats["requested_steps"]
        else:
            api_stats["latency_sec_avg"] = 0.0
        aggregated_stats["cloud_requested_steps"] = api_stats["requested_steps"]
        aggregated_stats["cloud_api_attempts"] = api_stats["attempts"]
        aggregated_stats["cloud_error_steps"] = api_stats["error_steps"]
        aggregated_stats["cloud_fallback_steps"] = api_stats["fallback_steps"]
        aggregated_stats["cloud_latency_sec_avg"] = api_stats["latency_sec_avg"]
        aggregated_stats["cloud_latency_sec_max"] = api_stats["latency_sec_max"]
        aggregated_stats["cloud_prompt_version"] = PROMPT_VERSION

        write_json(result_file, aggregated_stats, indent=4)
        call_stats_file = results_dir / "cloud_call_stats_{}_{}.json".format(model_tag, train_env.split)
        summary_csv_file = results_dir / "summary_cloud_{}_{}.csv".format(model_tag, train_env.split)
        failure_file = results_dir / "failure_analysis_cloud_{}_{}.json".format(model_tag, train_env.split)
        manifest_file = results_dir / "run_manifest.json"
        report_file = results_dir / "report_cloud_{}_{}.md".format(model_tag, train_env.split)
        evaluated_episode_list_file = results_dir / "episode_list_evaluated.json"
        write_json(evaluated_episode_list_file, episode_list_payload(train_env, evaluated_episode_ids), indent=2)
        write_json(call_stats_file, api_stats, indent=4)
        write_summary_csv(summary_csv_file, aggregated_stats)
        failure_payload = write_failure_analysis(failure_file, stats_episodes, episode_action_summaries, api_stats)
        write_run_manifest(
            manifest_file,
            cloud_client,
            train_env,
            model_tag,
            run_started_at,
            time.time(),
            {
                "results_dir": str(results_dir),
                "stats": str(result_file),
                "summary_csv": str(summary_csv_file),
                "cloud_call_stats": str(call_stats_file),
                "failure_analysis": str(failure_file),
                "report": str(report_file),
                "selected_episode_list": str(selected_episode_list_file),
                "evaluated_episode_list": str(evaluated_episode_list_file),
                "trajectories_dir": str(Path(args.project_prefix) / "DATA/output/{}/eval/trajectories/{}/cloud_{}".format(
                    args.name,
                    args.make_dir_time,
                    model_tag,
                )),
            },
        )
        write_markdown_report(
            report_file,
            aggregated_stats,
            api_stats,
            failure_payload,
            manifest_file,
            evaluated_episode_list_file,
        )

        logger.info("Episodes evaluated: {}".format(num_episodes))
        for key, value in aggregated_stats.items():
            if isinstance(value, (int, float)):
                logger.info("Average episode {}: {:.6f}".format(key, value))
                writer.add_scalar("eval_{}_{}".format(train_env.split, key), value, 1)
            else:
                logger.info("Average episode {}: {}".format(key, value))
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        try:
            writer.writer.close()
        except Exception as exc:
            logger.error(exc)
        try:
            train_env.simulator_tool.closeScenes()
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    setup()
    cloud_eval()
