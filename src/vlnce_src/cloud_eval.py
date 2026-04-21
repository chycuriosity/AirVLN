import gc
import json
import os
import random
import re
import sys
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


def setup():
    seed = 100
    random.seed(seed)
    np.random.seed(seed)


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


def safe_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def write_json(path, payload, indent=None):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def append_jsonl(path, payload):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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


def cloud_eval():
    if args.batchSize != 1:
        logger.warning("Cloud evaluation is most stable with --batchSize 1; current batchSize is {}".format(args.batchSize))

    logger.info(args_for_logging())
    writer = TensorboardWriter(
        str(Path(args.project_prefix) / "DATA/output/{}/eval/TensorBoard/{}".format(args.name, args.make_dir_time)),
        flush_secs=30,
    )
    cloud_client = CloudActionClient(args, logger)
    model_tag = safe_model_name(args.cloud_model)
    tok = initialize_tokenizer()
    train_env = initialize_env(tok)
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

    stats_episodes = {}
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
                        "action_id": int(action),
                        "action_name": ACTION_ID_TO_NAME.get(int(action), "UNKNOWN"),
                        "latency_sec": meta.get("latency_sec"),
                        "attempts": meta.get("attempts"),
                        "error": meta.get("error"),
                        "fallback_used": meta.get("fallback_used"),
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
                stats_episodes[str(train_env.batch[env_index]["episode_id"])] = infos[env_index]

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
        write_json(results_dir / "cloud_call_stats_{}_{}.json".format(model_tag, train_env.split), api_stats, indent=4)

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
