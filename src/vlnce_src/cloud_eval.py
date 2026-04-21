import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

# Make repository root importable when running as:
# python -u ./src/vlnce_src/cloud_eval.py ...
sys.path.append(str(Path(str(os.getcwd())).resolve()))

# Parse cloud-specific arguments first and leave the rest to src.common.param.
_cloud_parser = argparse.ArgumentParser(add_help=False)
_cloud_parser.add_argument("--cloud_model", default="qwen3.5-flash")
_cloud_parser.add_argument("--enable_thinking", action="store_true")
_cloud_parser.add_argument("--cloud_max_retries", type=int, default=3)
CLOUD_ARGS, _remaining_argv = _cloud_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("Please install openai first: pip install openai") from e

from utils.logger import logger
from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import (
    append_text_to_image,
    generate_video,
    get_checkpoint_id,
    poll_checkpoint_folder,
    observations_to_image,
)
from airsim_plugin.airsim_settings import AirsimActions


def setup():
    seed = 100
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_tokenizer():
    if args.tokenizer_use_bert:
        from transformers import BertTokenizer

        tok = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    return tok


def build_openai_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("DASHSCOPE_API_KEY is not set")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )


def summarize_obs(obs: Dict) -> Dict:
    pose = obs["pose"].tolist() if "pose" in obs else []

    depth_stats = {}
    if "depth" in obs and obs["depth"] is not None:
        depth = np.asarray(obs["depth"])
        depth_stats = {
            "mean": float(depth.mean()),
            "min": float(depth.min()),
            "max": float(depth.max()),
        }

    rgb_stats = {}
    if "rgb" in obs and obs["rgb"] is not None:
        rgb = np.asarray(obs["rgb"])
        rgb_stats = {
            "mean_rgb": [float(rgb[..., i].mean()) for i in range(3)],
            "std_rgb": [float(rgb[..., i].std()) for i in range(3)],
        }

    return {
        "pose": pose,
        "progress": float(obs.get("progress", 0.0)),
        "depth_stats": depth_stats,
        "rgb_stats": rgb_stats,
    }


def parse_action(raw_text: str) -> Tuple[int, str]:
    action_name_to_id = {
        "STOP": AirsimActions.STOP,
        "MOVE_FORWARD": AirsimActions.MOVE_FORWARD,
        "TURN_LEFT": AirsimActions.TURN_LEFT,
        "TURN_RIGHT": AirsimActions.TURN_RIGHT,
        "GO_UP": AirsimActions.GO_UP,
        "GO_DOWN": AirsimActions.GO_DOWN,
        "MOVE_LEFT": AirsimActions.MOVE_LEFT,
        "MOVE_RIGHT": AirsimActions.MOVE_RIGHT,
    }

    text = raw_text.strip()
    # Remove fenced markdown wrappers if present
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        action_name = str(data.get("action", "")).upper().strip()
        if action_name in action_name_to_id:
            return action_name_to_id[action_name], action_name
    except Exception:
        pass

    for key in action_name_to_id:
        if key in text.upper():
            return action_name_to_id[key], key

    return AirsimActions.STOP, "STOP"


def cloud_infer_action(
    client: OpenAI,
    instruction_text: str,
    obs_summary: Dict,
) -> Tuple[int, str, str]:
    system_prompt = (
        "You are a UAV navigation policy. Return only JSON like "
        "{\"action\":\"MOVE_FORWARD\"}. Valid actions: "
        "STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, GO_UP, GO_DOWN, MOVE_LEFT, MOVE_RIGHT."
    )

    user_payload = {
        "instruction": instruction_text,
        "observation_summary": obs_summary,
    }

    completion = client.chat.completions.create(
        model=CLOUD_ARGS.cloud_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        extra_body={"enable_thinking": CLOUD_ARGS.enable_thinking},
        stream=False,
    )

    raw_text = completion.choices[0].message.content or ""
    action_id, action_name = parse_action(raw_text)
    return action_id, action_name, raw_text


def cloud_infer_action_with_retry(
    client: OpenAI,
    instruction_text: str,
    obs_summary: Dict,
) -> Tuple[int, str, str]:
    for retry in range(max(1, CLOUD_ARGS.cloud_max_retries)):
        try:
            return cloud_infer_action(client, instruction_text, obs_summary)
        except Exception as e:
            logger.error(f"cloud inference failed retry={retry}: {e}")
            time.sleep(1)

    return AirsimActions.STOP, "STOP", "{\"action\":\"STOP\",\"reason\":\"cloud_retries_exhausted\"}"


def _build_eval_env(tok):
    if args.EVAL_DATASET == "train":
        return AirVLNENV(batch_size=args.batchSize, split="train", tokenizer=tok)
    if args.EVAL_DATASET == "val_seen":
        return AirVLNENV(batch_size=args.batchSize, split="val_seen", tokenizer=tok)
    if args.EVAL_DATASET == "val_unseen":
        return AirVLNENV(batch_size=args.batchSize, split="val_unseen", tokenizer=tok)
    if args.EVAL_DATASET == "test":
        return AirVLNENV(batch_size=args.batchSize, split="test", tokenizer=tok)
    raise KeyError("Unknown EVAL_DATASET")


def _eval_cloud_checkpoint(
    writer,
    tok,
    client: OpenAI,
    checkpoint_index: int = 0,
):
    train_env = _build_eval_env(tok)

    eval_results_dir = (
        Path(args.project_prefix)
        / f"DATA/output/{args.name}/eval/results/{args.make_dir_time}"
    )
    fname = os.path.join(
        eval_results_dir,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        logger.info("skipping -- evaluation exists.")
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    stats_episodes = {}
    episodes_to_eval = len(train_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    start_iter = 0
    end_iter = len(train_env.data)
    cnt = 0

    for idx in range(start_iter, end_iter, train_env.batch_size):
        if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
            break
        cnt += 1

        train_env.next_minibatch()
        if train_env.batch is None:
            logger.warning("train_env.batch is None, stop eval")
            break

        rgb_frames = [[] for _ in range(train_env.batch_size)]
        skips = [False for _ in range(train_env.batch_size)]

        outputs = train_env.reset()
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        for t in range(int(args.maxAction)):
            actions = []
            for i in range(train_env.batch_size):
                if dones[i]:
                    actions.append(AirsimActions.STOP)
                    continue

                obs_summary = summarize_obs(observations[i])
                action_id, action_name, raw_text = cloud_infer_action_with_retry(
                    client=client,
                    instruction_text=train_env.batch[i]["instruction"]["instruction_text"],
                    obs_summary=obs_summary,
                )
                logger.info(
                    f"checkpoint_index:{checkpoint_index} idx:{idx} step:{t} env:{i} "
                    f"action:{action_name}({action_id}) raw:{raw_text}"
                )
                actions.append(action_id)

            train_env.makeActions(actions)

            outputs = train_env.get_obs()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            for i in range(train_env.batch_size):
                if args.EVAL_GENERATE_VIDEO:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame,
                        train_env.batch[i]["instruction"]["instruction_text"],
                    )
                    rgb_frames[i].append(frame)

                if not dones[i] or skips[i]:
                    continue

                skips[i] = True
                pbar.update()

            if np.array(dones).all():
                break

        for i in range(int(train_env.batch_size)):
            stats_episodes[str(train_env.batch[i]["episode_id"])] = infos[i]

            eval_save_every_dir = (
                Path(args.project_prefix)
                / f"DATA/output/{args.name}/eval/intermediate_results_every/{args.make_dir_time}"
            )
            target_dir = eval_save_every_dir / str(checkpoint_index)
            if not os.path.exists(str(target_dir)):
                os.makedirs(str(target_dir), exist_ok=True)

            f_intermediate_result_name = os.path.join(
                str(target_dir),
                f"{train_env.batch[i]['episode_id']}.json",
            )
            with open(f_intermediate_result_name, "w") as f:
                json.dump(infos[i], f)

            if args.EVAL_GENERATE_VIDEO:
                eval_video_dir = (
                    Path(args.project_prefix)
                    / f"DATA/output/{args.name}/eval/videos/{args.make_dir_time}"
                )
                generate_video(
                    video_option=["disk"],
                    video_dir=str(eval_video_dir),
                    images=rgb_frames[i],
                    episode_id=train_env.batch[i]["episode_id"],
                    checkpoint_idx=checkpoint_index,
                    metrics={"ndtw": infos[i]["ndtw"]},
                    tb_writer=writer,
                )

    pbar.close()

    eval_intermediate_dir = (
        Path(args.project_prefix)
        / f"DATA/output/{args.name}/eval/intermediate_results/{args.make_dir_time}"
    )
    f_intermediate_name = os.path.join(
        eval_intermediate_dir,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(eval_intermediate_dir):
        os.makedirs(eval_intermediate_dir, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f)

    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = j.copy()
        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if isinstance(_j, (str, list, dict)):
                del temp_1[_i]
        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
        )

    fname = os.path.join(
        eval_results_dir,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except Exception:
        pass


def eval_vlnce_cloud():
    logger.info(args)
    logger.info(
        f"cloud_model={CLOUD_ARGS.cloud_model} enable_thinking={CLOUD_ARGS.enable_thinking}"
    )

    writer = TensorboardWriter(
        str(
            Path(args.project_prefix)
            / f"DATA/output/{args.name}/eval/TensorBoard/{args.make_dir_time}"
        ),
        flush_secs=30,
    )

    tok = initialize_tokenizer()
    client = build_openai_client()

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), "The eval file/folder does not exist"
    if os.path.isfile(args.EVAL_CKPT_PATH_DIR):
        proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
        ckpt_idx = proposed_index if proposed_index is not None else 100000
        _eval_cloud_checkpoint(
            writer=writer,
            tok=tok,
            client=client,
            checkpoint_index=ckpt_idx,
        )
        logger.info("END evaluate")
    else:
        prev_ckpt_ind = -1
        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                )
                time.sleep(2)
            logger.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += 1

            if prev_ckpt_ind <= 2:
                continue

            _eval_cloud_checkpoint(
                writer=writer,
                tok=tok,
                client=client,
                checkpoint_index=prev_ckpt_ind,
            )

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)

    logger.info("END evaluate")


if __name__ == "__main__":
    setup()

    if args.run_type != "eval":
        raise NotImplementedError("cloud_eval.py only supports --run_type eval")

    eval_vlnce_cloud()
