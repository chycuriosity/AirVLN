import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "Please install openai first: pip install openai"
    ) from e

# Make sure project root is importable when running as:
# python -u ./scripts/cloud_closed_loop_demo.py
sys.path.append(str(Path(str(os.getcwd())).resolve()))

# Reuse project tokenizer/env stack
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer
from src.common.param import args
from airsim_plugin.airsim_settings import AirsimActions


ACTION_NAME_TO_ID = {
    "STOP": AirsimActions.STOP,
    "MOVE_FORWARD": AirsimActions.MOVE_FORWARD,
    "TURN_LEFT": AirsimActions.TURN_LEFT,
    "TURN_RIGHT": AirsimActions.TURN_RIGHT,
    "GO_UP": AirsimActions.GO_UP,
    "GO_DOWN": AirsimActions.GO_DOWN,
    "MOVE_LEFT": AirsimActions.MOVE_LEFT,
    "MOVE_RIGHT": AirsimActions.MOVE_RIGHT,
}


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


def initialize_tokenizer():
    vocab = read_vocab(args.TRAIN_VOCAB)
    return Tokenizer(vocab=vocab, encoding_length=args.maxInput)


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
    text = raw_text.strip()

    # 1) Try JSON first
    try:
        data = json.loads(text)
        action_name = str(data.get("action", "")).upper().strip()
        if action_name in ACTION_NAME_TO_ID:
            return ACTION_NAME_TO_ID[action_name], action_name
    except Exception:
        pass

    # 2) Try to capture first known action token in free text
    pattern = r"\b(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|GO_UP|GO_DOWN|MOVE_LEFT|MOVE_RIGHT)\b"
    matched = re.search(pattern, text.upper())
    if matched:
        action_name = matched.group(1)
        return ACTION_NAME_TO_ID[action_name], action_name

    # fallback: stop for safety
    return AirsimActions.STOP, "STOP"


def cloud_infer_action(
    client: OpenAI,
    model_name: str,
    instruction_text: str,
    obs_summary: Dict,
    enable_thinking: bool,
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
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        extra_body={"enable_thinking": enable_thinking},
        stream=False,
    )

    raw_text = completion.choices[0].message.content or ""
    action_id, action_name = parse_action(raw_text)
    return action_id, action_name, raw_text


def main():
    parser = argparse.ArgumentParser("Cloud LLM closed-loop demo")
    parser.add_argument("--split", default="val_unseen", choices=["train", "val_seen", "val_unseen", "test"])
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--model", default="qwen3.5-flash")
    parser.add_argument("--enable_thinking", action="store_true")
    demo_args = parser.parse_args()

    # force single-env demo for easier cloud-loop debugging
    tok = initialize_tokenizer()
    env = AirVLNENV(batch_size=1, split=demo_args.split, tokenizer=tok)
    client = build_openai_client()

    try:
        env.next_minibatch()
        outputs = env.reset()
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        print("=" * 20 + " Cloud closed-loop demo start " + "=" * 20)
        print(f"episode_id={env.batch[0]['episode_id']} trajectory_id={env.batch[0]['trajectory_id']}")
        print("instruction:")
        print(env.batch[0]["instruction"]["instruction_text"])

        for t in range(demo_args.max_steps):
            if dones[0]:
                break

            obs_summary = summarize_obs(observations[0])
            action_id, action_name, raw_text = cloud_infer_action(
                client=client,
                model_name=demo_args.model,
                instruction_text=env.batch[0]["instruction"]["instruction_text"],
                obs_summary=obs_summary,
                enable_thinking=demo_args.enable_thinking,
            )

            print(f"[step={t}] cloud_action={action_name} ({action_id})")
            print(f"[step={t}] cloud_raw={raw_text}")

            env.makeActions([action_id])
            outputs = env.get_obs()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        print("=" * 20 + " Final metrics " + "=" * 20)
        print(json.dumps(infos[0], ensure_ascii=False, indent=2))

    finally:
        try:
            env.simulator_tool.closeScenes()
        except Exception:
            pass


if __name__ == "__main__":
    main()
