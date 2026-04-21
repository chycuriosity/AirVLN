import base64
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(str(os.getcwd())).resolve()))

import numpy as np
from PIL import Image

from airsim_plugin.airsim_settings import AirsimActions
from src.common.param import args
from src.vlnce_src.cloud_model import ACTION_ID_TO_NAME, CloudActionClient


class StdoutLogger:
    def info(self, message):
        print(message)

    def warning(self, message):
        print("WARNING: {}".format(message))

    def error(self, message):
        print("ERROR: {}".format(message))


def image_to_data_url(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/{};base64,{}".format(fmt.lower(), encoded)


def chat(client, messages, max_tokens=64):
    started_at = time.time()
    completion = client.client.chat.completions.create(
        model=args.cloud_model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=float(args.cloud_timeout),
        stream=False,
    )
    text = completion.choices[0].message.content or ""
    return text, time.time() - started_at


def run_text_check(client):
    response, latency = chat(
        client,
        [
            {"role": "system", "content": "You are a strict test responder."},
            {"role": "user", "content": "Reply exactly: text-ok"},
        ],
        max_tokens=16,
    )
    return {
        "ok": "text-ok" in response.lower(),
        "latency_sec": latency,
        "response": response,
    }


def run_multi_image_check(client):
    red = Image.new("RGB", (64, 64), (255, 0, 0))
    blue = Image.new("RGB", (64, 64), (0, 0, 255))
    response, latency = chat(
        client,
        [
            {"role": "system", "content": "Answer briefly."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "There are two images. Return JSON with image_count and the dominant color of each image.",
                    },
                    {"type": "image_url", "image_url": {"url": image_to_data_url(red, "PNG")}},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(blue, "PNG")}},
                ],
            },
        ],
        max_tokens=96,
    )
    normalized = response.lower()
    return {
        "ok": ("2" in normalized or "two" in normalized) and "red" in normalized and "blue" in normalized,
        "latency_sec": latency,
        "response": response,
    }


def run_json_action_check(client):
    response, latency = chat(
        client,
        [
            {"role": "system", "content": "Return only JSON."},
            {
                "role": "user",
                "content": "Return exactly one JSON object for action TURN_LEFT with action_id and action_name.",
            },
        ],
        max_tokens=64,
    )
    action, memory = client.parse_response(response)
    return {
        "ok": action == AirsimActions.TURN_LEFT,
        "latency_sec": latency,
        "response": response,
        "parsed_action_id": action,
        "parsed_memory": memory,
    }


def run_depth_reasoning_check(client):
    rgb = np.full((224, 224, 3), 128, dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32)
    depth[86:170, 86:170] = 0.01
    episode = {
        "episode_id": "synthetic_depth_check",
        "trajectory_id": "synthetic_depth_check",
        "instruction": {
            "instruction_text": (
                "Move toward the goal ahead, but if the front-center depth image shows a close obstacle, "
                "choose a turn or side movement instead of moving forward."
            )
        },
    }
    observation = {
        "rgb": rgb,
        "depth": depth,
        "pose": np.zeros(7, dtype=np.float32),
    }
    action, meta = client.predict_action(observation, episode, step=0, history=[])
    return {
        "ok": (
            action not in {AirsimActions.MOVE_FORWARD, None}
            and not meta.get("fallback_used")
            and not meta.get("error")
        ),
        "latency_sec": meta.get("latency_sec"),
        "response": meta.get("raw_response"),
        "parsed_action_id": int(action),
        "parsed_action_name": ACTION_ID_TO_NAME.get(int(action), "UNKNOWN"),
        "fallback_used": meta.get("fallback_used"),
        "error": meta.get("error"),
    }


def safe_args_snapshot():
    snapshot = vars(args).copy()
    if snapshot.get("cloud_api_key"):
        snapshot["cloud_api_key"] = "***"
    for key, value in list(snapshot.items()):
        if isinstance(value, Path):
            snapshot[key] = str(value)
    return snapshot


def main():
    client = CloudActionClient(args, StdoutLogger())
    checks = {}
    for name, func in [
        ("text", run_text_check),
        ("multi_image", run_multi_image_check),
        ("json_action", run_json_action_check),
        ("depth_reasoning", run_depth_reasoning_check),
    ]:
        started_at = time.time()
        try:
            checks[name] = func(client)
        except Exception as exc:
            checks[name] = {
                "ok": False,
                "latency_sec": time.time() - started_at,
                "error": repr(exc),
            }
        print("{}: {}".format(name, "ok" if checks[name].get("ok") else "failed"))

    payload = {
        "model": args.cloud_model,
        "base_url": args.cloud_base_url,
        "all_ok": all(item.get("ok") for item in checks.values()),
        "checks": checks,
        "args": safe_args_snapshot(),
        "prompt": client.get_prompt_metadata(),
    }

    output_dir = Path(args.project_prefix) / "DATA/output/{}/eval/model_checks/{}".format(
        args.name,
        args.make_dir_time,
    )
    os.makedirs(str(output_dir), exist_ok=True)
    output_file = output_dir / "cloud_model_check.json"
    with open(str(output_file), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print("result_file: {}".format(output_file))
    if not payload["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
