import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from airsim_plugin.airsim_settings import AirsimActions


ACTION_ID_TO_NAME = {
    AirsimActions.STOP: "STOP",
    AirsimActions.MOVE_FORWARD: "MOVE_FORWARD",
    AirsimActions.TURN_LEFT: "TURN_LEFT",
    AirsimActions.TURN_RIGHT: "TURN_RIGHT",
    AirsimActions.GO_UP: "GO_UP",
    AirsimActions.GO_DOWN: "GO_DOWN",
    AirsimActions.MOVE_LEFT: "MOVE_LEFT",
    AirsimActions.MOVE_RIGHT: "MOVE_RIGHT",
}
ACTION_NAME_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME.items()}
PROMPT_VERSION = "cloud-vln-depth-v2"


class CloudActionClient:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model = args.cloud_model
        self.depth_mode = self._parse_depth_mode(args.cloud_depth_mode)
        self.fallback_action = self._parse_fallback_action(args.cloud_fallback_action)
        self.client = self._build_client()

    def predict_action(
        self,
        observation: Dict[str, Any],
        episode: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        messages = self._build_messages(observation, episode, step, history)
        prompt_hash = self._hash_messages(messages)
        started_at = time.time()
        raw_response = ""
        error = None

        for attempt in range(int(self.args.cloud_max_retries)):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(self.args.cloud_temperature),
                    max_tokens=int(self.args.cloud_max_tokens),
                    timeout=float(self.args.cloud_timeout),
                    stream=False,
                )
                raw_response = completion.choices[0].message.content or ""
                action = self.parse_action(raw_response)
                if action is not None:
                    return action, {
                        "raw_response": raw_response,
                        "latency_sec": time.time() - started_at,
                        "attempts": attempt + 1,
                        "error": None,
                        "fallback_used": False,
                        "prompt_version": PROMPT_VERSION,
                        "prompt_hash": prompt_hash,
                    }

                error = "invalid action response"
                messages = messages + [
                    {
                        "role": "user",
                        "content": "Your previous response was invalid. Return only JSON like {\"action_id\": 1}.",
                    }
                ]
            except Exception as exc:
                error = repr(exc)
                self.logger.error("cloud action request failed on attempt {}: {}".format(attempt + 1, error))

        return self.fallback_action, {
            "raw_response": raw_response,
            "latency_sec": time.time() - started_at,
            "attempts": int(self.args.cloud_max_retries),
            "error": error,
            "fallback_action": self.fallback_action,
            "fallback_used": True,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_hash,
        }

    def save_inputs(
        self,
        observation: Dict[str, Any],
        episode: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Dict[str, str]:
        os.makedirs(str(output_dir), exist_ok=True)
        prompt_text = self._build_user_text(observation, episode, step, history)
        saved = {}

        prompt_path = output_dir / "step_{:04d}_prompt.txt".format(step)
        with open(str(prompt_path), "w", encoding="utf-8") as f:
            f.write(prompt_text)
        saved["prompt"] = str(prompt_path)

        if not self.args.cloud_no_rgb and "rgb" in observation and observation["rgb"] is not None:
            rgb_path = output_dir / "step_{:04d}_rgb.jpg".format(step)
            self._save_rgb_image(observation["rgb"], rgb_path)
            saved["rgb"] = str(rgb_path)

        if self._send_depth_image(observation):
            depth_path = output_dir / "step_{:04d}_depth.png".format(step)
            self._save_depth_image(observation["depth"], depth_path)
            saved["depth"] = str(depth_path)

        return saved

    def parse_action(self, text: str) -> Optional[int]:
        parsed = self._extract_json(text)
        if parsed is not None:
            action = self._action_from_payload(parsed)
            if action is not None:
                return action

        upper_text = text.upper()
        for name, action_id in ACTION_NAME_TO_ID.items():
            if name in upper_text:
                return action_id

        match = re.search(r"\b([0-7])\b", text)
        if match:
            return int(match.group(1))

        return None

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for cloud evaluation. Install it in the cxj conda environment."
            ) from exc

        api_key = self.args.cloud_api_key or os.getenv(self.args.cloud_api_key_env)
        if not api_key:
            raise EnvironmentError(
                "Missing cloud API key. Set cloud_api_key in config or set {} before running cloud evaluation.".format(
                    self.args.cloud_api_key_env
                )
            )

        http_client = None
        if self.args.cloud_disable_proxy or not self.args.cloud_verify_ssl:
            import httpx

            http_client = httpx.Client(
                trust_env=not bool(self.args.cloud_disable_proxy),
                verify=bool(self.args.cloud_verify_ssl),
                timeout=float(self.args.cloud_timeout),
            )

        return OpenAI(
            api_key=api_key,
            base_url=self.args.cloud_base_url,
            http_client=http_client,
        )

    def _build_messages(
        self,
        observation: Dict[str, Any],
        episode: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You control a UAV in a vision-and-language navigation benchmark. "
            "Choose exactly one discrete action for the next step. "
            "Use the RGB image for visual landmarks and the depth information to avoid collisions. "
            "Return only valid JSON with action_id and action_name. "
            "Do not include explanations."
        )
        user_text = self._build_user_text(observation, episode, step, history)

        content = [{"type": "text", "text": user_text}]
        if not self.args.cloud_no_rgb and "rgb" in observation and observation["rgb"] is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{}".format(
                            self._rgb_to_base64_jpeg(observation["rgb"])
                        )
                    },
                }
            )

        if self._send_depth_image(observation):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{}".format(
                            self._depth_to_base64_png(observation["depth"])
                        )
                    },
                }
            )

        if len(content) == 1:
            content = user_text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _build_user_text(
        self,
        observation: Dict[str, Any],
        episode: Dict[str, Any],
        step: int,
        history: List[Dict[str, Any]],
    ) -> str:
        instruction = episode["instruction"]["instruction_text"]
        pose = observation.get("pose", [])
        recent_history = self._format_action_history(history)

        lines = [
            "Navigation instruction:",
            instruction,
            "",
            "Current step: {}".format(step),
            "Available actions:",
            "0 STOP: stop and finish the episode",
            "1 MOVE_FORWARD: move forward 5 meters",
            "2 TURN_LEFT: rotate left 15 degrees",
            "3 TURN_RIGHT: rotate right 15 degrees",
            "4 GO_UP: move upward 2 meters",
            "5 GO_DOWN: move downward 2 meters",
            "6 MOVE_LEFT: move left 5 meters",
            "7 MOVE_RIGHT: move right 5 meters",
            "",
            "Recent action history:",
            json.dumps(recent_history, ensure_ascii=True),
        ]

        if self.args.cloud_use_pose:
            lines.extend([
                "",
                "Current pose [x, y, z, qw, qx, qy, qz]: {}".format(self._format_array(pose)),
            ])

        if not self.args.cloud_no_rgb and "rgb" in observation and observation["rgb"] is not None:
            lines.extend([
                "",
                "Image 1 is the RGB observation from the UAV camera.",
            ])

        if self._send_depth_image(observation):
            lines.extend([
                "",
                "Image 2 is a colorized depth map generated from the UAV depth sensor.",
                "In the depth map, red/yellow means close obstacles and blue means farther safer space.",
                "Avoid MOVE_FORWARD when the front-center region is red/yellow or the depth summary says it is very close.",
            ])

        if self._send_depth_summary(observation):
            lines.extend([
                "",
                "Depth summary:",
                json.dumps(self._depth_summary(observation["depth"]), ensure_ascii=True),
            ])

        lines.extend([
            "",
            "Return exactly one JSON object, for example:",
            "{\"action_id\": 1, \"action_name\": \"MOVE_FORWARD\"}",
        ])
        return "\n".join(lines)

    def _parse_fallback_action(self, value: str) -> int:
        normalized = str(value).strip().upper()
        if normalized.isdigit() and int(normalized) in ACTION_ID_TO_NAME:
            return int(normalized)
        if normalized in ACTION_NAME_TO_ID:
            return ACTION_NAME_TO_ID[normalized]
        raise ValueError("Invalid --cloud_fallback_action: {}".format(value))

    def _parse_depth_mode(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"none", "summary", "image", "both"}:
            raise ValueError("Invalid --cloud_depth_mode: {}".format(value))
        return normalized

    def _send_depth_summary(self, observation: Dict[str, Any]) -> bool:
        if "depth" not in observation or observation["depth"] is None:
            return False
        if self.depth_mode == "none":
            return False
        return bool(self.args.cloud_use_depth_summary) or self.depth_mode in {"summary", "both"}

    def _send_depth_image(self, observation: Dict[str, Any]) -> bool:
        if "depth" not in observation or observation["depth"] is None:
            return False
        return self.depth_mode in {"image", "both"}

    def _action_from_payload(self, payload: Dict[str, Any]) -> Optional[int]:
        if "action_id" in payload:
            try:
                action_id = int(payload["action_id"])
            except (TypeError, ValueError):
                action_id = None
            if action_id in ACTION_ID_TO_NAME:
                return action_id

        if "action_name" in payload:
            action_name = str(payload["action_name"]).strip().upper()
            if action_name in ACTION_NAME_TO_ID:
                return ACTION_NAME_TO_ID[action_name]

        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()

        candidates = [stripped]
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _rgb_to_base64_jpeg(self, rgb: np.ndarray) -> str:
        image = self._prepare_rgb_image(rgb)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _save_rgb_image(self, rgb: np.ndarray, path: Path) -> None:
        image = self._prepare_rgb_image(rgb)
        Image.fromarray(image).save(str(path), format="JPEG", quality=90)

    def _prepare_rgb_image(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def _depth_summary(self, depth: np.ndarray) -> Dict[str, float]:
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        height, width = depth_arr.shape[:2]
        y1, y2 = height // 3, 2 * height // 3
        x1, x2 = width // 3, 2 * width // 3

        regions = {
            "center_mean": depth_arr[y1:y2, x1:x2],
            "left_mean": depth_arr[:, :x1],
            "right_mean": depth_arr[:, x2:],
            "top_mean": depth_arr[:y1, :],
            "bottom_mean": depth_arr[y2:, :],
        }
        return {
            name: float(np.nanmean(region)) if region.size else 0.0
            for name, region in regions.items()
        }

    def _depth_to_base64_png(self, depth: np.ndarray) -> str:
        colorized = self._prepare_depth_image(depth)
        buffer = io.BytesIO()
        Image.fromarray(colorized).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _save_depth_image(self, depth: np.ndarray, path: Path) -> None:
        colorized = self._prepare_depth_image(depth)
        Image.fromarray(colorized).save(str(path), format="PNG")

    def _prepare_depth_image(self, depth: np.ndarray) -> np.ndarray:
        depth_arr = self._normalize_depth(depth)
        return self._colorize_depth(depth_arr)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]

        finite = depth_arr[np.isfinite(depth_arr)]
        if finite.size == 0:
            return np.zeros(depth_arr.shape, dtype=np.float32)

        near = np.percentile(finite, float(self.args.cloud_depth_near_percentile))
        far = np.percentile(finite, float(self.args.cloud_depth_far_percentile))
        if far <= near:
            far = near + 1e-6

        clipped = np.clip(depth_arr, near, far)
        normalized = (clipped - near) / (far - near)
        normalized[~np.isfinite(normalized)] = 0.0
        return normalized.astype(np.float32)

    def _colorize_depth(self, normalized_depth: np.ndarray) -> np.ndarray:
        # Near pixels become red/yellow; far pixels become blue. This is easier
        # for general vision-language models to read than a raw grayscale map.
        near_score = 1.0 - np.clip(normalized_depth, 0.0, 1.0)
        red = np.full_like(near_score, 255.0)
        green = np.where(near_score > 0.5, 255.0, near_score * 2.0 * 255.0)
        blue = (1.0 - near_score) * 255.0
        colorized = np.stack([red * near_score, green, blue], axis=-1)
        return np.clip(colorized, 0, 255).astype(np.uint8)

    def _format_action_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for item in history[-8:]:
            formatted.append(
                {
                    "step": item.get("step"),
                    "action_id": item.get("action_id"),
                    "action_name": item.get("action_name"),
                }
            )
        return formatted

    def _hash_messages(self, messages: List[Dict[str, Any]]) -> str:
        payload = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if item.get("type") == "image_url":
                        normalized_content.append({"type": "image_url", "image_url": "<image>"})
                    else:
                        normalized_content.append(item)
                content = normalized_content
            payload.append({"role": message["role"], "content": content})
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _format_array(self, value: Any) -> List[float]:
        try:
            return [round(float(item), 4) for item in np.asarray(value).flatten().tolist()]
        except Exception:
            return []
