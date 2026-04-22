import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from airsim_plugin.airsim_settings import AirsimActions
from src.vlnce_src.cloud_model import CloudActionClient


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


def action_name(action_id):
    return ACTION_ID_TO_NAME.get(int(action_id), "UNKNOWN")


def action_branch(action_id):
    action_id = int(action_id)
    if action_id in {AirsimActions.TURN_LEFT, AirsimActions.MOVE_LEFT}:
        return "left"
    if action_id in {AirsimActions.TURN_RIGHT, AirsimActions.MOVE_RIGHT}:
        return "right"
    if action_id == AirsimActions.MOVE_FORWARD:
        return "forward"
    if action_id == AirsimActions.GO_UP:
        return "up"
    if action_id == AirsimActions.GO_DOWN:
        return "down"
    if action_id == AirsimActions.STOP:
        return "stop"
    return "unknown"


def is_opposite_action(model_action, reference_action):
    pairs = {
        AirsimActions.TURN_LEFT: AirsimActions.TURN_RIGHT,
        AirsimActions.TURN_RIGHT: AirsimActions.TURN_LEFT,
        AirsimActions.MOVE_LEFT: AirsimActions.MOVE_RIGHT,
        AirsimActions.MOVE_RIGHT: AirsimActions.MOVE_LEFT,
        AirsimActions.GO_UP: AirsimActions.GO_DOWN,
        AirsimActions.GO_DOWN: AirsimActions.GO_UP,
    }
    return pairs.get(int(model_action)) == int(reference_action)


def is_branch_mismatch(model_action, reference_action):
    model_branch = action_branch(model_action)
    reference_branch = action_branch(reference_action)
    branch_set = {"left", "right", "forward"}
    if model_branch not in branch_set or reference_branch not in branch_set:
        return False
    return model_branch != reference_branch


def pose_position(pose_like):
    if pose_like is None:
        return None
    try:
        arr = np.asarray(pose_like, dtype=np.float32).flatten()
    except Exception:
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def nearest_reference_index(position, reference_path):
    if position is None or not reference_path:
        return None, None
    pos = np.asarray(position, dtype=np.float32)[:3]
    ref = np.asarray([item[:3] for item in reference_path], dtype=np.float32)
    distances = np.linalg.norm(ref - pos, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def local_reference_turn(actions, nearest_index, window):
    if nearest_index is None or not actions:
        return False
    start = max(0, int(nearest_index) - int(window))
    end = min(len(actions), int(nearest_index) + int(window) + 1)
    local_actions = [int(item) for item in actions[start:end]]
    return any(
        action in {
            AirsimActions.TURN_LEFT,
            AirsimActions.TURN_RIGHT,
            AirsimActions.MOVE_LEFT,
            AirsimActions.MOVE_RIGHT,
        }
        for action in local_actions
    )


DEFAULT_INTERSECTION_SYSTEM_PROMPT = (
    "You are a visual auditor for a UAV vision-and-language navigation benchmark. "
    "Your job is not to navigate. Decide whether the current UAV RGB/depth observation "
    "shows a high-stakes intersection, fork, crossroad, or branching decision point where "
    "choosing the wrong left/right/forward branch would likely cause a large navigation error. "
    "Use the visual input, depth information, instruction context, and recent action context. "
    "Return only valid JSON."
)


DEFAULT_INTERSECTION_USER_PROMPT = """Navigation instruction:
{instruction}

Current step: {step}
Local model proposed action: {model_action_name} ({model_action_id})
Recent executed action history:
{action_history_json}
{pose_block}
{depth_summary_block}

Image 1 is the UAV RGB observation.
If present, Image 2 is a colorized depth map where red/yellow means closer obstacles and blue means farther space.

Question:
Does the current visual scene contain an intersection-like branching decision point that should be counted as the target failure mode for this experiment?

Use this definition:
- Count true when the image shows a road/path/building-corridor style split, crossroad, fork, T-junction, left/right branch, or visually plausible multiple route choices.
- Count true only if a wrong branch choice would likely be costly for reaching the described destination.
- Count false for ordinary turning, open-space drifting, obstacle avoidance without route branching, altitude adjustment, or cases where there is no visible branching ambiguity.
- Do not decide whether the model action is correct; only judge whether the visual scene is the target intersection-like difficult decision point.

Return exactly one JSON object:
{{
  "is_intersection_challenge": true,
  "confidence": 0.0,
  "visual_cues": ["short cue"],
  "reason": "short reason"
}}"""


class CloudIntersectionJudge:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.client = CloudActionClient(args, logger)
        self.system_prompt = self._load_prompt(
            getattr(args, "intersection_cloud_prompt_system_path", None),
            DEFAULT_INTERSECTION_SYSTEM_PROMPT,
        )
        self.user_prompt = self._load_prompt(
            getattr(args, "intersection_cloud_prompt_user_path", None),
            DEFAULT_INTERSECTION_USER_PROMPT,
        )
        self.confidence_threshold = float(getattr(args, "intersection_cloud_confidence_threshold", 0.5))

    def judge(self, observation, episode, model_action, step, history, save_dir=None):
        messages = self._build_messages(observation, episode, model_action, step, history)
        prompt_hash = self.client._hash_messages(messages)
        saved_inputs = self._save_inputs(messages, observation, save_dir) if save_dir is not None else {}
        started_at = time.time()
        raw_response = ""
        error = None

        for attempt in range(int(self.args.cloud_max_retries)):
            try:
                completion = self.client.client.chat.completions.create(
                    model=self.args.cloud_model,
                    messages=messages,
                    temperature=float(self.args.cloud_temperature),
                    max_tokens=int(self.args.cloud_max_tokens),
                    timeout=float(self.args.cloud_timeout),
                    stream=False,
                )
                raw_response = completion.choices[0].message.content or ""
                payload = self._extract_json(raw_response)
                if payload is not None:
                    self._save_response(saved_inputs, raw_response, payload, None)
                    confidence = float(payload.get("confidence", 0.0) or 0.0)
                    is_positive = bool(payload.get("is_intersection_challenge")) and confidence >= self.confidence_threshold
                    return is_positive, {
                        "raw_response": raw_response,
                        "parsed": payload,
                        "confidence": confidence,
                        "latency_sec": time.time() - started_at,
                        "attempts": attempt + 1,
                        "error": None,
                        "prompt_hash": prompt_hash,
                        "saved_inputs": saved_inputs,
                    }
                error = "invalid intersection judge response"
                messages = messages + [
                    {
                        "role": "user",
                        "content": "Your previous response was invalid. Return only the required JSON object.",
                    }
                ]
            except Exception as exc:
                error = repr(exc)
                self.logger.error("cloud intersection judge failed on attempt {}: {}".format(attempt + 1, error))

        self._save_response(saved_inputs, raw_response, None, error)
        return False, {
            "raw_response": raw_response,
            "parsed": None,
            "confidence": 0.0,
            "latency_sec": time.time() - started_at,
            "attempts": int(self.args.cloud_max_retries),
            "error": error,
            "prompt_hash": prompt_hash,
            "saved_inputs": saved_inputs,
        }

    def metadata(self):
        return {
            "detector": "cloud",
            "model": self.args.cloud_model,
            "base_url": self.args.cloud_base_url,
            "confidence_threshold": self.confidence_threshold,
            "system_prompt_path": self._prompt_path(getattr(self.args, "intersection_cloud_prompt_system_path", None)),
            "user_prompt_path": self._prompt_path(getattr(self.args, "intersection_cloud_prompt_user_path", None)),
        }

    def _build_messages(self, observation, episode, model_action, step, history):
        context = self._prompt_context(observation, episode, model_action, step, history)
        content = [{"type": "text", "text": self.user_prompt.format_map(_SafeFormatDict(context)).strip()}]
        if not self.args.cloud_no_rgb and observation.get("rgb") is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{}".format(
                        self.client._rgb_to_base64_jpeg(observation["rgb"])
                    )
                },
            })
        if self.client._send_depth_image(observation):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,{}".format(
                        self.client._depth_to_base64_png(observation["depth"])
                    )
                },
            })

        return [
            {"role": "system", "content": self.system_prompt.format_map(_SafeFormatDict(context)).strip()},
            {"role": "user", "content": content},
        ]

    def _prompt_context(self, observation, episode, model_action, step, history):
        pose_block = ""
        depth_summary_block = ""
        if self.args.cloud_use_pose:
            pose_block = "\nCurrent pose [x, y, z, qw, qx, qy, qz]: {}".format(
                self.client._format_array(observation.get("pose", []))
            )
        if self.client._send_depth_summary(observation):
            depth_summary_block = "\nDepth summary:\n{}".format(
                json.dumps(self.client._depth_summary(observation["depth"]), ensure_ascii=True)
            )
        return {
            "instruction": episode["instruction"]["instruction_text"],
            "step": str(step),
            "model_action_id": str(int(model_action)),
            "model_action_name": action_name(model_action),
            "action_history_json": json.dumps(self._format_history(history), ensure_ascii=True),
            "pose_block": pose_block,
            "depth_summary_block": depth_summary_block,
        }

    def _format_history(self, history):
        return [
            {
                "step": item.get("step"),
                "model_action": item.get("model_action_name") or item.get("action_name"),
                "executed_action": item.get("executed_action_name") or item.get("action_name"),
                "correction_applied": item.get("correction_applied", False),
            }
            for item in history[-8:]
        ]

    def _load_prompt(self, path_value, default):
        path = self._prompt_path(path_value)
        if path is None:
            return default
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _prompt_path(self, path_value):
        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = Path(str(Path.cwd())) / path
        return str(path)

    def _extract_json(self, text):
        return self.client._extract_json(text)

    def _save_inputs(self, messages, observation, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        prompt_path = save_dir / "prompt.txt"
        user_content = messages[1]["content"]
        if isinstance(user_content, list):
            prompt_text = "\n\n".join(
                item.get("text", "")
                for item in user_content
                if item.get("type") == "text"
            )
        else:
            prompt_text = str(user_content)
        prompt_path.write_text(prompt_text, encoding="utf-8")
        saved["prompt"] = str(prompt_path)

        if not self.args.cloud_no_rgb and observation.get("rgb") is not None:
            rgb_path = save_dir / "rgb.jpg"
            self.client._save_rgb_image(observation["rgb"], rgb_path)
            saved["rgb"] = str(rgb_path)

        if self.client._send_depth_image(observation):
            depth_path = save_dir / "depth.png"
            self.client._save_depth_image(observation["depth"], depth_path)
            saved["depth"] = str(depth_path)

        request_path = save_dir / "request.json"
        with open(str(request_path), "w", encoding="utf-8") as f:
            json.dump(
                self.client._messages_for_snapshot(messages, saved),
                f,
                indent=2,
                ensure_ascii=True,
            )
        saved["request"] = str(request_path)
        return saved

    def _save_response(self, saved_inputs, raw_response, parsed, error):
        if not saved_inputs or "request" not in saved_inputs:
            return
        response_path = Path(saved_inputs["request"]).parent / "response.json"
        with open(str(response_path), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_response": raw_response,
                    "parsed": parsed,
                    "error": error,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        saved_inputs["response"] = str(response_path)


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class IntersectionInterventionMonitor:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.mode = str(getattr(args, "intersection_eval_mode", "off")).lower()
        self.detector = str(getattr(args, "intersection_detector", "cloud")).lower()
        self.policy = str(getattr(args, "intersection_wrong_policy", "branch_mismatch")).lower()
        self.turn_window = int(getattr(args, "intersection_turn_window", 4))
        self.max_events_per_episode = int(getattr(args, "intersection_max_events_per_episode", -1))
        self.save_inputs = bool(getattr(args, "intersection_save_inputs", False))
        self.input_root = Path(args.project_prefix) / "DATA/output/{}/eval/intersection_inputs/{}".format(
            args.name,
            args.make_dir_time,
        )
        self.cloud_judge = None
        if self.enabled() and self.detector == "cloud":
            if logger is None:
                raise ValueError("Cloud intersection detector requires a logger")
            self.cloud_judge = CloudIntersectionJudge(args, logger)
        self.events = []
        self.cloud_checks = []
        self.events_by_episode = defaultdict(int)
        self.stats = {
            "mode": self.mode,
            "detector": self.detector,
            "wrong_policy": self.policy,
            "turn_window": self.turn_window,
            "candidate_events": 0,
            "intersection_errors": 0,
            "corrections_applied": 0,
            "cloud_checked_candidates": 0,
            "cloud_positive_events": 0,
            "cloud_error_checks": 0,
            "cloud_latency_sec_total": 0.0,
            "cloud_latency_sec_max": 0.0,
            "input_snapshots_saved": 0,
        }

    def enabled(self):
        return self.mode in {"detect", "correct"}

    def _is_wrong_decision(self, model_action, reference_action):
        if self.policy == "opposite":
            return is_opposite_action(model_action, reference_action)
        if self.policy == "branch_mismatch":
            return is_branch_mismatch(model_action, reference_action)
        if self.policy == "action_mismatch":
            return int(model_action) != int(reference_action)
        raise ValueError("Unsupported intersection_wrong_policy: {}".format(self.policy))

    def evaluate_step(self, episode, current_pose, model_action, step, observation=None, history=None):
        if not self.enabled():
            return int(model_action), None

        episode_id = str(episode["episode_id"])
        if self.max_events_per_episode >= 0 and self.events_by_episode[episode_id] >= self.max_events_per_episode:
            return int(model_action), None

        reference_path = episode.get("reference_path") or []
        reference_actions = episode.get("actions") or []
        position = pose_position(current_pose)
        nearest_index, deviation = nearest_reference_index(position, reference_path)
        if nearest_index is None or not reference_actions:
            return int(model_action), None

        reference_index = min(int(nearest_index), len(reference_actions) - 1)
        reference_action = int(reference_actions[reference_index])
        near_turn = local_reference_turn(reference_actions, reference_index, self.turn_window)
        wrong_decision = self._is_wrong_decision(model_action, reference_action)

        if not (near_turn and wrong_decision):
            return int(model_action), None

        self.stats["candidate_events"] += 1

        cloud_meta = None
        if self.detector == "cloud":
            if observation is None:
                raise ValueError("Cloud intersection detector requires observation")
            is_intersection, cloud_meta = self.cloud_judge.judge(
                observation=observation,
                episode=episode,
                model_action=model_action,
                step=step,
                history=history or [],
                save_dir=self._input_save_dir(episode_id, step) if self.save_inputs else None,
            )
            if cloud_meta.get("saved_inputs"):
                self.stats["input_snapshots_saved"] += 1
            self.stats["cloud_checked_candidates"] += 1
            latency = cloud_meta.get("latency_sec")
            if latency is not None:
                self.stats["cloud_latency_sec_total"] += float(latency)
                self.stats["cloud_latency_sec_max"] = max(
                    self.stats["cloud_latency_sec_max"],
                    float(latency),
                )
            if cloud_meta.get("error"):
                self.stats["cloud_error_checks"] += 1

            check = {
                "episode_id": episode_id,
                "trajectory_id": episode.get("trajectory_id"),
                "step": int(step),
                "model_action_id": int(model_action),
                "model_action_name": action_name(model_action),
                "reference_action_id": int(reference_action),
                "reference_action_name": action_name(reference_action),
                "is_intersection_challenge": bool(is_intersection),
                "cloud": cloud_meta,
            }
            self.cloud_checks.append(check)
            if not is_intersection:
                return int(model_action), None
            self.stats["cloud_positive_events"] += 1
        elif self.detector != "reference_proxy":
            raise ValueError("Unsupported intersection_detector: {}".format(self.detector))

        self.stats["intersection_errors"] += 1
        self.events_by_episode[episode_id] += 1

        executed_action = int(model_action)
        correction_applied = False
        if self.mode == "correct":
            executed_action = reference_action
            correction_applied = True
            self.stats["corrections_applied"] += 1

        event = {
            "episode_id": episode_id,
            "trajectory_id": episode.get("trajectory_id"),
            "step": int(step),
            "nearest_reference_index": int(reference_index),
            "deviation_m": deviation,
            "near_reference_turn": bool(near_turn),
            "wrong_policy": self.policy,
            "model_action_id": int(model_action),
            "model_action_name": action_name(model_action),
            "reference_action_id": int(reference_action),
            "reference_action_name": action_name(reference_action),
            "executed_action_id": int(executed_action),
            "executed_action_name": action_name(executed_action),
            "correction_applied": correction_applied,
            "position": position.tolist() if position is not None else None,
        }
        if cloud_meta is not None:
            event["cloud"] = cloud_meta
        self.events.append(event)
        return executed_action, event

    def _input_save_dir(self, episode_id, step):
        safe_episode_id = str(episode_id).replace("/", "_")
        return self.input_root / safe_episode_id / "step_{:04d}".format(int(step))

    def summary(self, evaluated_episodes):
        summary = dict(self.stats)
        summary["event_count"] = len(self.events)
        summary["episodes_with_events"] = len(self.events_by_episode)
        summary["evaluated_episodes"] = int(evaluated_episodes)
        if evaluated_episodes:
            summary["events_per_episode"] = float(len(self.events)) / float(evaluated_episodes)
            summary["episode_event_rate"] = float(len(self.events_by_episode)) / float(evaluated_episodes)
        else:
            summary["events_per_episode"] = 0.0
            summary["episode_event_rate"] = 0.0
        checked = int(summary.get("cloud_checked_candidates") or 0)
        if checked:
            summary["cloud_positive_rate"] = float(summary.get("cloud_positive_events") or 0) / float(checked)
            summary["cloud_latency_sec_avg"] = float(summary.get("cloud_latency_sec_total") or 0.0) / float(checked)
        else:
            summary["cloud_positive_rate"] = 0.0
            summary["cloud_latency_sec_avg"] = 0.0
        if self.cloud_judge is not None:
            summary["cloud_detector_metadata"] = self.cloud_judge.metadata()
        return summary
