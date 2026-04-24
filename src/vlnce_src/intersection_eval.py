import json
import shutil
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

TURN_ACTIONS = {
    AirsimActions.TURN_LEFT,
    AirsimActions.TURN_RIGHT,
    AirsimActions.MOVE_LEFT,
    AirsimActions.MOVE_RIGHT,
}

LANDMARK_CUE_WORDS = {
    "building",
    "bridge",
    "tower",
    "statue",
    "plaza",
    "square",
    "church",
    "corner",
    "gate",
    "entrance",
    "exit",
    "road",
    "street",
    "alley",
    "river",
    "lake",
    "park",
    "tree",
    "fountain",
    "stairs",
    "staircase",
    "wall",
    "door",
    "doorway",
    "roof",
    "house",
    "courtyard",
    "landmark",
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


def instruction_landmark_cues(text):
    text = str(text or "").lower()
    return sorted(word for word in LANDMARK_CUE_WORDS if word in text)


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
    return closest_local_reference_turn(actions, nearest_index, window)[0]


def closest_local_reference_turn(actions, nearest_index, window):
    if nearest_index is None or not actions:
        return False, None, None
    start = max(0, int(nearest_index) - int(window))
    end = min(len(actions), int(nearest_index) + int(window) + 1)
    candidates = []
    for index in range(start, end):
        if int(actions[index]) in TURN_ACTIONS:
            candidates.append((abs(index - int(nearest_index)), index))
    if not candidates:
        return False, None, None
    distance, index = min(candidates)
    return True, int(index), int(distance)


def depth_branch_openness(depth):
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[..., 0]
    if depth_arr.ndim != 2 or depth_arr.size == 0:
        return None

    height, width = depth_arr.shape[:2]
    y1 = int(height * 0.18)
    y2 = int(height * 0.58)
    x0 = int(width * 0.08)
    x1 = int(width * 0.38)
    x2 = int(width * 0.62)
    x3 = int(width * 0.92)

    band = depth_arr[y1:y2, :]
    if band.size == 0:
        return None

    left = band[:, x0:x1]
    front = band[:, x1:x2]
    right = band[:, x2:x3]

    def region_mean(region):
        if region.size == 0:
            return 0.0
        return float(np.nanmean(region))

    def region_min(region):
        if region.size == 0:
            return 0.0
        return float(np.nanmin(region))

    return {
        "left_mean": region_mean(left),
        "front_mean": region_mean(front),
        "right_mean": region_mean(right),
        "left_min": region_min(left),
        "front_min": region_min(front),
        "right_min": region_min(right),
    }


DEFAULT_INTERSECTION_SYSTEM_PROMPT = (
    "You are a visual auditor for a UAV vision-and-language navigation benchmark. "
    "Your job is not to navigate. Decide whether the current UAV RGB/depth observation "
    "shows a high-risk decision difficulty where the local model could easily make a costly mistake. "
    "The target failure modes include: explicit junction-like branching, partially occluded target landmarks, ambiguous landmark matching, and limited-visibility turns where the correct branch or landmark cue is hard to confirm from the current viewpoint. "
    "Prefer true only when the scene is genuinely difficult for route choice or landmark alignment under the instruction. "
    "Prefer false for ordinary turning, open drifting, obstacle bypassing, altitude-only adjustment, gentle curvature, or scenes with no clear branch or semantic ambiguity. "
    "Use the visual input as primary evidence, and use depth information, instruction context, and recent action context as supporting evidence. "
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
{branch_visibility_block}

Image 1 is the UAV RGB observation.
If present, Image 2 is a colorized depth map where red/yellow means closer obstacles and blue means farther space.

Local candidate tags suggested by the local trigger:
{candidate_tags_block}

Question:
Does the current visual scene contain a decision difficulty that should be counted as the target failure mode for this experiment?

Use this definition:
- Count true when the image shows a costly branch-choice or landmark-alignment difficulty.
- Valid true cases include:
  1. junction_like: road/path/building-corridor split, crossroad, fork, T-junction, bridge entry/exit choice, street-corner branch.
  2. occluded_landmark: the instruction depends on a landmark or waypoint cue, but it is only partially visible, distant, or blocked.
  3. ambiguous_landmark: multiple similar landmarks or route options are visible and the current frame does not clearly disambiguate the instruction target.
  4. limited_visibility_turn: the correct turn depends on structure hidden around a corner, behind a facade, or outside the current field of view.
  5. memory_required: the current frame alone is insufficient and the decision depends on short-term visual history.
- Count false for ordinary turning, open-space drifting, obstacle avoidance without route ambiguity, altitude adjustment, rooftop openness, curved sidewalks, or scenes with no clear branch or semantic ambiguity.
- Do not decide whether the model action is correct; only judge whether this scene is a meaningful decision difficulty.

Return exactly one JSON object:
{{
  "is_decision_difficulty": true,
  "difficulty_type": "junction_like",
  "confidence": 0.0,
  "visual_cues": ["short cue"],
  "reason": "short reason"
}}"""


class CloudIntersectionJudge:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.client = CloudActionClient(args, logger)
        self.history_size = max(1, int(getattr(args, "intersection_judge_history_size", 4)))
        self.system_prompt = self._load_prompt(
            getattr(args, "intersection_cloud_prompt_system_path", None),
            DEFAULT_INTERSECTION_SYSTEM_PROMPT,
        )
        self.user_prompt = self._load_prompt(
            getattr(args, "intersection_cloud_prompt_user_path", None),
            DEFAULT_INTERSECTION_USER_PROMPT,
        )
        self.confidence_threshold = float(getattr(args, "intersection_cloud_confidence_threshold", 0.5))
        self.response_cache = {}

    def restore_cache(self, cloud_checks):
        self.response_cache = {}
        for check in cloud_checks or []:
            cloud_meta = check.get("cloud") or {}
            prompt_hash = cloud_meta.get("prompt_hash")
            parsed = cloud_meta.get("parsed")
            if prompt_hash and parsed is not None:
                self.response_cache[str(prompt_hash)] = {
                    "raw_response": cloud_meta.get("raw_response", ""),
                    "parsed": parsed,
                    "confidence": float(cloud_meta.get("confidence", 0.0) or 0.0),
                    "attempts": int(cloud_meta.get("attempts", 1) or 1),
                }

    def judge(self, observation, episode, model_action, step, history, candidate_tags=None, save_dir=None):
        messages = self._build_messages(observation, episode, model_action, step, history, candidate_tags)
        prompt_hash = self.client._hash_messages(messages)
        saved_inputs = self._save_inputs(messages, observation, save_dir) if save_dir is not None else {}
        started_at = time.time()
        raw_response = ""
        error = None

        cached = self.response_cache.get(prompt_hash)
        if cached is not None:
            payload = self._normalize_payload(cached["parsed"])
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            is_positive = bool(payload.get("is_decision_difficulty")) and confidence >= self.confidence_threshold
            self._save_response(saved_inputs, cached.get("raw_response", ""), payload, None)
            return is_positive, {
                "raw_response": cached.get("raw_response", ""),
                "parsed": payload,
                "confidence": confidence,
                "latency_sec": 0.0,
                "attempts": int(cached.get("attempts", 1) or 1),
                "error": None,
                "prompt_hash": prompt_hash,
                "saved_inputs": saved_inputs,
                "cache_hit": True,
            }

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
                    payload = self._normalize_payload(payload)
                    self._save_response(saved_inputs, raw_response, payload, None)
                    confidence = float(payload.get("confidence", 0.0) or 0.0)
                    is_positive = bool(payload.get("is_decision_difficulty")) and confidence >= self.confidence_threshold
                    self.response_cache[prompt_hash] = {
                        "raw_response": raw_response,
                        "parsed": payload,
                        "confidence": confidence,
                        "attempts": attempt + 1,
                    }
                    return is_positive, {
                        "raw_response": raw_response,
                        "parsed": payload,
                        "confidence": confidence,
                        "latency_sec": time.time() - started_at,
                        "attempts": attempt + 1,
                        "error": None,
                        "prompt_hash": prompt_hash,
                        "saved_inputs": saved_inputs,
                        "cache_hit": False,
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
            "cache_hit": False,
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

    def _build_messages(self, observation, episode, model_action, step, history, candidate_tags):
        context = self._prompt_context(observation, episode, model_action, step, history, candidate_tags)
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

    def _prompt_context(self, observation, episode, model_action, step, history, candidate_tags):
        pose_block = ""
        depth_summary_block = ""
        branch_visibility_block = ""
        if self.args.cloud_use_pose:
            pose_block = "\nCurrent pose [x, y, z, qw, qx, qy, qz]: {}".format(
                self.client._format_array(observation.get("pose", []))
            )
        if self.client._send_depth_summary(observation):
            depth_summary_block = "\nDepth summary:\n{}".format(
                json.dumps(self.client._depth_summary(observation["depth"]), ensure_ascii=True)
            )
        visibility = depth_branch_openness(observation.get("depth"))
        if visibility is not None:
            branch_visibility_block = "\nForward branch visibility summary from depth:\n{}".format(
                json.dumps(visibility, ensure_ascii=True)
            )
        return {
            "instruction": episode["instruction"]["instruction_text"],
            "step": str(step),
            "model_action_id": str(int(model_action)),
            "model_action_name": action_name(model_action),
            "action_history_json": json.dumps(self._format_history(history), ensure_ascii=True),
            "pose_block": pose_block,
            "depth_summary_block": depth_summary_block,
            "branch_visibility_block": branch_visibility_block,
            "candidate_tags_block": json.dumps(candidate_tags or [], ensure_ascii=True),
        }

    def _format_history(self, history):
        return [
            {
                "step": item.get("step"),
                "model_action": item.get("model_action_name") or item.get("action_name"),
                "executed_action": item.get("executed_action_name") or item.get("action_name"),
                "correction_applied": item.get("correction_applied", False),
            }
            for item in history[-self.history_size:]
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

    def _normalize_payload(self, payload):
        payload = dict(payload or {})
        if "is_decision_difficulty" not in payload:
            payload["is_decision_difficulty"] = bool(payload.get("is_intersection_challenge"))
        if "difficulty_type" not in payload or not payload.get("difficulty_type"):
            payload["difficulty_type"] = "junction_like" if payload.get("is_decision_difficulty") else "none"
        payload["difficulty_type"] = str(payload.get("difficulty_type") or "none")
        payload["is_intersection_challenge"] = bool(payload.get("is_decision_difficulty"))
        return payload

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
        self.candidate_mode = str(getattr(args, "intersection_candidate_mode", "cheap")).lower()
        self.turn_window = int(getattr(args, "intersection_turn_window", 4))
        self.max_deviation_m = float(getattr(args, "intersection_max_deviation_m", 20.0))
        self.max_events_per_episode = int(getattr(args, "intersection_max_events_per_episode", -1))
        self.max_cloud_checks_per_episode = int(getattr(args, "intersection_max_cloud_checks_per_episode", 6))
        self.depth_gate_mode = str(getattr(args, "intersection_depth_gate_mode", "cheap_strict")).lower()
        self.min_open_branches = int(getattr(args, "intersection_min_open_branches", 2))
        self.depth_open_threshold = float(getattr(args, "intersection_depth_open_threshold", 0.18))
        self.cooldown_steps = int(getattr(args, "intersection_correction_cooldown_steps", 0))
        self.max_corrections_per_episode = int(getattr(args, "intersection_max_corrections_per_episode", -1))
        self.max_corrections_per_cluster = int(getattr(args, "intersection_max_corrections_per_cluster", -1))
        self.save_inputs = bool(getattr(args, "intersection_save_inputs", False))
        self.save_positive_inputs = bool(getattr(args, "intersection_save_positive_inputs", False))
        self.save_positive_videos = bool(getattr(args, "intersection_save_positive_videos", False))
        self.input_root = Path(args.project_prefix) / "DATA/output/{}/eval/decision_candidates/{}".format(
            args.name,
            args.make_dir_time,
        )
        self.positive_input_root = Path(args.project_prefix) / "DATA/output/{}/eval/decision_positive_inputs/{}".format(
            args.name,
            args.make_dir_time,
        )
        self.positive_video_root = Path(args.project_prefix) / "DATA/output/{}/eval/decision_positive_videos/{}".format(
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
        self.corrections_by_episode = defaultdict(int)
        self.last_correction_step_by_episode = {}
        self.corrections_by_episode_cluster = defaultdict(int)
        self.cloud_checks_by_episode = defaultdict(int)
        self.checked_candidate_clusters = set()
        self.positive_episode_ids = set()
        self.positive_input_dirs = defaultdict(list)
        self.positive_video_paths = {}
        self.stats = {
            "mode": self.mode,
            "detector": self.detector,
            "wrong_policy": self.policy,
            "candidate_mode": self.candidate_mode,
            "turn_window": self.turn_window,
            "max_deviation_m": self.max_deviation_m,
            "max_cloud_checks_per_episode": self.max_cloud_checks_per_episode,
            "depth_gate_mode": self.depth_gate_mode,
            "min_open_branches": self.min_open_branches,
            "depth_open_threshold": self.depth_open_threshold,
            "correction_cooldown_steps": self.cooldown_steps,
            "max_corrections_per_episode": self.max_corrections_per_episode,
            "max_corrections_per_cluster": self.max_corrections_per_cluster,
            "candidate_events": 0,
            "wrong_decision_candidates": 0,
            "intersection_errors": 0,
            "corrections_applied": 0,
            "corrections_suppressed": 0,
            "corrections_suppressed_by_cooldown": 0,
            "corrections_suppressed_by_episode_limit": 0,
            "corrections_suppressed_by_cluster_limit": 0,
            "cloud_checked_candidates": 0,
            "candidates_suppressed_by_cluster": 0,
            "candidates_suppressed_by_deviation": 0,
            "candidates_suppressed_by_cloud_limit": 0,
            "candidates_suppressed_by_visibility": 0,
            "cloud_cache_hits": 0,
            "cloud_positive_events": 0,
            "cloud_error_checks": 0,
            "cloud_latency_sec_total": 0.0,
            "cloud_latency_sec_max": 0.0,
            "input_snapshots_saved": 0,
            "positive_input_snapshots_saved": 0,
            "positive_videos_saved": 0,
        }

    def restore(self, events=None, cloud_checks=None):
        self.events = list(events or [])
        self.cloud_checks = list(cloud_checks or [])
        self.events_by_episode = defaultdict(int)
        self.corrections_by_episode = defaultdict(int)
        self.last_correction_step_by_episode = {}
        self.corrections_by_episode_cluster = defaultdict(int)
        self.cloud_checks_by_episode = defaultdict(int)
        self.checked_candidate_clusters = set()
        self.positive_episode_ids = set()
        self.positive_input_dirs = defaultdict(list)
        self.positive_video_paths = {}

        self.stats.update({
            "candidate_events": len(self.cloud_checks) if self.cloud_checks else len(self.events),
            "wrong_decision_candidates": 0,
            "intersection_errors": len(self.events),
            "corrections_applied": 0,
            "corrections_suppressed": 0,
            "corrections_suppressed_by_cooldown": 0,
            "corrections_suppressed_by_episode_limit": 0,
            "corrections_suppressed_by_cluster_limit": 0,
            "cloud_checked_candidates": len(self.cloud_checks),
            "candidates_suppressed_by_cluster": 0,
            "candidates_suppressed_by_deviation": 0,
            "candidates_suppressed_by_cloud_limit": 0,
            "candidates_suppressed_by_visibility": 0,
            "cloud_cache_hits": 0,
            "cloud_positive_events": 0,
            "cloud_error_checks": 0,
            "cloud_latency_sec_total": 0.0,
            "cloud_latency_sec_max": 0.0,
            "input_snapshots_saved": 0,
            "positive_input_snapshots_saved": 0,
            "positive_videos_saved": 0,
        })

        if self.cloud_judge is not None:
            self.cloud_judge.restore_cache(self.cloud_checks)

        for check in self.cloud_checks:
            episode_id = str(check.get("episode_id"))
            self.cloud_checks_by_episode[episode_id] += 1
            if check.get("wrong_decision"):
                self.stats["wrong_decision_candidates"] += 1
            if check.get("is_intersection_challenge"):
                self.stats["cloud_positive_events"] += 1
            cloud_meta = check.get("cloud") or {}
            latency = cloud_meta.get("latency_sec")
            if latency is not None:
                self.stats["cloud_latency_sec_total"] += float(latency)
                self.stats["cloud_latency_sec_max"] = max(
                    self.stats["cloud_latency_sec_max"],
                    float(latency),
                )
            if cloud_meta.get("error"):
                self.stats["cloud_error_checks"] += 1
            if cloud_meta.get("saved_inputs"):
                self.stats["input_snapshots_saved"] += 1
            positive_inputs = cloud_meta.get("positive_inputs")
            if positive_inputs:
                self.stats["positive_input_snapshots_saved"] += 1
                self.positive_input_dirs[episode_id].append(positive_inputs)
            if cloud_meta.get("cache_hit"):
                self.stats["cloud_cache_hits"] += 1
            cluster_id = check.get("candidate_cluster_id")
            if cluster_id is None:
                cluster_id = check.get("cluster_id")
            if cluster_id is not None:
                self.checked_candidate_clusters.add((str(check.get("episode_id")), int(cluster_id)))
            if check.get("is_decision_difficulty") or check.get("is_intersection_challenge"):
                self.positive_episode_ids.add(episode_id)

        if not self.cloud_checks:
            self.stats["wrong_decision_candidates"] = sum(
                1 for event in self.events if event.get("wrong_decision", True)
            )

        for event in self.events:
            episode_id = str(event.get("episode_id"))
            step = int(event.get("step", 0))
            cluster_id = event.get("cluster_id")
            if cluster_id is None:
                cluster_id = self._cluster_id(int(event.get("nearest_reference_index", 0)))
            self.events_by_episode[episode_id] += 1
            if event.get("correction_applied"):
                self.stats["corrections_applied"] += 1
                self.corrections_by_episode[episode_id] += 1
                self.last_correction_step_by_episode[episode_id] = max(
                    self.last_correction_step_by_episode.get(episode_id, -1),
                    step,
                )
                self.corrections_by_episode_cluster[(episode_id, int(cluster_id))] += 1
            reason = event.get("correction_suppressed_reason")
            if reason:
                self.stats["corrections_suppressed"] += 1
                key = "corrections_suppressed_by_{}".format(reason)
                if key in self.stats:
                    self.stats[key] += 1
            if event.get("is_decision_difficulty") or event.get("is_intersection_challenge"):
                self.positive_episode_ids.add(episode_id)

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

    def _should_check_cloud(self, near_turn, wrong_decision):
        if self.candidate_mode == "cheap":
            return bool(near_turn and wrong_decision)
        if self.candidate_mode == "strict":
            return bool(near_turn and wrong_decision)
        if self.candidate_mode == "balanced":
            return bool(near_turn)
        if self.candidate_mode == "expensive":
            return True
        raise ValueError("Unsupported intersection_candidate_mode: {}".format(self.candidate_mode))

    def _should_dedupe_candidate_cluster(self):
        return self.candidate_mode in {"cheap", "strict", "balanced"}

    def _strict_candidate_ok(self, reference_actions, reference_index, turn_anchor_distance):
        if turn_anchor_distance is None or int(turn_anchor_distance) > 1:
            return False
        reference_action = int(reference_actions[reference_index])
        return reference_action in {
            AirsimActions.MOVE_FORWARD,
            AirsimActions.TURN_LEFT,
            AirsimActions.TURN_RIGHT,
            AirsimActions.MOVE_LEFT,
            AirsimActions.MOVE_RIGHT,
        }

    def _should_apply_visibility_gate(self):
        if self.depth_gate_mode == "off":
            return False
        if self.depth_gate_mode == "strict":
            return self.candidate_mode == "strict"
        if self.depth_gate_mode == "cheap_strict":
            return self.candidate_mode in {"cheap", "strict"}
        if self.depth_gate_mode == "all":
            return True
        raise ValueError("Unsupported intersection_depth_gate_mode: {}".format(self.depth_gate_mode))

    def _visibility_gate(self, observation, reference_action):
        if observation is None or observation.get("depth") is None:
            return True, None

        branch_visibility = depth_branch_openness(observation.get("depth"))
        if branch_visibility is None:
            return True, None

        threshold = float(self.depth_open_threshold)
        open_flags = {
            "left": branch_visibility["left_mean"] >= threshold,
            "front": branch_visibility["front_mean"] >= threshold,
            "right": branch_visibility["right_mean"] >= threshold,
        }
        open_count = sum(1 for flag in open_flags.values() if flag)
        target_branch = action_branch(reference_action)
        target_branch_open = open_flags.get(target_branch, True)
        pass_gate = open_count >= max(1, int(self.min_open_branches)) and bool(target_branch_open)

        details = {
            "mode": self.depth_gate_mode,
            "threshold": threshold,
            "min_open_branches": int(self.min_open_branches),
            "open_flags": open_flags,
            "open_count": int(open_count),
            "target_branch": target_branch,
            "target_branch_open": bool(target_branch_open),
            "depth_branch_visibility": branch_visibility,
        }
        return pass_gate, details

    def _decision_difficulty_assessment(self, observation, episode, reference_action, near_turn):
        landmark_cues = instruction_landmark_cues(episode["instruction"]["instruction_text"])
        landmark_instruction = bool(landmark_cues)
        if observation is None or observation.get("depth") is None:
            return {
                "pass_gate": bool(near_turn and landmark_instruction),
                "candidate_tags": ["memory_required"] if near_turn and landmark_instruction else [],
                "landmark_instruction": landmark_instruction,
                "landmark_cues": landmark_cues,
                "depth_available": False,
            }

        branch_visibility = depth_branch_openness(observation.get("depth"))
        if branch_visibility is None:
            return {
                "pass_gate": bool(near_turn),
                "candidate_tags": ["memory_required"] if near_turn else [],
                "landmark_instruction": landmark_instruction,
                "landmark_cues": landmark_cues,
                "depth_available": False,
            }

        threshold = float(self.depth_open_threshold)
        open_flags = {
            "left": branch_visibility["left_mean"] >= threshold,
            "front": branch_visibility["front_mean"] >= threshold,
            "right": branch_visibility["right_mean"] >= threshold,
        }
        open_count = sum(1 for flag in open_flags.values() if flag)
        target_branch = action_branch(reference_action)
        target_branch_open = open_flags.get(target_branch, True)
        forward_blocked = (
            branch_visibility["front_mean"] < threshold * 0.85
            or branch_visibility["front_min"] < threshold * 0.45
        )
        side_open = bool(open_flags["left"] or open_flags["right"])

        candidate_tags = []
        if open_count >= max(1, int(self.min_open_branches)) and target_branch_open:
            candidate_tags.append("junction_like")
        if landmark_instruction and forward_blocked:
            candidate_tags.append("occluded_landmark")
        if landmark_instruction and open_count >= 2:
            candidate_tags.append("ambiguous_landmark")
        if near_turn and (forward_blocked or (target_branch in {"left", "right"} and not target_branch_open)):
            candidate_tags.append("limited_visibility_turn")
        if near_turn and landmark_instruction and not side_open and forward_blocked:
            candidate_tags.append("memory_required")

        return {
            "pass_gate": bool(candidate_tags),
            "candidate_tags": sorted(set(candidate_tags)),
            "landmark_instruction": landmark_instruction,
            "landmark_cues": landmark_cues,
            "open_flags": open_flags,
            "open_count": int(open_count),
            "target_branch": target_branch,
            "target_branch_open": bool(target_branch_open),
            "forward_blocked": bool(forward_blocked),
            "depth_branch_visibility": branch_visibility,
            "threshold": threshold,
            "min_open_branches": int(self.min_open_branches),
            "depth_available": True,
        }

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
        if deviation is not None and deviation > self.max_deviation_m:
            self.stats["candidates_suppressed_by_deviation"] += 1
            return int(model_action), None
        near_turn, turn_anchor_index, turn_anchor_distance = closest_local_reference_turn(
            reference_actions,
            reference_index,
            self.turn_window,
        )
        wrong_decision = self._is_wrong_decision(model_action, reference_action)
        difficulty_assessment = self._decision_difficulty_assessment(
            observation,
            episode,
            reference_action,
            near_turn,
        )
        local_candidate_tags = difficulty_assessment.get("candidate_tags", [])
        local_difficulty_candidate = bool(local_candidate_tags)

        if not (wrong_decision and (near_turn or local_difficulty_candidate)):
            return int(model_action), None
        if self.candidate_mode == "strict":
            if near_turn and not self._strict_candidate_ok(reference_actions, reference_index, turn_anchor_distance):
                return int(model_action), None
        visibility_gate = difficulty_assessment
        if self._should_apply_visibility_gate():
            if not difficulty_assessment.get("pass_gate", False):
                self.stats["candidates_suppressed_by_visibility"] += 1
                return int(model_action), None

        candidate_cluster_id = self._cluster_id(
            turn_anchor_index if turn_anchor_index is not None else reference_index
        )
        candidate_cluster_key = (episode_id, candidate_cluster_id)
        if self._should_dedupe_candidate_cluster() and candidate_cluster_key in self.checked_candidate_clusters:
            self.stats["candidates_suppressed_by_cluster"] += 1
            return int(model_action), None
        if self._should_dedupe_candidate_cluster():
            self.checked_candidate_clusters.add(candidate_cluster_key)

        self.stats["candidate_events"] += 1
        if wrong_decision:
            self.stats["wrong_decision_candidates"] += 1

        cloud_meta = None
        if self.detector == "cloud":
            if observation is None:
                raise ValueError("Cloud intersection detector requires observation")
            if self.max_cloud_checks_per_episode >= 0 and self.cloud_checks_by_episode[episode_id] >= self.max_cloud_checks_per_episode:
                self.stats["candidates_suppressed_by_cloud_limit"] += 1
                return int(model_action), None
            is_intersection, cloud_meta = self.cloud_judge.judge(
                observation=observation,
                episode=episode,
                model_action=model_action,
                step=step,
                history=history or [],
                candidate_tags=local_candidate_tags,
                save_dir=self._input_save_dir(episode_id, step) if self.save_inputs else None,
            )
            if cloud_meta.get("saved_inputs"):
                self.stats["input_snapshots_saved"] += 1
            if cloud_meta.get("cache_hit"):
                self.stats["cloud_cache_hits"] += 1
            self.stats["cloud_checked_candidates"] += 1
            self.cloud_checks_by_episode[episode_id] += 1
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
                "near_reference_turn": bool(near_turn),
                "nearest_reference_index": int(reference_index),
                "turn_anchor_index": int(turn_anchor_index) if turn_anchor_index is not None else None,
                "turn_anchor_distance": int(turn_anchor_distance) if turn_anchor_distance is not None else None,
                "candidate_cluster_id": int(candidate_cluster_id),
                "wrong_decision": bool(wrong_decision),
                "candidate_mode": self.candidate_mode,
                "local_candidate_tags": local_candidate_tags,
                "visibility_gate": visibility_gate,
                "is_decision_difficulty": bool(is_intersection),
                "is_intersection_challenge": bool(is_intersection),
                "cloud": cloud_meta,
            }
            self.cloud_checks.append(check)
            if not is_intersection:
                return int(model_action), None
            self.stats["cloud_positive_events"] += 1
            positive_dir = self._archive_positive_inputs(episode_id, step, cloud_meta)
            if positive_dir is not None:
                cloud_meta["positive_inputs"] = positive_dir
                check["positive_input_dir"] = positive_dir
                self.positive_episode_ids.add(episode_id)
        elif self.detector != "reference_proxy":
            raise ValueError("Unsupported intersection_detector: {}".format(self.detector))

        if not wrong_decision:
            return int(model_action), None

        self.stats["intersection_errors"] += 1
        self.events_by_episode[episode_id] += 1

        executed_action = int(model_action)
        correction_applied = False
        correction_suppressed_reason = None
        if self.mode == "correct":
            correction_suppressed_reason = self._correction_suppression_reason(
                episode_id,
                step,
                candidate_cluster_id,
            )
            if correction_suppressed_reason is None:
                executed_action = reference_action
                correction_applied = True
                self.stats["corrections_applied"] += 1
                self.corrections_by_episode[episode_id] += 1
                self.last_correction_step_by_episode[episode_id] = int(step)
                self.corrections_by_episode_cluster[(episode_id, int(candidate_cluster_id))] += 1
            else:
                self.stats["corrections_suppressed"] += 1
                self.stats["corrections_suppressed_by_{}".format(correction_suppressed_reason)] += 1

        event = {
            "episode_id": episode_id,
            "trajectory_id": episode.get("trajectory_id"),
            "step": int(step),
            "nearest_reference_index": int(reference_index),
            "deviation_m": deviation,
            "near_reference_turn": bool(near_turn),
            "turn_anchor_index": int(turn_anchor_index) if turn_anchor_index is not None else None,
            "turn_anchor_distance": int(turn_anchor_distance) if turn_anchor_distance is not None else None,
            "candidate_mode": self.candidate_mode,
            "local_candidate_tags": local_candidate_tags,
            "visibility_gate": visibility_gate,
            "wrong_policy": self.policy,
            "wrong_decision": bool(wrong_decision),
            "cluster_id": int(candidate_cluster_id),
            "model_action_id": int(model_action),
            "model_action_name": action_name(model_action),
            "reference_action_id": int(reference_action),
            "reference_action_name": action_name(reference_action),
            "executed_action_id": int(executed_action),
            "executed_action_name": action_name(executed_action),
            "correction_applied": correction_applied,
            "correction_suppressed_reason": correction_suppressed_reason,
            "position": position.tolist() if position is not None else None,
        }
        if cloud_meta is not None:
            event["cloud"] = cloud_meta
            event["difficulty_type"] = cloud_meta.get("parsed", {}).get("difficulty_type")
            event["is_decision_difficulty"] = bool(cloud_meta.get("parsed", {}).get("is_decision_difficulty"))
            event["is_intersection_challenge"] = event["is_decision_difficulty"]
            if cloud_meta.get("positive_inputs"):
                event["positive_input_dir"] = cloud_meta.get("positive_inputs")
        self.events.append(event)
        return executed_action, event

    def _input_save_dir(self, episode_id, step):
        safe_episode_id = str(episode_id).replace("/", "_")
        return self.input_root / safe_episode_id / "step_{:04d}".format(int(step))

    def _positive_input_save_dir(self, episode_id, step):
        safe_episode_id = str(episode_id).replace("/", "_")
        return self.positive_input_root / safe_episode_id / "step_{:04d}".format(int(step))

    def _archive_positive_inputs(self, episode_id, step, cloud_meta):
        if not self.save_positive_inputs:
            return None
        saved_inputs = (cloud_meta or {}).get("saved_inputs") or {}
        request_path = saved_inputs.get("request")
        if not request_path:
            return None
        src_dir = Path(request_path).parent
        dst_dir = self._positive_input_save_dir(episode_id, step)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for child in src_dir.iterdir():
            if child.is_file():
                shutil.copy2(str(child), str(dst_dir / child.name))
        verdict_path = dst_dir / "decision_difficulty.json"
        verdict_path.write_text(
            json.dumps(
                {
                    "episode_id": str(episode_id),
                    "step": int(step),
                    "difficulty_type": (cloud_meta.get("parsed") or {}).get("difficulty_type"),
                    "parsed": cloud_meta.get("parsed"),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        self.stats["positive_input_snapshots_saved"] += 1
        positive_dir = str(dst_dir)
        self.positive_input_dirs[str(episode_id)].append(positive_dir)
        return positive_dir

    def should_preserve_positive_video(self, episode_id):
        return self.save_positive_videos and str(episode_id) in self.positive_episode_ids

    def preserve_positive_video(self, episode_id, video_path):
        if not self.should_preserve_positive_video(episode_id):
            return None
        src_path = Path(video_path)
        if not src_path.exists():
            return None
        self.positive_video_root.mkdir(parents=True, exist_ok=True)
        dst_path = self.positive_video_root / src_path.name
        shutil.copy2(str(src_path), str(dst_path))
        self.positive_video_paths[str(episode_id)] = str(dst_path)
        self.stats["positive_videos_saved"] += 1
        return str(dst_path)

    def _cluster_id(self, reference_index):
        width = max(1, int(self.turn_window) * 2 + 1)
        return int(reference_index) // width

    def _correction_suppression_reason(self, episode_id, step, candidate_cluster_id):
        if self.max_corrections_per_episode >= 0:
            if self.corrections_by_episode[episode_id] >= self.max_corrections_per_episode:
                return "episode_limit"

        last_step = self.last_correction_step_by_episode.get(episode_id)
        if last_step is not None and self.cooldown_steps > 0:
            if int(step) - int(last_step) <= self.cooldown_steps:
                return "cooldown"

        if self.max_corrections_per_cluster >= 0:
            cluster_key = (episode_id, int(candidate_cluster_id))
            if self.corrections_by_episode_cluster[cluster_key] >= self.max_corrections_per_cluster:
                return "cluster_limit"

        return None

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
