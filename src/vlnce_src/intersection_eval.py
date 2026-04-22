import math
from collections import defaultdict

import numpy as np

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


class IntersectionInterventionMonitor:
    def __init__(self, args):
        self.args = args
        self.mode = str(getattr(args, "intersection_eval_mode", "off")).lower()
        self.policy = str(getattr(args, "intersection_wrong_policy", "branch_mismatch")).lower()
        self.turn_window = int(getattr(args, "intersection_turn_window", 4))
        self.max_events_per_episode = int(getattr(args, "intersection_max_events_per_episode", -1))
        self.events = []
        self.events_by_episode = defaultdict(int)
        self.stats = {
            "mode": self.mode,
            "wrong_policy": self.policy,
            "turn_window": self.turn_window,
            "candidate_events": 0,
            "intersection_errors": 0,
            "corrections_applied": 0,
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

    def evaluate_step(self, episode, current_pose, model_action, step):
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
        self.events.append(event)
        return executed_action, event

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
        return summary
