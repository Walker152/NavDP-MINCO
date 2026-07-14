from __future__ import annotations

import itertools


def decide_warm_start(*, has_history, episode_matches, history_safe, remaining_duration, age, position_error, velocity_error, direction_dot, shifted_seed_valid, max_age=1.0, max_position_error=1.0, max_velocity_error=1.0, min_direction_dot=.5):
    if not has_history: return "COLD_START", "HOT_REJECT_NO_HISTORY"
    if not episode_matches: return "COLD_START", "HOT_REJECT_EPISODE_MISMATCH"
    if not history_safe: return "COLD_START", "HOT_REJECT_HISTORY_UNSAFE"
    if remaining_duration <= 0 or age > max_age: return "COLD_START", "HOT_REJECT_EXPIRED"
    if position_error > max_position_error: return "COLD_START", "HOT_REJECT_POSITION"
    if velocity_error > max_velocity_error: return "COLD_START", "HOT_REJECT_VELOCITY"
    if direction_dot < min_direction_dot: return "COLD_START", "HOT_REJECT_DIRECTION"
    if not shifted_seed_valid: return "COLD_START", "HOT_REJECT_INVALID_SHIFT"
    return "HOT_START", "HOT_ACCEPTED"


class ProposalLifecycle:
    def __init__(self):
        self._ids = itertools.count(1); self._generation = {}; self._history = {}; self._proposals = {}; self._snapshot = {}

    def reset_history(self, env_id, episode_generation):
        self._generation[env_id] = int(episode_generation); self._history.pop(env_id, None); self._snapshot[env_id] = self._snapshot.get(env_id, 0) + 1
        for key in [key for key in self._proposals if key[0] == env_id]: self._proposals.pop(key)

    def committed_history(self, env_id): return self._history.get(env_id)

    def optimize_preview(self, env_id, episode_generation, payload, safe):
        if self._generation.get(env_id) != int(episode_generation): raise ValueError("episode generation mismatch")
        proposal_id = next(self._ids); proposal = {"proposal_id":proposal_id, "env_id":env_id, "episode_generation":int(episode_generation), "history_snapshot_id":self._snapshot.get(env_id, 0), "payload":payload, "safe":bool(safe)}
        self._proposals[(env_id, proposal_id)] = proposal; return dict(proposal)

    def discard_proposal(self, env_id, proposal_id): self._proposals.pop((env_id, proposal_id), None)

    def commit_history(self, env_id, proposal_id, *, selected, published, stale):
        proposal = self._proposals.get((env_id, proposal_id))
        if proposal is None: raise ValueError("unknown proposal")
        if not proposal["safe"]: raise ValueError("unsafe proposal cannot be committed")
        if not selected or not published or stale: raise ValueError("only selected, published, non-stale proposals can be committed")
        if proposal["episode_generation"] != self._generation.get(env_id): raise ValueError("episode generation mismatch")
        self._history[env_id] = dict(proposal); self._snapshot[env_id] = self._snapshot.get(env_id, 0) + 1
        for key in [key for key in self._proposals if key[0] == env_id]: self._proposals.pop(key)
        return dict(self._history[env_id])
