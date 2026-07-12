import time


def _env_flag(value, env_idx):
    try:
        value = value[env_idx]
    except (IndexError, KeyError, TypeError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, ValueError):
            return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def infer_termination_reason(infos, env_idx):
    """Return only a reason explicitly exposed by the environment."""
    if not isinstance(infos, dict):
        return "UNKNOWN"
    for container_name in ("termination_terms", "terminations"):
        terms = infos.get(container_name)
        if isinstance(terms, dict):
            active = [name for name, values in terms.items() if _env_flag(values, env_idx)]
            if active:
                return "+".join(active)
    if _env_flag(infos.get("time_outs", False), env_idx):
        return "time_out"
    return "UNKNOWN"


class EpisodeStartupDiagnostics:
    def __init__(self, num_envs, clock=time.monotonic):
        self._clock = clock
        self._states = []
        for _ in range(num_envs):
            self._states.append(self._new_state(0))

    def _new_state(self, generation):
        return {
            "generation": int(generation),
            "started_at": float(self._clock()),
            "first_valid_plan_received": False,
            "startup_attempt_count": 0,
            "last_minco_status": "NOT_RUN",
        }

    def begin_generation(self, env_idx, generation):
        self._states[env_idx] = self._new_state(generation)

    def note_planning_result(self, env_idx, generation, status, success):
        state = self._states[env_idx]
        if int(generation) != state["generation"]:
            return False
        if not state["first_valid_plan_received"]:
            state["startup_attempt_count"] += 1
        state["last_minco_status"] = str(status)
        if success:
            state["first_valid_plan_received"] = True
        return True

    def snapshot(self, env_idx, generation):
        state = self._states[env_idx]
        if int(generation) != state["generation"]:
            return {
                "first_valid_plan_received": False,
                "startup_attempt_count": 0,
                "startup_planning_elapsed": 0.0,
                "last_minco_status": "STALE_GENERATION",
            }
        return {
            "first_valid_plan_received": bool(state["first_valid_plan_received"]),
            "startup_attempt_count": int(state["startup_attempt_count"]),
            "startup_planning_elapsed": max(0.0, float(self._clock()) - state["started_at"]),
            "last_minco_status": state["last_minco_status"],
        }
