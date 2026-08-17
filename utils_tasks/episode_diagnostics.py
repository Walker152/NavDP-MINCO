import time
import re
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TerminationObservation:
    reason: str
    raw_terms: str
    contact_detected: bool | None
    collision_object: str
    impact_force_n: float | None


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


def canonical_termination_reason(active_terms):
    terms = [str(term) for term in active_terms if str(term)]
    raw = "+".join(terms)
    if not terms:
        return "UNKNOWN", ""
    normalized = [re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_") for term in terms]
    if any("goal" in term or "success" in term for term in normalized):
        return "GOAL_REACHED", raw
    if any("collision" in term or "contact" in term for term in normalized):
        return "COLLISION", raw
    if any("time_out" in term or "timeout" in term or "time_limit" in term for term in normalized):
        return "TIMEOUT", raw
    return "+".join(term.upper() for term in normalized), raw


def infer_termination_details(infos, env_idx):
    """Return a canonical reason plus the explicit raw environment terms."""
    if not isinstance(infos, dict):
        return "UNKNOWN", ""
    for container_name in ("termination_terms", "terminations"):
        terms = infos.get(container_name)
        if isinstance(terms, dict):
            active = [name for name, values in terms.items() if _env_flag(values, env_idx)]
            if active:
                return canonical_termination_reason(active)
    if _env_flag(infos.get("time_outs", False), env_idx):
        return canonical_termination_reason(["time_out"])
    return "UNKNOWN", ""


def infer_termination_reason(infos, env_idx):
    return infer_termination_details(infos, env_idx)[0]


def termination_terms_from_manager(manager):
    """Snapshot IsaacLab v1.2 termination terms before the next env step."""
    if manager is None:
        return {}
    terms = {}
    for name in getattr(manager, "active_terms", ()):
        try:
            terms[str(name)] = manager.get_term(name)
        except (AttributeError, KeyError, RuntimeError):
            continue
    return terms


def observe_termination(
    infos,
    env_idx,
    *,
    contact_detected=None,
    collision_object="",
    impact_force_n=None,
    goal_reached=None,
):
    reason, raw_terms = infer_termination_details(infos, env_idx)
    contact = None if contact_detected is None else bool(contact_detected)
    if reason == "UNKNOWN" and contact is True:
        reason, raw_terms = canonical_termination_reason(
            ["contact_sensor_force"]
        )
    elif reason == "UNKNOWN" and goal_reached is True:
        reason, raw_terms = canonical_termination_reason(
            ["goal_distance_threshold"]
        )
    try:
        force = float(impact_force_n) if impact_force_n is not None else None
    except (TypeError, ValueError):
        force = None
    if force is not None and not math.isfinite(force):
        force = None
    return TerminationObservation(
        reason=reason,
        raw_terms=raw_terms,
        contact_detected=contact,
        collision_object=str(collision_object or ""),
        impact_force_n=force,
    )


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
