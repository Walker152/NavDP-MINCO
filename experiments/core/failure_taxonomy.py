from __future__ import annotations

import hashlib
import json
from typing import Any


TAXONOMY_VERSION = "navdp-failure-taxonomy-v1"

_ROWS = {
    "NONE": ("PUBLISH", True, False, "NONE"),
    "INVALID_INPUT": ("INPUT", True, False, "STOP"),
    "NONFINITE_PATH": ("INPUT", True, False, "STOP"),
    "CANDIDATE_TIME_BUDGET_EXHAUSTED": (
        "CANDIDATE_SCREEN", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "CORRIDOR_GUIDE_UNSAFE": ("CORRIDOR", True, True, "TRY_NEXT_CANDIDATE"),
    "CORRIDOR_GUIDE_NEGATIVE_ESDF": (
        "CORRIDOR", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "CORRIDOR_GUIDE_OOB": ("CORRIDOR", True, True, "TRY_NEXT_CANDIDATE"),
    "CORRIDOR_GENERATION_FAILED": (
        "CORRIDOR", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "CORRIDOR_DISCONNECTED": ("CORRIDOR", True, True, "TRY_NEXT_CANDIDATE"),
    "OPTIMIZER_FAILED": ("OPTIMIZER", True, True, "TRY_NEXT_CANDIDATE"),
    "OPTIMIZER_NONFINITE": ("OPTIMIZER", True, True, "TRY_NEXT_CANDIDATE"),
    "OPTIMIZER_MAX_ITER": ("OPTIMIZER", True, True, "TRY_NEXT_CANDIDATE"),
    "VALIDATION_ESDF_OOB": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_CLEARANCE": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_NEGATIVE_ESDF": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_VELOCITY": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_ACCELERATION": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_JERK": ("CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"),
    "VALIDATION_YAW_RATE": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_WHEEL_SPEED": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_CORRIDOR": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_BUDGET_EXHAUSTED": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "VALIDATION_DEPTH_EXHAUSTED": (
        "CPP_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "PY_ESDF_VALIDATION_CLEARANCE": (
        "PYTHON_VALIDATION", True, True, "TRY_NEXT_CANDIDATE"
    ),
    "STALE_RESULT": ("PUBLISH", True, False, "HOLD_LAST_OR_STOP"),
    "HOLD_LAST": ("PUBLISH", True, False, "HOLD_LAST"),
    "STOP": ("PUBLISH", True, False, "STOP"),
    "MPC_SOLVE_FAILED": ("CONTROL", True, True, "STOP"),
    "COLLISION": ("SIMULATION", False, False, "TERMINATE"),
    "TIMEOUT": ("SIMULATION", False, False, "TERMINATE"),
    "GOAL_REACHED": ("SIMULATION", True, False, "TERMINATE"),
    "PLANNING_EXCEPTION": ("INFRASTRUCTURE", True, False, "STOP"),
}


def taxonomy_payload() -> dict[str, Any]:
    reasons = {
        reason: {
            "stage": stage,
            "safe_failure": safe_failure,
            "retry_next_candidate": retry,
            "recommended_recovery": recovery,
        }
        for reason, (stage, safe_failure, retry, recovery) in sorted(_ROWS.items())
    }
    payload = {"taxonomy_version": TAXONOMY_VERSION, "reasons": reasons}
    payload["taxonomy_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def classify_reason(reason: str) -> dict[str, Any]:
    canonical = str(reason or "NONE").split(":", 1)[0].strip()
    row = _ROWS.get(canonical)
    if row is None:
        return {
            "failure_stage": "UNKNOWN",
            "primary_reason": canonical or "UNKNOWN",
            "reason_source": "UNMAPPED",
            "recovery_action": "STOP",
            "safe_failure": False,
            "retry_next_candidate": False,
        }
    stage, safe_failure, retry, recovery = row
    return {
        "failure_stage": stage,
        "primary_reason": canonical,
        "reason_source": TAXONOMY_VERSION,
        "recovery_action": recovery,
        "safe_failure": safe_failure,
        "retry_next_candidate": retry,
    }
