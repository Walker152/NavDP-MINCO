from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def _classify_failure(episode_uid, plans, cycles):
    """Extract primary failure reason from planning data for an episode.

    Priority order:
      1. planning_cycles.failure_reason (non-empty, non-NONE)
      2. planning_cycles.validation_failure_reason
      3. plan_metrics failure indicators (hot_reject, python_validation)
      4. No plan ever published
      5. Fallback to done_reason or UNCLASSIFIED
    """
    ep_plans = [p for p in plans if p.get("episode_uid") == episode_uid]
    ep_cycles = [c for c in cycles if c.get("episode_uid") == episode_uid]

    # 1. Check planning_cycles failure_reason
    for c in ep_cycles:
        reason = str(c.get("failure_reason", "")).strip()
        if reason and reason not in ("NONE", "", "nan"):
            return f"MINCO_FAILED: {reason}"

    # 2. Check validation failure reason
    for c in ep_cycles:
        val_reason = str(c.get("validation_failure_reason", "")).strip()
        if val_reason and val_reason not in ("NONE", "", "nan"):
            return f"VALIDATION: {val_reason}"

    # 3. Check hot-start rejection as dominant pattern
    hot_rejects = [
        str(p.get("hot_reject_reason", ""))
        for p in ep_plans
        if str(p.get("hot_reject_reason", "")) not in ("", "HOT_ACCEPTED", "NONE")
    ]
    if hot_rejects:
        top = Counter(hot_rejects).most_common(1)[0]
        return f"HOT_START: {top[0]} (x{top[1]})"

    # 4. Check python validation failures
    py_fails = [
        p for p in ep_plans
        if str(p.get("python_validation_success", "")).lower() == "false"
    ]
    if py_fails:
        return f"PYTHON_VALIDATION_FAILED (x{len(py_fails)} plans)"

    # 5. Check if any plan was ever published in this episode
    published = [
        c for c in ep_cycles
        if str(c.get("published", "")).lower() == "true"
    ]
    if ep_cycles and not published:
        return "NO_PLAN_EVER_PUBLISHED"

    # 6. Check fallback mode dominance
    fallbacks = [
        str(c.get("fallback_mode", ""))
        for c in ep_cycles
        if str(c.get("fallback_mode", "")) not in ("", "NONE")
    ]
    if fallbacks:
        top = Counter(fallbacks).most_common(1)[0]
        return f"FALLBACK: {top[0]} (x{top[1]})"

    # 7. Check optimizer failures
    opt_fails = [
        c for c in ep_cycles
        if str(c.get("optimizer_success", "")).lower() == "false"
    ]
    if opt_fails:
        return f"OPTIMIZER_FAILED (x{len(opt_fails)} cycles)"

    # 8. Fallback to done_reason
    for row in episodes if isinstance(episodes, list) else []:
        if row.get("episode_uid") == episode_uid:
            done = str(row.get("done_reason", "")).strip()
            if done:
                return done

    return "TIMEOUT_OR_UNCLASSIFIED"


def generate_failure_report(suite_dir, episodes, plans=None, cycles=None):
    plans = plans or []
    cycles = cycles or []
    # ensure episodes is iterable for _classify_fallback
    episode_list = list(episodes) if not isinstance(episodes, list) else episodes
    reports = Path(suite_dir) / "reports"
    failures = [row for row in episode_list if str(row.get("success", "")).lower() != "true"]
    fields = ["episode_uid", "primary_reason", "secondary_reason", "scene_label", "variant", "data_source"]
    with (reports / "failure_case_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in failures:
            ep_uid = row.get("episode_uid", "")
            primary = _classify_failure(ep_uid, plans, cycles)
            writer.writerow({
                "episode_uid": ep_uid,
                "primary_reason": primary,
                "secondary_reason": row.get("done_reason", ""),
                "scene_label": row.get("scene_label", ""),
                "variant": row.get("variant", ""),
                "data_source": row.get("data_source", "UNKNOWN"),
            })
    sources = {row.get("data_source", "UNKNOWN") for row in episode_list}
    source = next(iter(sources)) if len(sources) == 1 else "MIXED"
    warning = "> SIMULATED DATA\n\n" if source == "SIMULATED" else ""
    lines = [f"- {row.get('episode_uid')}: {_classify_failure(row.get('episode_uid', ''), plans, cycles)}"
             for row in failures]
    text = f"# Failure Case Report\n\n{warning}Data source: {source}.\n\n" + ("\n".join(lines) or "No failures recorded.") + "\n"
    (reports / "failure_case_report.md").write_text(text, encoding="utf-8")
    return failures
