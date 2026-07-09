import time

import numpy as np


class NavDPMincoAdapter:
    def __init__(
        self,
        esdf: dict,
        safe_dist=0.30,
        top_k=4,
        sample_dt=0.05,
        speed=0.5,
        enable=True,
        fallback_to_raw=True,
    ):
        self.esdf = esdf
        self.safe_dist = float(safe_dist)
        self.top_k = int(top_k)
        self.sample_dt = float(sample_dt)
        self.speed = float(speed)
        self.enabled = bool(enable)
        self.fallback_to_raw = bool(fallback_to_raw)
        self.processor = None
        if self.enabled:
            import minco_processor

            self.processor = minco_processor.MincoProcessor()
            if hasattr(self.processor, "configure"):
                self.processor.configure(
                    max_vel=self.speed,
                    safe_dist=self.safe_dist,
                    sample_dt=self.sample_dt,
                )
            self.processor.set_static_esdf_2d(
                distance=np.asarray(esdf["distance"], dtype=np.float64),
                free=np.asarray(esdf["free"], dtype=np.uint8),
                origin=np.asarray(esdf["origin"], dtype=np.float64),
                resolution=float(esdf["resolution"]),
            )

    def optimize_candidates(
        self,
        candidates_world,
        critic_values,
        states,
        raw_top1_world,
    ):
        candidates_world = np.asarray(candidates_world, dtype=object)
        critic_values = np.asarray(critic_values)
        raw_top1_world = np.asarray(raw_top1_world, dtype=object)
        batch_size = len(candidates_world)
        results = []
        for env_idx in range(batch_size):
            start_time = time.time()
            best = None
            failures = []
            order = self._candidate_order(critic_values[env_idx], len(candidates_world[env_idx]))
            for selected_idx in order[:max(0, self.top_k)]:
                candidate = self._as_guide_path(candidates_world[env_idx][selected_idx])
                if candidate is None:
                    failures.append(f"idx={selected_idx}: invalid_candidate")
                    continue
                try:
                    result = self.processor.optimize(
                        guide_path=candidate,
                        position=np.asarray(states[env_idx].get("position", np.zeros(3)), dtype=np.float64),
                        velocity=np.asarray(states[env_idx].get("velocity", np.zeros(3)), dtype=np.float64),
                        acceleration=np.asarray(states[env_idx].get("acceleration", np.zeros(3)), dtype=np.float64),
                        yaw=float(states[env_idx].get("yaw", 0.0)),
                        yaw_rate=float(states[env_idx].get("yaw_rate", 0.0)),
                    )
                except Exception as exc:
                    failures.append(f"idx={selected_idx}: {exc}")
                    continue
                if not result.get("success", False):
                    failures.append(f"idx={selected_idx}: {result.get('failure_reason', 'FAILED')}")
                    continue
                waypoints = np.asarray(result.get("waypoints", []), dtype=np.float64)
                if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
                    failures.append(f"idx={selected_idx}: invalid_waypoints")
                    continue
                scored = dict(result)
                scored["selected_index"] = int(selected_idx)
                if best is None or self._is_better(scored, best):
                    best = scored

            elapsed_ms = (time.time() - start_time) * 1000.0
            if best is not None:
                result = {
                    "success": True,
                    "waypoints": np.asarray(best["waypoints"], dtype=np.float64)[:, :2],
                    "samples": best.get("samples"),
                    "selected_index": int(best["selected_index"]),
                    "objective": float(best.get("objective", np.inf)),
                    "min_esdf": float(best.get("min_esdf", np.nan)),
                    "failure_reason": best.get("failure_reason", "NONE"),
                    "fallback": False,
                    "time_ms": elapsed_ms,
                }
                print(
                    "[NavDP-Minco] "
                    f"env={env_idx} success=1 fallback=0 selected_idx={result['selected_index']} "
                    f"objective={result['objective']:.4f} min_esdf={result['min_esdf']:.4f} "
                    f"time_ms={elapsed_ms:.2f}"
                )
            else:
                reason = "; ".join(failures[-3:]) if failures else "NO_VALID_CANDIDATE"
                result = self._fallback_result(env_idx, raw_top1_world[env_idx], reason, elapsed_ms)
            results.append(result)
        return results

    def _fallback_result(self, env_idx, raw_top1_world, reason, elapsed_ms):
        waypoints = np.asarray(raw_top1_world, dtype=np.float64)
        result = {
            "success": False,
            "waypoints": waypoints[:, :2] if waypoints.ndim == 2 else waypoints,
            "samples": None,
            "selected_index": -1,
            "objective": float("inf"),
            "min_esdf": float("nan"),
            "failure_reason": reason,
            "fallback": self.fallback_to_raw,
            "time_ms": elapsed_ms,
        }
        print(f"[NavDP-Minco] env={env_idx} success=0 fallback=1 reason={reason}")
        return result

    @staticmethod
    def _candidate_order(values, candidate_count):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        count = min(candidate_count, values.size)
        if count == 0:
            return []
        safe_values = values[:count].copy()
        safe_values[~np.isfinite(safe_values)] = -np.inf
        return list(np.argsort(-safe_values))

    @staticmethod
    def _as_guide_path(candidate):
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.ndim != 2 or candidate.shape[0] < 2 or candidate.shape[1] < 2:
            return None
        if not np.all(np.isfinite(candidate)):
            return None
        if candidate.shape[1] == 2:
            zeros = np.zeros((candidate.shape[0], 1), dtype=np.float64)
            candidate = np.concatenate([candidate, zeros], axis=1)
        else:
            candidate = candidate[:, :3]
        candidate[:, 2] = 0.0
        length = np.linalg.norm(np.diff(candidate[:, :2], axis=0), axis=1).sum()
        if not np.isfinite(length) or length < 1e-3:
            return None
        return candidate

    @staticmethod
    def _is_better(candidate, current):
        candidate_score = (float(candidate.get("objective", np.inf)), -float(candidate.get("min_esdf", -np.inf)))
        current_score = (float(current.get("objective", np.inf)), -float(current.get("min_esdf", -np.inf)))
        return candidate_score < current_score
