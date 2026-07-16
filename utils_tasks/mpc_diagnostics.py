from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExpectedMotionZeroResult:
    expected_motion_zero: bool
    streak: int
    reason: str
    recovery_action: str


class ExpectedMotionZeroDetector:
    def __init__(self, planned_v_threshold=0.05, command_v_threshold=0.01, stall_frames=3):
        self.planned_v_threshold = float(planned_v_threshold)
        self.command_v_threshold = float(command_v_threshold)
        self.stall_frames = max(1, int(stall_frames))
        self.streak = 0

    def reset(self):
        self.streak = 0

    def update(self, planned_v, cmd_v, solve_success):
        if not solve_success:
            self.reset()
            return ExpectedMotionZeroResult(False, 0, "MPC_SOLVE_EXCEPTION", "RESET_MPC")
        planned = float(planned_v) if planned_v is not None else math.nan
        command = float(cmd_v) if cmd_v is not None else math.nan
        mismatch = (
            math.isfinite(planned) and math.isfinite(command)
            and planned > self.planned_v_threshold
            and abs(command) <= self.command_v_threshold
        )
        if not mismatch:
            self.reset()
            return ExpectedMotionZeroResult(False, 0, "NONE", "NONE")
        self.streak += 1
        stalled = self.streak >= self.stall_frames
        return ExpectedMotionZeroResult(
            True,
            self.streak,
            "EXPECTED_MOTION_ZERO_STALL" if stalled else "EXPECTED_MOTION_ZERO",
            "RESET_MPC_AND_REPLAN" if stalled else "OBSERVE",
        )
