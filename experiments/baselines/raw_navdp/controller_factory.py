from __future__ import annotations


def create_tracking_controller(
    variant,
    geometric_path,
    trajectory_samples=None,
    *,
    raw_controller_cls=None,
    minco_controller_cls=None,
    **kwargs,
):
    """Select the method-specific controller without crossing experiment arms."""
    if variant == "raw":
        from .original_mpc import ORIGINAL_MPC_SPEC
        if raw_controller_cls is None:
            from .original_mpc import RawNavDPMPCController
            raw_controller_cls = RawNavDPMPCController
        fixed = {key: ORIGINAL_MPC_SPEC[key] for key in ("N", "desired_v", "v_max", "w_max", "ref_gap", "T")}
        return raw_controller_cls(geometric_path, **fixed)
    if variant not in {"minco-cold", "minco-hot"}:
        raise ValueError(f"unsupported experiment variant: {variant}")
    if trajectory_samples is None:
        raise ValueError(f"{variant} requires MINCO temporal samples")
    if minco_controller_cls is None:
        from utils_tasks.tracking_utils import MPC_Controller
        minco_controller_cls = MPC_Controller
    return minco_controller_cls(
        geometric_path,
        trajectory_samples=trajectory_samples,
        allow_geometric_fallback=False,
        **kwargs,
    )


def update_tracking_reference(controller, variant, geometric_path, trajectory_samples=None, desired_v=None):
    if variant == "raw":
        from .original_mpc import ORIGINAL_MPC_SPEC
        return controller.update_reference(geometric_path, desired_v=ORIGINAL_MPC_SPEC["desired_v"])
    if trajectory_samples is None:
        return False
    return controller.update_reference(geometric_path, trajectory_samples=trajectory_samples, desired_v=desired_v)
