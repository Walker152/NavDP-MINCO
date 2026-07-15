# Original NavDP RAW Baseline Provenance

The authoritative read-only source is the repository path `navdp_raw/` (the specification's `NAVDP/navdp_raw` directory is not present in this checkout).

| Source | SHA-256 |
|---|---|
| `navdp_raw/eval_pointgoal_wheeled.py` | `1239daeb5aa1d16194efe0cdabedfe30cb53a18fd0ab1425b6e282e3dcb78f39` |
| `navdp_raw/utils_tasks/tracking_utils.py` | `003e606926c03df83a93890cb555f66e1743cd3a836b23eca6dc470f9f51159a` |
| `navdp_raw/utils_tasks/basic_utils.py` | `c83cf81c8526dfd1d11828c699f3d8c1d1452581842011995479fcb9650` |
| `navdp_raw/utils_tasks/visualization_utils.py` | `641740ba146958a37c5e496f0b6afadf54dad3e06add44c87a09c56574299516` |

Defaults are N=15, desired/v/w limits=0.5, ref_gap=3, T=0.1, dense ratio=50, Q=(10,10,0), R=(.02,.15), IPOPT max_iter=100, acceptable_tol=1e-8 and acceptable_obj_change_tol=1e-6.

`original_mpc.py` is the executable RAW-arm adapter of `tracking_utils.py::MPC_Controller`. It preserves the original unicycle dynamics, objective, constraints, densification, reference selection, solver settings and warm-start arrays. Adaptations are limited to lazy CasADi import, finite/shape validation and the common `update_reference`, `get_current_reference` interface used by the experiment eval. `controller_factory.py` fixes all RAW algorithm parameters to the values above and never forwards MINCO temporal samples. Static tests compare densification and reference selection numerically with the read-only source without constructing or solving CasADi.
