import os
import sys

_build_dir = os.path.join(os.path.dirname(__file__), "build")
if os.path.isdir(_build_dir) and _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

try:
    from _minco_processor import MincoProcessor
except ImportError as exc:
    raise ImportError(
        "minco_processor native module is not built. Run "
        "`cmake -S minco_processor -B minco_processor/build` and "
        "`cmake --build minco_processor/build --target minco_processor_py`."
    ) from exc

__all__ = ["MincoProcessor"]
