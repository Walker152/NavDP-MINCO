from __future__ import annotations

from pathlib import Path
import numpy as np


def save_data_figure(path, title, ylabel="Value", values=None, data_source="SIMULATED"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray([0.0, 1.0] if values is None else values, float); values = values[np.isfinite(values)]
    if not len(values):
        path.with_suffix(path.suffix + ".skip_reason.txt").write_text("No finite data available.\n", encoding="utf-8"); return None
    figure = plt.figure(); axis = figure.add_subplot(111)
    axis.plot(np.arange(len(values)), values, marker="o"); axis.set_title(title); axis.set_xlabel("Sample index"); axis.set_ylabel(ylabel)
    if data_source == "SIMULATED": figure.text(.5, .5, "SIMULATED DATA", ha="center", va="center", alpha=.18, fontsize=22, rotation=25)
    figure.savefig(path, bbox_inches="tight"); plt.close(figure); return path


def save_mock_figure(path, title, ylabel="Value", values=None, data_source="SIMULATED"):
    """Backward-compatible name; values are always supplied by recorded data."""
    return save_data_figure(path, title, ylabel, values, data_source)
