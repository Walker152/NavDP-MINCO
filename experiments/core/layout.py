from pathlib import Path

from .models import RunSpec


class ResultLayout:
    def __init__(self, output_root: Path | str):
        self.output_root = Path(output_root)

    def suite_dir(self, suite_id: str) -> Path:
        return self.output_root / suite_id

    def run_dir(self, run: RunSpec) -> Path:
        return self.suite_dir(run.suite_id) / "experiments" / run.experiment_id / run.scene_label / run.variant / str(run.seed) / run.run_id

    def reports_dir(self, suite_id: str) -> Path:
        return self.suite_dir(suite_id) / "reports"
