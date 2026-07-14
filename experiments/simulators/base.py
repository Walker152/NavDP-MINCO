from typing import Protocol


class ExperimentBackend(Protocol):
    name: str
    def run(self, run, episodes, writer) -> None: ...
