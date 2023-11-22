from typing import Protocol


class RayNotifier(Protocol):
    def notify(self, message: str) -> None:
        ...
