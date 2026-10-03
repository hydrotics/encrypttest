from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatusSnapshot:
    reveal_id: str | None
    requested_users: int
    successful_users: int

    @property
    def percent(self) -> float:
        if self.requested_users <= 0:
            return 0.0
        return min(100.0, self.successful_users * 100.0 / self.requested_users)

    @property
    def completed_display(self) -> str:
        if self.requested_users <= 0:
            return "0"
        return f"{self.successful_users} / {self.requested_users}"

    def progress_bar(self, width: int = 20) -> str:
        width = max(1, int(width))
        filled = min(width, int(self.percent * width / 100.0))
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    def bot_state(self, waiting: int, running: int, estimated_wait: float, ready: bool) -> str:
        if not ready:
            return "OFFLINE"
        if waiting >= 10 or estimated_wait >= 20:
            return "OVERLOADED"
        if waiting > 0 or running > 0:
            return "BUSY"
        return "NORMAL"


class StatusTracker:
    def __init__(self) -> None:
        self.reveal_id: str | None = None
        self._requested: set[int] = set()
        self._successful: set[int] = set()

    def reset(self, reveal_id: str | None) -> None:
        self.reveal_id = str(reveal_id) if reveal_id else None
        self._requested.clear()
        self._successful.clear()

    def request(self, user_id: int) -> None:
        self._requested.add(int(user_id))

    def success(self, user_id: int) -> None:
        uid = int(user_id)
        self._requested.add(uid)
        self._successful.add(uid)

    def snapshot(self) -> StatusSnapshot:
        return StatusSnapshot(
            reveal_id=self.reveal_id,
            requested_users=len(self._requested),
            successful_users=len(self._successful),
        )
