"""Protocols services depend on; RQ/Redis details stay in the implementations."""

from collections.abc import Mapping
from typing import Protocol

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class JobQueue(Protocol):
    def enqueue(self, job: str, /, **kwargs: JsonValue) -> str:
        """Enqueue the job function at dotted path `job`; return the job id."""
        ...


class EventPublisher(Protocol):
    def publish(self, channel: str, event: Mapping[str, JsonValue]) -> None:
        """Publish one JSON event to `channel` (e.g. a run's progress stream)."""
        ...
