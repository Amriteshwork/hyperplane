"""Protocols services depend on; RQ/Redis details stay in the implementations."""

from collections.abc import AsyncIterator, Mapping
from typing import Protocol

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type Event = Mapping[str, JsonValue]


class JobQueue(Protocol):
    def enqueue(self, job: str, /, **kwargs: JsonValue) -> str:
        """Enqueue the job function at dotted path `job`; return the job id."""
        ...


class EventPublisher(Protocol):
    def publish(self, channel: str, event: Event) -> None:
        """Publish one JSON event to `channel` (e.g. a run's progress stream)."""
        ...


class EventStream(AsyncIterator[Event], Protocol):
    async def aclose(self) -> None:
        """Stop listening; safe to call more than once, and before iterating."""
        ...


class EventSubscriber(Protocol):
    async def subscribe(self, channel: str) -> EventStream:
        """Start listening to `channel`; events published from now on arrive in order.

        The subscription is active when this returns.
        """
        ...
