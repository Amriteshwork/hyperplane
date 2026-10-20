"""`InMemoryEventBus`: publisher and subscriber in one, for tests on a single event loop."""

import asyncio
from collections import defaultdict

from app.jobs.base import Event


class InMemoryEventBus:
    def __init__(self) -> None:
        self.published: dict[str, list[Event]] = defaultdict(list)
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = defaultdict(list)

    def publish(self, channel: str, event: Event) -> None:
        self.published[channel].append(event)
        for queue in self._subscribers[channel]:
            queue.put_nowait(event)

    async def subscribe(self, channel: str) -> _Subscription:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers[channel].append(queue)
        return _Subscription(self._subscribers[channel], queue)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers[channel])


class _Subscription:
    def __init__(
        self, subscribers: list[asyncio.Queue[Event]], queue: asyncio.Queue[Event]
    ) -> None:
        self._subscribers = subscribers
        self._queue = queue

    def __aiter__(self) -> _Subscription:
        return self

    async def __anext__(self) -> Event:
        return await self._queue.get()

    async def aclose(self) -> None:
        if self._queue in self._subscribers:
            self._subscribers.remove(self._queue)
