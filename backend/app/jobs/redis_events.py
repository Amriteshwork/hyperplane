"""Progress events over Redis pub/sub: the worker publishes, the SSE endpoint subscribes."""

import json
import time

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub

from app.jobs.base import Event, EventStream

# Poll interval; kept below the client's socket timeout so an idle channel isn't an error.
_POLL_SECONDS = 1.0


class RedisEventPublisher:
    def __init__(self, connection: Redis) -> None:
        self._redis = connection

    def publish(self, channel: str, event: Event) -> None:
        self._redis.publish(channel, json.dumps(event, separators=(",", ":")))


class RedisEventSubscriber:
    def __init__(self, connection: AsyncRedis, *, max_seconds: float) -> None:
        self._redis = connection
        self._max_seconds = max_seconds

    async def subscribe(self, channel: str) -> EventStream:
        pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel)
        return _Subscription(pubsub, deadline=time.monotonic() + self._max_seconds)


class _Subscription:
    """Ends at the deadline too, so a stuck job can't hold a stream open forever."""

    def __init__(self, pubsub: PubSub, *, deadline: float) -> None:
        self._pubsub = pubsub
        self._deadline = deadline
        self._closed = False

    def __aiter__(self) -> _Subscription:
        return self

    async def __anext__(self) -> Event:
        while not self._closed and time.monotonic() < self._deadline:
            message = await self._pubsub.get_message(timeout=_POLL_SECONDS)
            if message is not None and message["type"] == "message":
                event: Event = json.loads(message["data"])
                return event
        await self.aclose()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._pubsub.aclose()  # type: ignore[no-untyped-call]  # redis-py leaves it unannotated
