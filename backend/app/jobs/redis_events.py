"""`EventPublisher` backed by Redis pub/sub (consumed by the SSE endpoint)."""

import json
from collections.abc import Mapping

from redis import Redis

from app.jobs.base import JsonValue


class RedisEventPublisher:
    def __init__(self, connection: Redis) -> None:
        self._redis = connection

    def publish(self, channel: str, event: Mapping[str, JsonValue]) -> None:
        self._redis.publish(channel, json.dumps(event, separators=(",", ":")))
