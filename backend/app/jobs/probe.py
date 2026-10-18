"""Health probe for Redis."""

from redis.asyncio import Redis


class RedisProbe:
    name = "redis"

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def check(self) -> None:
        await self._client.ping()
