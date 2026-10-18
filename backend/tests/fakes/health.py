"""Health probes with a scripted outcome."""


class StaticProbe:
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self._error = error

    async def check(self) -> None:
        if self._error is not None:
            raise self._error
