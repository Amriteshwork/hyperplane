import pytest

from app.core.config import Settings


def test_cors_origins_are_comma_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test,")
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]
