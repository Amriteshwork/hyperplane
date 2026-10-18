"""Container healthcheck: exit 0 if an RQ worker from this host is registered in Redis.

Usage: python -m app.workers.healthcheck
"""

import socket
import sys

from redis import Redis
from rq import Worker

from app.core.config import get_settings


def main() -> int:
    """Return the process exit code."""
    connection = Redis.from_url(get_settings().redis_url, socket_connect_timeout=2)
    hostname = socket.gethostname()
    alive = any(worker.hostname == hostname for worker in Worker.all(connection=connection))
    return 0 if alive else 1


if __name__ == "__main__":
    sys.exit(main())
