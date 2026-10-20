"""Dotted paths of worker job functions. The API enqueues by name and never imports them."""

from uuid import UUID

PROFILE_DATASET = "app.workers.jobs.profile_dataset"
TRAIN_RUN = "app.workers.jobs.train_run"


def run_events_channel(run_id: UUID) -> str:
    """Pub/sub channel of one run's progress events."""
    return f"runs:{run_id}:events"
