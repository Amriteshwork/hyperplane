"""Column types shared by the ORM models."""

from datetime import datetime
from typing import Annotated

from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import mapped_column

Timestamp = Annotated[datetime, mapped_column(DateTime(timezone=True))]
Json = JSONB(none_as_null=True)
