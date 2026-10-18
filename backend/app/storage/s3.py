"""S3-compatible storage (Garage in Compose, any S3 API in prod)."""

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.storage.base import BlobNotFoundError, validate_key

_MISSING = {"404", "NoSuchKey", "NotFound"}


class S3Storage:
    """Blobs are objects in one bucket."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None,
        region: str,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> None:
        self._bucket = bucket
        # boto3 ships no type information; keep the untyped client private.
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=2,
                read_timeout=30,
            ),
        )

    def read(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=validate_key(key))
        except ClientError as exc:
            if _error_code(exc) in _MISSING:
                raise BlobNotFoundError(key) from None
            raise
        body: bytes = response["Body"].read()
        return body

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=validate_key(key))
        except ClientError as exc:
            if _error_code(exc) in _MISSING:
                return False
            raise
        return True

    def write(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=validate_key(key), Body=data)

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=validate_key(key))


def _error_code(exc: ClientError) -> str:
    code: str = exc.response.get("Error", {}).get("Code", "")
    return code
