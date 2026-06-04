"""Pluggable output sinks for generated vCons and media.

* :class:`LocalStorage` - write to the filesystem (default; no AWS needed).
* :class:`S3Storage`    - upload to S3 and return a presigned URL (lazy boto3).

Future backends (GCS, vcon-server, vcon-mcp) implement the same :class:`Storage`
interface.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from vconfaker.config import Settings


class Storage(ABC):
    """Persist a blob under ``key`` and return a locator (URL or file URI)."""

    @abstractmethod
    def save(self, key: str, data: bytes, content_type: str | None = None) -> str:
        ...


class LocalStorage(Storage):
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def save(self, key: str, data: bytes, content_type: str | None = None) -> str:
        path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return f"file://{os.path.abspath(path)}"


class S3Storage(Storage):
    def __init__(self, bucket: str, access_key: str | None, secret_key: str | None) -> None:
        self.bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key

    def _client(self):  # type: ignore[no-untyped-def]
        import boto3  # lazy import; only needed for the s3 backend

        return boto3.client(
            "s3",
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
        )

    def save(self, key: str, data: bytes, content_type: str | None = None) -> str:
        client = self._client()
        extra = {"ContentType": content_type} if content_type else {}
        client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}
        )


def get_storage(settings: Settings) -> Storage:
    """Resolve a :class:`Storage` backend from settings."""
    if settings.storage_backend == "local":
        return LocalStorage(settings.output_dir)
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET is not configured for the 's3' storage backend")
        return S3Storage(settings.s3_bucket, settings.aws_access_key, settings.aws_secret_key)
    raise ValueError(f"unknown storage backend: {settings.storage_backend!r}")
