"""Pluggable output sinks for generated vCons and media.

* :class:`LocalStorage`     - write to the filesystem (default; no AWS needed).
* :class:`S3Storage`        - upload to S3, return a presigned URL (lazy boto3).
* :class:`ConserverStorage` - POST vCon JSON into a vcon-server ingress.
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


class ConserverStorage(Storage):
    """POST generated vCons into a vcon-server ingress (``POST /vcon``).

    Text modes only: the conserver API takes vCon JSON, so audio media must go
    to local/s3 storage instead.
    """

    def __init__(self, base_url: str, api_token: str | None, ingress: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = api_token
        self._ingress = ingress

    def save(self, key: str, data: bytes, content_type: str | None = None) -> str:
        if content_type != "application/json":
            raise ValueError(
                "conserver storage only accepts vCon JSON; use local or s3 storage for audio"
            )
        import requests  # vcon-lib dependency, already installed

        params = {"ingress_list": self._ingress} if self._ingress else {}
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["x-conserver-api-token"] = self._token
        response = requests.post(
            f"{self.base_url}/vcon", data=data, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
        vcon_uuid = os.path.basename(key).removesuffix(".vcon.json")
        return f"{self.base_url}/vcon/{vcon_uuid}"


def get_storage(settings: Settings) -> Storage:
    """Resolve a :class:`Storage` backend from settings."""
    if settings.storage_backend == "local":
        return LocalStorage(settings.output_dir)
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET is not configured for the 's3' storage backend")
        return S3Storage(settings.s3_bucket, settings.aws_access_key, settings.aws_secret_key)
    if settings.storage_backend == "conserver":
        if not settings.conserver_url:
            raise ValueError("CONSERVER_URL is not configured for the 'conserver' storage backend")
        return ConserverStorage(
            settings.conserver_url, settings.conserver_api_token, settings.conserver_ingress
        )
    raise ValueError(f"unknown storage backend: {settings.storage_backend!r}")
