"""
Wrapper around Nectar's Swift object storage.

Note: Nectar object storage is native OpenStack Swift, NOT S3-compatible —
don't reach for boto3/s3fs here, they won't authenticate against it.

Containers are flat (no real subdirectories), but Swift lets you use "/"
in object names and most tools (incl. the dashboard) render that as a
pseudo-folder hierarchy. We use that to keep the same layout described
in schema.sql's swift_key columns, e.g.:

    raw/{video_id}.mp4
    segments/{video_id}/{segment_id}.wav
    features/{video_id}/{segment_id}.npy
"""
import os
import io
from functools import lru_cache

import swiftclient
import numpy as np


@lru_cache(maxsize=1)
def get_connection() -> swiftclient.Connection:
    return swiftclient.Connection(
        authurl=os.environ["SWIFT_AUTH_URL"],
        auth_version="3",
        os_options={
            "application_credential_id": os.environ["SWIFT_APPLICATION_CREDENTIAL_ID"],
            "application_credential_secret": os.environ["SWIFT_APPLICATION_CREDENTIAL_SECRET"],
        },
    )


def ensure_container(container: str) -> None:
    conn = get_connection()
    conn.put_container(container)  # no-op if it already exists


def put_bytes(container: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    conn = get_connection()
    conn.put_object(container, key, contents=data, content_type=content_type)
    return key


def put_file(container: str, key: str, local_path: str) -> str:
    with open(local_path, "rb") as f:
        return put_bytes(container, key, f.read())


def put_array(container: str, key: str, array: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, array)
    return put_bytes(container, key, buf.getvalue(), content_type="application/octet-stream")


def get_bytes(container: str, key: str) -> bytes:
    conn = get_connection()
    _, contents = conn.get_object(container, key)
    return contents


def get_array(container: str, key: str) -> np.ndarray:
    buf = io.BytesIO(get_bytes(container, key))
    return np.load(buf)


def object_exists(container: str, key: str) -> bool:
    conn = get_connection()
    try:
        conn.head_object(container, key)
        return True
    except swiftclient.exceptions.ClientException as e:
        if e.http_status == 404:
            return False
        raise
