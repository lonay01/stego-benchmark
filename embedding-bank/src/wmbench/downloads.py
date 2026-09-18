"""Bounded retry of transport failures, without disabling TLS or masking bad models."""

from __future__ import annotations

import time


def download_snapshot(
    repo: str, revision: str, directory: str, ignore_patterns=None, attempts: int = 3
) -> None:
    from huggingface_hub import snapshot_download
    from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
    from requests.exceptions import HTTPError

    for attempt in range(attempts):
        try:
            snapshot_download(
                repo_id=repo,
                revision=revision,
                local_dir=directory,
                max_workers=2,
                ignore_patterns=ignore_patterns or None,
            )
            return
        except (ChunkedEncodingError, ConnectionError, Timeout, HTTPError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status is not None and status not in (408, 429, 500, 502, 503, 504):
                raise  # Missing/gated assets do not become endless retries.
            if attempt + 1 == attempts:
                raise
            print(
                f"HF transport retry {attempt + 1}/{attempts - 1}: {type(error).__name__}",
                flush=True,
            )
            time.sleep(2 ** (attempt + 1))
