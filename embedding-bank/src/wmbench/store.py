"""A result is reusable only when its specification and all artifacts still match."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

from .io import digest, read_json, retry_read, sha256, write_json


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for subdir in ("records", "artifacts", "locks", "plans", "logs", "tables"):
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def identifier(self, spec: dict) -> str:
        return digest(spec)

    def path(self, spec: dict) -> Path:
        return self.root / "records" / f"{self.identifier(spec)}.json"

    def valid(self, spec: dict, include_terminal: bool = True) -> dict | None:
        path = self.path(spec)
        try:
            record = read_json(path)
            if record["spec"] != spec or record["job_id"] != self.identifier(spec):
                return None
            if record["status"] != "ok" and not include_terminal:
                return None
            if record["status"] == "failed":
                return None
            for artifact in record["artifacts"]:
                target = Path(artifact["path"])
                if not retry_read(target.is_file, target) or sha256(target) != artifact["sha256"]:
                    return None
            return record
        except (ValueError, FileNotFoundError, KeyError):
            return None

    def publish(
        self, spec: dict, status: str, data: dict, paths: list[str | Path], seconds: float = 0
    ) -> dict:
        artifacts = [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in paths]
        record = {
            "job_id": self.identifier(spec),
            "spec": spec,
            "status": status,
            "data": data,
            "artifacts": artifacts,
            "seconds": seconds,
            "finished_unix": time.time(),
        }
        write_json(self.path(spec), record)
        return record

    @contextmanager
    def claim(self, spec: dict):
        """Exclusive claim, including across processes. Stale claims require explicit unlock."""
        path = self.root / "locks" / f"{self.identifier(spec)}.lock"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, f"pid={os.getpid()} time={time.time()}".encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)
