"""Cross-process file lock for index mutation.

GLDIS persists FAISS and BM25 indices to disk. When multiple documents are
processed concurrently (or when uvicorn uses multiple workers), concurrent
writers can corrupt these files.

This lock uses an atomic lockfile create (O_EXCL) so it works on Windows and
Linux without extra dependencies.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class LockOptions:
    timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.1
    stale_seconds: float = 30.0 * 60.0  # 30 minutes


class LockTimeoutError(TimeoutError):
    pass


def _now() -> float:
    return time.time()


def _is_stale(lock_path: Path, *, stale_seconds: float) -> bool:
    try:
        age = _now() - lock_path.stat().st_mtime
        return age > stale_seconds
    except FileNotFoundError:
        return False


@contextmanager
def index_lock(
    lock_path: str | Path,
    *,
    options: Optional[LockOptions] = None,
    metadata: Optional[dict] = None,
) -> Iterator[None]:
    """Acquire an exclusive lock via a lockfile.

    Args:
        lock_path: Path to the lockfile (e.g., <index_dir>/.write.lock).
        options: Lock behavior (timeout/poll/stale).
        metadata: Optional dict to write inside the lockfile for debugging.

    Raises:
        LockTimeoutError: if lock cannot be acquired within timeout.
    """

    opts = options or LockOptions()
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = _now()
    fd: Optional[int] = None

    try:
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {
                    "pid": os.getpid(),
                    "created_at": _now(),
                    "metadata": metadata or {},
                }
                try:
                    os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
                except Exception:
                    # Best-effort only
                    pass
                break
            except FileExistsError:
                if _is_stale(lock_path, stale_seconds=opts.stale_seconds):
                    try:
                        lock_path.unlink(missing_ok=True)  # py3.11
                        continue
                    except Exception:
                        # If we can't remove, we still respect timeout
                        pass

                if (_now() - start) >= opts.timeout_seconds:
                    raise LockTimeoutError(f"Timed out waiting for lock: {lock_path}")

                time.sleep(opts.poll_interval_seconds)

        yield

    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
