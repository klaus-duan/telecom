from __future__ import annotations

import time
import uuid


def new_id() -> str:
    return str(uuid.uuid4())


def now_ts() -> float:
    return time.time()
