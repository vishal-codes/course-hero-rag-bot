import json
import time
from typing import Any, Tuple

def _now_s() -> int:
    return int(time.time())

async def check_and_increment(ip: str, kv: Any, limit: int = 3, window_secs: int = 60
) -> Tuple[bool, int, int]:
    # If can't identify IP, do not block
    if not ip:
        return True, limit, window_secs

    now = _now_s()
    # Align windows to exact minute boundaries for predictable resets
    window_start = now - (now % window_secs)
    window_end = window_start + window_secs
    ttl = max(1, window_end - now)

    key = f"rl:{ip}"
    # Read current counter
    raw = await kv.get(key)  
    data = None
    if raw:
        # raw is a JS string bridged to Python; parse it
        if hasattr(raw, "to_py"):
            raw = raw.to_py()
        data = json.loads(raw)

    if not data or data.get("window_start") != window_start:
        # New window
        data = {"window_start": window_start, "count": 1}
        await kv.put(key, json.dumps(data), {"expirationTtl": ttl})
        remaining = max(0, limit - 1)
        return True, remaining, ttl

    # Same window; increment if under limit
    count = int(data.get("count", 0))
    if count >= limit:
        # Already at limit
        reset = max(1, window_end - now)
        return False, 0, reset

    data["count"] = count + 1
    await kv.put(key, json.dumps(data), {"expirationTtl": ttl})
    remaining = max(0, limit - data["count"])
    reset = max(1, window_end - now)
    return True, remaining, reset
