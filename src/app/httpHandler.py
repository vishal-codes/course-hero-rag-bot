from typing import Any, Dict, Optional
from workers import Response
import json

# Custom headers for frontend 
EXPOSED_HEADERS = (
    "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After, X-Version, X-Service"
)

def get_cors_headers(origin: str | None = None) -> Dict[str, str]:
    headers = {
        "Access-Control-Allow-Origin": origin or "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "86400",
        "Access-Control-Expose-Headers": EXPOSED_HEADERS,
    }
    # Help caches/CDNs keep separate variants per Origin when not using '*'
    if origin and origin != "*":
        headers["Vary"] = "Origin"
    return headers

def json_response(
    data: Any,
    status: int = 200,
    origin: str | None = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Response:
    headers = {"content-type": "application/json"}
    headers.update(get_cors_headers(origin))
    if extra_headers:
        headers.update(extra_headers)
    return Response(json.dumps(data), status=status, headers=headers)

def json_error(
    message: str,
    detail: str | None = None,
    status: int = 400,
    origin: str | None = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Response:
    payload = {"error": message}
    if detail is not None:
        payload["detail"] = detail
    return json_response(payload, status=status, origin=origin, extra_headers=extra_headers)
