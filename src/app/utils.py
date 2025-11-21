from typing import Any, Dict
import json
from js import Object
from pyodide.ffi import to_js, JsProxy

def jsobj(data: dict):
    return to_js(data, dict_converter=Object.fromEntries)

def to_py(obj: Any) -> Any:
    return obj.to_py() if isinstance(obj, JsProxy) else obj

def env_get(env: Any, key: str, default: Any = None) -> Any:
    if env is None:
        return default
    val = getattr(env, key, None)
    if val is not None:
        return val
    try:
        return env.get(key, default)
    except Exception:
        return default

async def parse_json_body(request) -> Dict[str, Any]:
    try:
        body = await request.json()
        body = to_py(body)
        if isinstance(body, str):
            body = json.loads(body)
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body
    except Exception:
        try:
            text = await request.text()
            body = json.loads(text or "")
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            return body
        except Exception as e:
            raise ValueError("Body must be JSON object") from e
