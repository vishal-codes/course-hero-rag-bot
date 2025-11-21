from workers import Response, WorkerEntrypoint
from urllib.parse import urlparse

from app.config import Config
from app.httpHandler import get_cors_headers, json_error, json_response
from app.pipeline import RAGPipeline
from app.utils import parse_json_body, env_get
from app.ratelimit import check_and_increment

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        config = Config.from_env(self.env)
        url = urlparse(str(request.url))
        origin = request.headers.get("Origin")

        if request.method == "OPTIONS":
            return Response("", status=204, headers=get_cors_headers(origin))

        if request.method == "GET":
            if  url.path == "/":
                return json_response({"message": "Hello Titan!"}, origin=origin,)
            elif url.path == "/health":
                return json_response({"OK": True}, origin=origin)
            elif url.path == "/version":
                return json_response({"version": config.app_version}, origin=origin)

        if request.method == "POST" and url.path == "/ask":
            return await self._handle_rag_request(request, config, origin)

        return Response("Not Found", status=404)


    async def _handle_rag_request(self, request, config: Config, origin: str) -> Response:
        rate_headers = {}

        # --- Identify IP and get KV binding ---
        ip = (request.headers.get("CF-Connecting-IP")
              or (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
              or None)
        kv = env_get(self.env, "KV_RATE")
        
        # --- Rate limit check BEFORE any heavy work ---
        if kv:
            allowed, remaining, reset = await check_and_increment(ip, kv, limit=3, window_secs=60)
            rate_headers.update({
                "X-RateLimit-Limit": "3",
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset),
            })
            if not allowed:
                rate_headers["Retry-After"] = str(reset)
                return json_error(
                    "Rate limit exceeded",
                    detail="3 requests per minute allowed for this IP.",
                    status=429,
                    origin=origin,
                    extra_headers=rate_headers,
                )

        # --- Parse and validate request body ---
        try:
            body = await parse_json_body(request)
        except ValueError as e:
            return json_error(str(e), status=400, origin=origin, extra_headers=rate_headers)

        question = (body.get("question") or "").strip()
        if not question:
            return json_error("Missing 'question'", status=400, origin=origin, extra_headers=rate_headers)

        try:
            top_k = int(body.get("topK") or config.topk)
            top_k = max(1, min(10, top_k))
        except Exception:
            return json_error("Invalid 'topK'", status=400, origin=origin, extra_headers=rate_headers)

        ai_binding = env_get(self.env, "AI")
        courses_binding = env_get(self.env, "COURSES")
        if not ai_binding:
            return json_error("AI binding missing", status=500, origin=origin, extra_headers=rate_headers)
        if not courses_binding:
            return json_error("COURSES binding missing", status=500, origin=origin, extra_headers=rate_headers)

        try:
            pipeline = RAGPipeline(config, ai_binding, courses_binding)
            result = await pipeline.process_question(question, top_k)
            return json_response(result, origin=origin, extra_headers=rate_headers)
        except Exception as e:
            msg = str(e)
            if "Embedding failed" in msg:
                message, status = "Embedding failed", 502
            elif "Vector search failed" in msg:
                message, status = "Vector search failed", 502
            elif "Answer generation failed" in msg:
                message, status = "Answer generation failed", 502
            else:
                message, status = "Pipeline failed", 502
            return json_error(message, detail=msg, status=status, origin=origin, extra_headers=rate_headers)
