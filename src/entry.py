from workers import Response, WorkerEntrypoint
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, Any, List
from js import Object
from pyodide.ffi import to_js, JsProxy

import json
import re

@dataclass
class Config:
    """Application configuration with defaults"""
    embed_model: str = "@cf/baai/bge-base-en-v1.5"
    chat_model: str = "@cf/meta/llama-3.1-8b-instruct-fast"
    topk: int = 5
    temperature: float = 0.2
    max_tokens: int = 350
    app_version: str = "dev"
    @classmethod
    def from_env(cls, env) -> 'Config':
        """Create config from environment variables"""
        return cls(
            embed_model=env_get(env, "CF_EMBED_MODEL", cls.embed_model),
            chat_model=env_get(env, "CF_CHAT_MODEL", cls.chat_model),
            topk=int(env_get(env, "CF_TOPK", cls.topk)),
            temperature=float(env_get(env, "GEN_TEMPERATURE", cls.temperature)),
            max_tokens=int(env_get(env, "GEN_MAX_TOKENS", cls.max_tokens)),
            app_version=env_get(env, "APP_VERSION", cls.app_version),
        )


# Utilities
def jsobj(data: dict):
    """Convert Python dict to JS object for Workers bindings."""
    return to_js(data, dict_converter=Object.fromEntries)

def to_py(obj):
    """Convert JsProxy to Python object."""
    return obj.to_py() if isinstance(obj, JsProxy) else obj

def env_get(env, key: str, default=None):
    """Safely get environment variable with fallback."""
    if env is None:
        return default
    value = getattr(env, key, None)
    if value is not None:
        return value
    try:
        return env.get(key, default)
    except Exception:
        return default


# Response Helpers 
def get_cors_headers(origin: str = None) -> Dict[str, str]:
    """Simple CORS headers"""
    headers = {
        "Access-Control-Allow-Origin": origin or "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "86400"
    }
    return headers

def json_response(data: Any, status: int = 200, origin: str = None) -> Response:
    """Create JSON response with CORS headers"""
    headers = {"content-type": "application/json"}
    headers.update(get_cors_headers(origin))
    return Response(json.dumps(data), status=status, headers=headers)

def json_error(message: str, detail: str = None, status: int = 400, origin: str = None) -> Response:
    """Create JSON error response"""
    payload = {"error": message}
    if detail is not None:
        payload["detail"] = detail
    return json_response(payload, status=status, origin=origin)

async def parse_json_body(request) -> Dict[str, Any]:
    """Parse and validate JSON request body"""
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


# RAG Pipeline 
class RAGPipeline:
    """Handles the RAG pipeline operations"""
    def __init__(self, config: Config, ai_binding, courses_binding):
        self.config = config
        self.ai = ai_binding
        self.courses = courses_binding
    def strip_references(self, text: str) -> str:
        """Remove citation markers like [1], [2], etc."""
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\(\s*ref:\s*\[\s*\d+\s*\]\s*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            response = await self.ai.run(self.config.embed_model, jsobj({"text": text}))
            result = to_py(response)
            data = result.get("data")
            if not data:
                raise RuntimeError(f"Empty embedding result: {result}")
            return data[0]
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
    async def search_vectors(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search vector database for similar matches"""
        vector_js = to_js(vector)
        options = {"topK": int(top_k), "returnValues": False, "returnMetadata": "all"}
        options_js = jsobj(options)
        try:
            response = await self.courses.query(vector_js, options_js)
            result = to_py(response)
        except Exception as e:
            raise RuntimeError(f"Vector search failed. New API: {e}")
        
        if not isinstance(result, dict):
            raise RuntimeError(f"Vector search failed: expected dict, got {type(result).__name__}")
        matches = result.get("matches")
        if not isinstance(matches, list):
            keys = list(result.keys())
            raise RuntimeError(
                f"Vector search failed: expected 'matches' list, got {type(matches).__name__}; keys={keys}"
            )
        return matches

    def build_context(self, matches: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """Build context string and sources array from vector matches"""
        blocks, sources = [], []
        for match in matches:
            metadata = match.get("metadata") or {}
            course = metadata.get("Course Name") or metadata.get("Course") or ""
            instructor = metadata.get("First Last") or ""
            description = metadata.get("Description") or ""
            prerequisite = metadata.get("Prerequisite") or ""
            lines = [f"Course: {course}", f"Instructor: {instructor}"]
            if description:
                lines.append(f"Description: {description}")
            if prerequisite:
                lines.append(f"Prerequisites: {prerequisite}")
            blocks.append("\n".join(lines))
            sources.append({
                "id": match.get("id"),
                "score": round(match.get("score") or 0, 3),
                "course": metadata.get("Course"),
                "courseName": metadata.get("Course Name"),
                "instructor": metadata.get("First Last"),
            })
        context = "\n\n".join(blocks) if blocks else "No context."
        print(f"Built context with {len(blocks)} blocks.")
        print("Context preview:", context)
        return context, sources
    
    def make_prompt(self, question: str, context: str) -> str:
        """Create prompt for the chat model"""
        return (
            "You are a helpful CSU Fullerton course assistant.\n"
            "Answer concisely using ONLY the provided context. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"# Context\n{context}\n\n"
            f"# Question\n{question}\n\n"
            "# Style\n- Keep it under 6 sentences.\n"
            "- Do not include references, IDs, or bracketed numbers in the answer.\n"
        )        

    async def generate_answer(self, prompt: str) -> str:
        """Generate answer using chat model"""
        try:
            inputs = jsobj({
                "prompt": prompt,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            })
            response = await self.ai.run(self.config.chat_model, inputs)
            result = to_py(response)
            answer = result.get("response") or (result.get("result") or {}).get("response") or ""
            answer = self.strip_references(answer)
            if not answer:
                raise RuntimeError(f"Empty generation result: {result}")
            return answer
        except Exception as e:
            raise RuntimeError(f"Answer generation failed: {e}") from e
    
    async def process_question(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Complete RAG pipeline: embed -> search -> generate"""
        if top_k is None:
            top_k = self.config.topk
        top_k = max(1, min(10, top_k))
        vector = await self.embed_text(question)
        matches = await self.search_vectors(vector, top_k)
        context, sources = self.build_context(matches)
        prompt = self.make_prompt(question, context)
        answer = await self.generate_answer(prompt)
        return {"answer": answer, "sources": sources}


# Main Worker 
class Default(WorkerEntrypoint):
    async def fetch(self, request):
        config = Config.from_env(self.env)
        url = urlparse(str(request.url))
        origin = request.headers.get("Origin")
        if request.method == "OPTIONS":
            return Response("", status=204, headers=get_cors_headers(origin))
        if request.method == "GET":
            if url.path == "/health":
                return json_response({"ok": True}, origin=origin)
            elif url.path == "/version":
                return json_response({"version": config.app_version}, origin=origin)
        if request.method == "POST" and url.path == "/ask":
            return await self._handle_rag_request(request, config, origin)
        return Response("Not Found", status=404)
    
    async def _handle_rag_request(self, request, config: Config, origin: str) -> Response:
        try:
            body = await parse_json_body(request)
        except ValueError as e:
            return json_error(str(e), status=400, origin=origin)
        question = (body.get("question") or "").strip()
        if not question:
            return json_error("Missing 'question'", status=400, origin=origin)
        try:
            top_k = int(body.get("topK") or config.topk)
            top_k = max(1, min(10, top_k))
        except Exception:
            return json_error("Invalid 'topK'", status=400, origin=origin)
        ai_binding = env_get(self.env, "AI")
        courses_binding = env_get(self.env, "COURSES")
        if not ai_binding:
            return json_error("AI binding missing", status=500, origin=origin)
        if not courses_binding:
            return json_error("COURSES binding missing", status=500, origin=origin)
        try:
            pipeline = RAGPipeline(config, ai_binding, courses_binding)
            result = await pipeline.process_question(question, top_k)
            return json_response(result, origin=origin)
        except RuntimeError as e:
            error_message = str(e)
            if "Embedding failed" in error_message:
                return json_error("Embedding failed", detail=error_message, status=502, origin=origin)
            elif "Vector search failed" in error_message:
                return json_error("Vector search failed", detail=error_message, status=502, origin=origin)
            elif "Answer generation failed" in error_message:
                return json_error("Answer generation failed", detail=error_message, status=502, origin=origin)
            else:
                return json_error("Pipeline failed", detail=error_message, status=502, origin=origin)
