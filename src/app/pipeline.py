from typing import Any, Dict, List, Tuple
import re
from pyodide.ffi import to_js
from .config import Config
from .utils import jsobj, to_py

class RAGPipeline:
    def __init__(self, config: Config, ai_binding: Any, courses_binding: Any):
        self.config = config
        self.ai = ai_binding
        self.courses = courses_binding

    def strip_references(self, text: str) -> str:
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\(\s*ref:\s*\[\s*\d+\s*\]\s*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    async def embed_text(self, text: str) -> List[float]:
        resp = await self.ai.run(self.config.embed_model, jsobj({"text": text}))
        result = to_py(resp)
        data = result.get("data")
        if not data:
            raise RuntimeError(f"Embedding failed: Empty embedding result: {result}")
        return data[0]

    async def search_vectors(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        vector_js = to_js(vector)
        options_js = jsobj({"topK": int(top_k), "returnValues": False, "returnMetadata": "all"})
        try:
            resp = await self.courses.query(vector_js, options_js)
            result = to_py(resp)
        except Exception as e:
            raise RuntimeError(f"Vector search failed. New API: {e}")
        if not isinstance(result, dict):
            raise RuntimeError(f"Vector search failed: expected dict, got {type(result).__name__}")
        matches = result.get("matches")
        if not isinstance(matches, list):
            keys = list(result.keys())
            raise RuntimeError(f"Vector search failed: expected 'matches' list, got {type(matches).__name__}; keys={keys}")
        return matches

    def build_context(self, matches: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        blocks, sources = [], []
        for match in matches:
            md = match.get("metadata") or {}
            course = md.get("Course Name") or md.get("Course") or ""
            instructor = md.get("First Last") or ""
            description = md.get("Description") or ""
            prerequisite = md.get("Prerequisite") or ""
            lines = [f"Course: {course}", f"Instructor: {instructor}"]
            if description:
                lines.append(f"Description: {description}")
            if prerequisite:
                lines.append(f"Prerequisites: {prerequisite}")
            blocks.append("\n".join(lines))
            sources.append({
                "id": match.get("id"),
                "score": round((match.get("score") or 0), 3),
                "course": md.get("Course"),
                "courseName": md.get("Course Name"),
                "instructor": md.get("First Last"),
            })
        context = "\n\n".join(blocks) if blocks else "No context."
        return context, sources

    def make_prompt(self, question: str, context: str) -> str:
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
        inputs = jsobj({
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        })
        resp = await self.ai.run(self.config.chat_model, inputs)
        result = to_py(resp)
        answer = result.get("response") or (result.get("result") or {}).get("response") or ""
        answer = self.strip_references(answer)
        if not answer:
            raise RuntimeError(f"Answer generation failed: Empty generation result: {result}")
        return answer

    async def process_question(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        if top_k is None:
            top_k = self.config.topk
        top_k = max(1, min(10, int(top_k)))
        vec = await self.embed_text(question)
        matches = await self.search_vectors(vec, top_k)
        context, sources = self.build_context(matches)
        prompt = self.make_prompt(question, context)
        answer = await self.generate_answer(prompt)
        return {"answer": answer, "sources": sources}
