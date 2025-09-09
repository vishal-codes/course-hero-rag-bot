import os, requests, textwrap
from dotenv import load_dotenv

EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"
CHAT_MODEL  = os.getenv("CF_CHAT_MODEL", "@cf/meta/llama-3.1-8b-instruct-fast")
INDEX       = os.getenv("CF_INDEX", "csuf-courses")
TOP_K       = int(os.getenv("CF_TOPK", "5"))

load_dotenv("./.dev.vars")
ACC = os.getenv("CLOUDFLARE_ACCOUNT_ID")
TOK = os.getenv("CLOUDFLARE_API_TOKEN")
if not ACC or not TOK:
    raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the env")

API = f"https://api.cloudflare.com/client/v4/accounts/{ACC}"

def embed(text: str):
    r = requests.post(
        f"{API}/ai/run/{EMBED_MODEL}",
        headers={"Authorization": f"Bearer {TOK}"},
        json={"text": [text]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["result"]["data"][0]

def search(vec):
    r = requests.post(
        f"{API}/vectorize/v2/indexes/{INDEX}/query",
        headers={"Authorization": f"Bearer {TOK}"},
        json={"vector": vec, "topK": TOP_K, "returnValues": False, "returnMetadata": "all"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["result"]["matches"]

def build_context(matches):
    blocks = []
    for i, m in enumerate(matches, 1):
        md = m.get("metadata", {})
        parts = [
            f"Course: {md.get('Course Name','')} ({md.get('Course','')})",
            f"Instructor: {md.get('First Last','')}",
        ]
        if md.get("Description"):
            parts.append(f"Description: {md['Description']}")
        if md.get("Prerequisite"):
            parts.append(f"Prerequisites: {md['Prerequisite']}")
        blocks.append(f"[{i}] id={m.get('id')}  score={m.get('score'):.3f}\n" + "\n".join(parts))
    return "\n\n".join(blocks)

def make_prompt(question: str, context: str):
    return textwrap.dedent(f"""
    You are a helpful university course assistant. Answer using ONLY the context.
    If the answer is not in the context, say you don't know.

    # Context
    {context}

    # Question
    {question}

    # Instructions
    - Do not include references, IDs, or bracketed numbers in the answer.
    - Keep the answer concise (<= 6 sentences).
    - If prerequisites vary by instructor, note that explicitly.
    """)

def generate(prompt: str):
    r = requests.post(
        f"{API}/ai/run/{CHAT_MODEL}",
        headers={"Authorization": f"Bearer {TOK}"},
        json={"prompt": prompt},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["result"]["response"]

if __name__ == "__main__":
    import sys
    question = "How important is Data Structures course? Is it a pre req for any other course?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    qvec = embed(question)
    hits = search(qvec)
    ctx = build_context(hits)
    prompt = make_prompt(question, ctx)
    answer = generate(prompt)

    print("\n=== Answer ===\n", answer)
    print("\n=== Sources ===")
    for i, m in enumerate(hits, 1):
        md = m.get("metadata", {})
        print(f"[{i}] {md.get('Course Name')} ({md.get('Course')}) · {md.get('First Last')} · id={m.get('id')} · score={m.get('score'):.3f}")
