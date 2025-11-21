

# CSUF Python RAG Bot

A compact Retrieval-Augmented Generation (RAG) API on **Cloudflare Workers (Python)** that:

* embeds user questions with **Workers AI**, queries a **Vectorize** index, and returns concise answers
* enforces CORS and consistent JSON errors
* supports per-IP rate limiting (3 requests/minute) via **Workers KV** 


This api powers the [Course_Hero Frontend](https://github.com/vishal-codes/course-hero)

## Directory structure

```
vishal-codes-course-hero-rag-bot/
├── README.md
├── example_query.py
├── example_rag_answer.py
├── package.json
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .python-version
├── data_loader/
│   ├── courses.csv
│   ├── vector_builder.py
│   └── vectors_to_upload.ndjson
└── src/
    ├── entry.py
    └── app/
        ├── __init__.py
        ├── config.py
        ├── httpHandler.py
        ├── pipeline.py
        ├── ratelimit.py
        └── utils.py
```



## API endpoints

* `GET /` – simple hello JSON for smoke testing. 
* `GET /health` – liveness probe. 
* `GET /version` – app version from env. 
* `POST /ask` – runs the RAG pipeline and returns `{ answer, sources }`. 

### Request shape

```http
POST /ask
Content-Type: application/json

{
  "question": "How important is CPSC 131? Is it a prerequisite for other courses?",
  "topK": 5   // optional, 1..10
}
```

### Success response (200)

```json
{
  "answer": "CPSC 131 is a foundation course and a prerequisite for several follow-on modules including Compilers and Database Systems.",
  "sources": [
    {
      "id": "CPSC_323_Mohamadreza_Ahmadnia_42",
      "score": 0.688,
      "course": "CPSC 323",
      "courseName": "Compilers and Languages",
      "instructor": "Mohamadreza Ahmadnia"
    }
  ]
}
```

(Same shape as produced by `pipeline.build_context`.) 

### Error response (examples)

* Bad body:

```json
{ "error": "Body must be JSON object" }
```



* Missing or invalid fields:

```json
{ "error": "Missing 'question'" }
{ "error": "Invalid 'topK'" }
```



### Rate limiting

* Limit: **3 requests per minute per IP**
* Returns standard headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and on block `Retry-After`.
* CORS exposes these headers to the browser via `Access-Control-Expose-Headers`.  

#### Example 429 response

```json
{
  "error": "Rate limit exceeded",
  "detail": "3 requests per minute allowed for this IP."
}
```



## Local setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```



2. **Create `.dev.vars`** with your Cloudflare credentials and app config:

```dotenv
CLOUDFLARE_ACCOUNT_ID=acc_XXXXXXXXXXXXXXXXXXXXXXXXX
CLOUDFLARE_API_TOKEN=cf_api_token_with_ai_vectorize_permissions

# App vars
ALLOWED_ORIGINS=http://localhost:3000
APP_VERSION=2025.09.02-1
DEBUG=1
CF_EMBED_MODEL=@cf/baai/bge-base-en-v1.5
CF_CHAT_MODEL=@cf/meta/llama-3.1-8b-instruct-fast
CF_TOPK=5
GEN_TEMPERATURE=0.2
GEN_MAX_TOKENS=350
```



3. **Ensure `wrangler.jsonc` has bindings** (AI, Vectorize, and your KV for rate limiting). Example:

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "<name>",
  "main": "src/entry.py",
  "compatibility_date": "<date>",
  "compatibility_flags": ["python_workers"],

  "ai": { "binding": "AI" },
  "vectorize": [{ "binding": "<binding_namesapce>", "index_name": "<index_namespace>" }],

  "env": {
    "staging": {
      "ai": { "binding": "AI" },
      "vectorize": [{ "binding": "<COURSES>", "index_name": "<index_namespace>" }],
      "kv_namespaces": [
        { "binding": "<binding_namesapce>", "id": "<prod-id>", "preview_id": "<preview-id>" }
      ],
      "vars": { "ALLOWED_ORIGINS": "http://localhost:3000", "DEBUG": "1", "APP_VERSION": "<date>" }
    }
  }
}
```

(Envs don’t inherit—repeat bindings under `env.staging`.) 

4. **Build vectors (optional for local testing)**

```bash
python data_loader/vector_builder.py \
  --csv data_loader/courses.csv \
  --out data_loader/vectors_to_upload.ndjson \
  --index csuf-courses \
  --insert \
  --env ./.dev.vars
```



5. **Run dev server**

```bash
npx wrangler dev --env staging --experimental-vectorize-bind-to-prod
# → http://localhost:8787
```

Quick checks:

```bash
curl -s http://localhost:8787/health
curl -s http://localhost:8787/version

# CORS preflight
curl -i -X OPTIONS "http://localhost:8787/ask" \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"

# Ask
curl -i -X POST "http://localhost:8787/ask" \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  --data '{"question":"How imp is this 131 course? is it a prereq for any other course?"}'
```



