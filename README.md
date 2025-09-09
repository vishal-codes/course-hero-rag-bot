# CSUF Python RAG Bot

A compact Retrieval-Augmented Generation (RAG) API on **Cloudflare Workers (Python)** that:

* embeds questions with **Workers AI**,
* searches a **Vectorize** index of course data,
* and generates concise answers with a chat model.

It’s designed to power a simple frontend chat with strict CORS, clear JSON errors, and environment-driven config for staging/production.


# Description

**What it does**

* Ingests a `courses.csv`, generates embeddings with Workers AI, and writes **strict NDJSON** lines (`{ id, values, metadata }`) for bulk insert.
* Uploads those vectors into a **Cloudflare Vectorize** index.
* Exposes an API:

  * `GET /health` – liveness
  * `GET /version` – app version (from env)
  * `POST /ask` – RAG pipeline → `{ "answer": "..." }`




# Local Setup

## 1) Clone & install

```bash
git clone https://github.com/vishal-codes/course-hero-rag-bot
cd course-hero-rag-bot

pip install -r requirements.txt  

```

## 2) Create environment file

Create `.dev.vars` (or `.env`) with your Cloudflare account credentials:

```dotenv
CLOUDFLARE_ACCOUNT_ID=acc_XXXXXXXXXXXXXXXXXXXXXXXXX
CLOUDFLARE_API_TOKEN=cf_api_token_with_ai_vectorize_permissions

# Optional app vars (entry.py will read these)
ALLOWED_ORIGINS=http://localhost:3000,your frontend URL
APP_VERSION=2025.09.02-1
DEBUG=1
CF_EMBED_MODEL=@cf/baai/bge-base-en-v1.5
CF_CHAT_MODEL=@cf/meta/llama-3.1-8b-instruct-fast
CF_TOPK=5
GEN_TEMPERATURE=0.2
GEN_MAX_TOKENS=350
```
Create a `wrangler.jsonc` file in root directory

> Important: **envs do not inherit**. Repeat `ai`, `vectorize`, and `vars` under `env.staging`.

```jsonc
{
  "$schema": "node_modules/wrangler/config-schema.json",
  "name": "python-rag-bot",
  "main": "src/entry.py",
  "compatibility_date": "2025-09-02",
  "compatibility_flags": ["python_workers"],
  "observability": { "enabled": true },

  // PROD (top-level)
  "ai": { "binding": "AI" },
  "vectorize": [{ "binding": "COURSES", "index_name": "csuf-courses" }],
  "vars": {
    "ALLOWED_ORIGINS": "your-frontend-origin",
    "DEBUG": "0",
    "APP_VERSION": "2025.09.02-1",
    "CF_EMBED_MODEL": "@cf/baai/bge-base-en-v1.5",
    "CF_CHAT_MODEL": "@cf/meta/llama-3.1-8b-instruct-fast",
    "CF_TOPK": "5",
    "GEN_TEMPERATURE": "0.2",
    "GEN_MAX_TOKENS": "350"
  },

  // STAGING
  "env": {
    "staging": {
      "ai": { "binding": "AI" },
      "vectorize": [{ "binding": "COURSES", "index_name": "csuf-courses" }],
      "vars": {
        "ALLOWED_ORIGINS": "http://localhost:3000,your-frontend-origin",
        "DEBUG": "1",
        "APP_VERSION": "2025.09.02-1",
        "CF_EMBED_MODEL": "@cf/baai/bge-base-en-v1.5",
        "CF_CHAT_MODEL": "@cf/meta/llama-3.1-8b-instruct-fast",
        "CF_TOPK": "5",
        "GEN_TEMPERATURE": "0.2",
        "GEN_MAX_TOKENS": "350"
      }
    }
  }
}
```

## 3) Build vectors (NDJSON)

```
data_loader/
  └─ vector_builder.py
```

Build NDJSON (embeds with Workers AI) and optionally upload:

```bash
#  Build + upload to Vectorize in one step
python data_loader/vector_builder.py \
  --csv data_loader/courses.csv \
  --out data_loader/vectors_to_upload.ndjson \
  --index csuf-courses \
  --insert \
  --env ./.dev.vars
```

Verify vectors exist:

```bash
npx wrangler vectorize list-vectors csuf-courses --limit 5
```

## 4) Run the API (staging config)

Start dev server using **staging** environment and bind Vectorize to prod:

```bash
npx wrangler dev --env staging --experimental-vectorize-bind-to-prod
# → Ready on http://localhost:8787
```

Quick checks (in another terminal):

```bash
curl -s http://localhost:8787/health
curl -s http://localhost:8787/version
curl -i -X OPTIONS "http://localhost:8787/ask" \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"

curl -i -X POST "http://localhost:8787/ask" \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  --data '{"question":"How imp is this 131 course? is it a prereq for any other course?"}'
```

Logs:

```bash
npx wrangler tail --env staging
```

## 5) Deploy

**Staging**

```bash
npx wrangler deploy --env staging
```

**Production**

```bash
npx wrangler deploy
```



# Architecture

```
+---------------------+        +-------------------+        +-------------------------+
|  Frontend (Vercel)  |  --->  |  Worker (Python)  |  --->  |  Cloudflare Workers AI  |
|  /ask (POST JSON)   |        |  src/entry.py     |        |  (embed + generate)     |
+---------------------+        |  CORS, JSON, RAG  |        +-------------------------+
         ^                     |                    |                   |
         |                     |                    |                   v
         |                     |                    |        +-------------------------+
         |                     +--------------------+------> |  Cloudflare Vectorize   |
         |                             (query vectors)       |  (csuf-courses index)   |
         |                                                   +-------------------------+
         +----------------------------- responses (JSON: { "answer": "..." })
```

* **Workers AI**: runs the embedding model and the chat model.
* **Vectorize**: stores your course vectors and metadata for semantic search.
* **Worker (Python)**: HTTP entrypoint, orchestrates RAG, enforces CORS, formats output.
* **(Optional) AI Gateway**: you can configure caching/analytics for AI calls later without code changes.


# Terminologies (Cloudflare)

* **Workers (Python)**: Serverless runtime at the edge. Your API lives here (`src/entry.py`).
* **Workers AI**: Hosted inference endpoints for embeddings & LLMs (`env.AI.run(model, inputs)`).
* **Vectorize**: Managed vector database. You insert NDJSON vectors and query via bindings (`env.COURSES.query(...)`).
* **Bindings**: How a Worker accesses platform resources:

  * `ai` binding → `env.AI`
  * `vectorize` binding → `env.COURSES`
* **Environments**: `staging` / `production` blocks in `wrangler.jsonc` with separate vars/bindings.
* **Compatibility Date**: Locks runtime features for deterministic behavior.


# Flow of the Codebase

## Ingestion

1. **`data_loader/vector_builder.py`**

   * Loads `courses.csv` → cleans text & metadata.
   * Embeds course “documents” in batches with Workers AI.
   * Writes **strict NDJSON** lines:
     `{"id": "...","values":[...768 floats...],"metadata":{...}}`
   * Optional: `--insert` POSTs NDJSON to Vectorize (`/vectorize/v2/indexes/<index>/insert`).

## Query (API)

2. **`src/entry.py` (Worker)**

   * **CORS**: Handles `OPTIONS` preflight; restricts to `ALLOWED_ORIGINS`.
   * `POST /ask`:

     1. Embed question with Workers AI (`CF_EMBED_MODEL`).
     2. Query Vectorize (`env.COURSES.query`) for top-K matches (returning metadata).
     3. Build a context block from matches.
     4. Generate answer with chat model (`CF_CHAT_MODEL`), **strip bracketed refs**, return:

        ```json
        { "answer": "...", "sources": [] }
        ```
   * Errors are returned as JSON with appropriate status codes (400/403/502/500).

## Local CLI (optional)

* `rag_answer.py` & `example_query.py` show the same flow using REST (useful for sanity checks).



# API Shape (for frontend)

**Request**

```
POST /ask
Content-Type: application/json

{
  "question": "how imp is this 131 course? is it a prereq for any other course?
?",
  "topK": 5    // optional, 1..10 (default from env)
}
```

**Response**

```json
{
  "answer": "The CPSC 131 course is a fundamental course in the Computer Science program at CSU Fullerton. It is a prerequisite for several courses, including Compilers and Languages, File Structures and Database Systems, and likely others. Its importance lies in providing a solid foundation in programming concepts and principles. As a prerequisite, it is essential for students to have a strong understanding of programming before taking these courses. Without this course, students may struggle to keep up with the material in the subsequent courses.",
  "sources": [
    {
      "id": "CPSC_323_Mohamadreza_Ahmadnia_42",
      "score": 0.688,
      "course": "CPSC 323",
      "courseName": "Compilers and Languages",
      "instructor": "Mohamadreza Ahmadnia"
    },
    {
      "id": "CPSC_323_Shohrat_Geldiyev_55",
      "score": 0.68,
      "course": "CPSC 323",
      "courseName": "Compilers and Languages",
      "instructor": "Shohrat Geldiyev"
    },
    {
      "id": "CPSC_332_Shawn_Wang_54",
      "score": 0.675,
      "course": "CPSC 332",
      "courseName": "File Structures and Database Systems",
      "instructor": "Shawn Wang"
    },
    {
      "id": "CPSC_332_David_Heckathorn_10",
      "score": 0.669,
      "course": "CPSC 332",
      "courseName": "File Structures and Database Systems",
      "instructor": "David Heckathorn"
    },
    {
      "id": "CPSC_323_Song_Choi_58",
      "score": 0.667,
      "course": "CPSC 323",
      "courseName": "Compilers and Languages",
      "instructor": "Song Choi"
    }
  ]
}
```

**Errors**

```json
{ "error": "Missing 'question'" }                   // 400
{ "error": "CORS not allowed" }                     // 403
{ "error": "Embedding failed", "detail": "..." }    // 502
{ "error": "Vector search failed", "detail": "..." }// 502
{ "error": "Answer generation failed", "detail": "..." } // 502
{ "error": "unhandled_exception", "detail": "..." } // 500
```



# Tips & Troubleshooting

* **Vectors not listing?** Ensure NDJSON is strict JSON (no `NaN`, `Infinity`) and Content-Type is `application/x-ndjson`.
* **CORS 403?** Make sure the exact origin (scheme + host + port) is in `ALLOWED_ORIGINS`.
* **Embedding error 5006 / “oneOf at '/' not met”**: Workers AI expects `{ "text": "..." }` (string) or `{ "text": ["...","..."] }` (batch) — ensure you use the correct shape.
* **Dev charges**: Workers AI & Vectorize calls in `wrangler dev` hit real services.

