import os, requests
from dotenv import load_dotenv

load_dotenv('./.dev.vars')  # or ".env"
ACC = os.getenv("CLOUDFLARE_ACCOUNT_ID")
TOK = os.getenv("CLOUDFLARE_API_TOKEN")
INDEX = "csuf-courses"

# 1) Embed a sample question
qtext = "data structures prerequisites"
r = requests.post(
    f"https://api.cloudflare.com/client/v4/accounts/{ACC}/ai/run/@cf/baai/bge-base-en-v1.5",
    headers={"Authorization": f"Bearer {TOK}"},
    json={"text": [qtext]},
    timeout=60
)
r.raise_for_status()
vec = r.json()["result"]["data"][0]

# 2) Query the index
q = requests.post(
    f"https://api.cloudflare.com/client/v4/accounts/{ACC}/vectorize/v2/indexes/{INDEX}/query",
    headers={"Authorization": f"Bearer {TOK}"},
    json={"vector": vec, "topK": 5, "returnValues": False, "returnMetadata": "all"},
    timeout=60
)
q.raise_for_status()
res = q.json()["result"]["matches"]
for i, m in enumerate(res, 1):
    print(f"{i}. id={m.get('id')}  score={m.get('score'):.4f}")
    md = m.get("metadata", {})
    print("   course:", md.get("Course Name"), "| prof:", md.get("First Last"))
    