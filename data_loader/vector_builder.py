"""
Purpose
-------
Build clean NDJSON (strict JSON, no NaN/Infinity) from a CSV, generate embeddings
with Cloudflare Workers AI (@cf/baai/bge-base-en-v1.5), and write one vector per
line for Cloudflare Vectorize bulk insert.

Design
------
- Config: loads env + paths (12‑factor friendly)
- CSVProcessor: loads/cleans rows, builds text, prepares metadata
- CloudflareAIClient: batches embedding calls (<=100/texts per request)
- VectorRecord: dataclass holding id/values/metadata
- NDJSONWriter: serialises records with allow_nan=False
- Pipeline: orchestration + CLI

Usage
-----
python vector_builder.py \
  --csv ./courses.csv \
  --out ./vectors_to_upload.ndjson \
  --index csuf-courses \
  --env ../.dev.vars

Insert after build
python vector_builder.py --csv ./courses.csv --out ./vectors_to_upload.ndjson \
  --index csuf-courses --insert --env ../.dev.vars
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

LOG = logging.getLogger("vectorize_builder")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DEFAULT_MODEL = "@cf/baai/bge-base-en-v1.5"  # 768 dims
MAX_BATCH = 100  # Cloudflare model accepts up to 100 texts in one call
FORBIDDEN_KEY_CHARS = re.compile(r"[.\"$]")
ID_SAFE = re.compile(r"[^a-zA-Z0-9_\-.]")

def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def sanitize_text(text: Any) -> Any:
    """Collapse whitespace + ensure UTF-8; do not alter quotes."""
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text.encode("utf-8", "ignore").decode("utf-8")

def normalize_metadata_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        k2 = FORBIDDEN_KEY_CHARS.sub("_", k)
        if k2.startswith("$"):
            k2 = "_" + k2[1:]
        out[k2 or "_"] = v
    return out

def to_native(v: Any) -> Any:
    """Convert numpy scalars so json.dumps works, keep None for missing."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (isinstance(f, float) and not math.isfinite(f)) else f
    return v

def json_safe(obj: Any) -> Any:
    """Recursively turn NaN/Infinity into None to satisfy strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

def stable_id(row: pd.Series, idx: int) -> str:
    base = f"{row.get('Course','unknown')}_{row.get('First Last','unknown')}_{idx}"
    return ID_SAFE.sub("_", base)


@dataclasses.dataclass
class VectorRecord:
    id: str
    values: List[float]
    metadata: Dict[str, Any]

    def to_json_line(self) -> str:
        payload = {
            "id": self.id,
            "values": [float(x) for x in self.values],
            "metadata": json_safe(self.metadata),
        }
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


class Config:
    def __init__(self, env_path: Optional[str] = None):
        if env_path:
            load_dotenv(env_path)
        # allow also default .env if present
        load_dotenv(override=False)
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        if not self.account_id or not self.api_token:
            raise SystemExit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in your env")

    @property
    def api_base(self) -> str:
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"


class CloudflareAIClient:
    def __init__(self, cfg: Config, model: str = DEFAULT_MODEL, batch_size: int = MAX_BATCH, sleep_s: float = 0.0):
        self.cfg = cfg
        self.model = model
        self.batch_size = max(1, min(batch_size, MAX_BATCH))
        self.sleep_s = max(0.0, sleep_s)
        self.url = f"{cfg.api_base}/ai/run/{model}"
        self.headers = {"Authorization": f"Bearer {cfg.api_token}"}

    def embed(self, texts: List[str]) -> List[List[float]]:
        LOG.info("Embedding %d texts in batches of %d", len(texts), self.batch_size)
        out: List[List[float]] = []
        for chunk in chunked(texts, self.batch_size):
            resp = requests.post(self.url, headers=self.headers, json={"text": list(chunk)}, timeout=120)
            try:
                resp.raise_for_status()
                data = resp.json()
                vecs = data["result"]["data"]
                if len(vecs) != len(chunk):
                    raise RuntimeError(f"Embedding count mismatch: expected {len(chunk)} got {len(vecs)}")
                out.extend(vecs)
            finally:
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        if not out:
            raise RuntimeError("No embeddings returned")
        LOG.info("Got embeddings; dimension=%d", len(out[0]))
        return out


class CSVProcessor:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise SystemExit(f"CSV not found: {self.csv_path}")

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        # Drop unnamed columns often created by CSV editors
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        # Clean a few expected string columns if present
        for col in [
            "First Last",
            "Course",
            "Course Name",
            "Description",
            "Prerequisite",
            "Corequisite",
            "Graduate Eligibility",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("").map(sanitize_text)
        return df

    @staticmethod
    def build_document(row: pd.Series) -> str:
        g = row.get
        parts: List[str] = []
        if g("Course Name") or g("Course"):
            parts.append(f"Course: {g('Course Name','').strip()} ({g('Course','').strip()}).")
        if g("First Last"):
            parts.append(f"Taught by Professor {g('First Last','').strip()}.")
        if g("Description"):
            parts.append(f"Description: {g('Description','').strip()}.")
        if g("Prerequisite"):
            parts.append(f"Prerequisites: {g('Prerequisite','').strip()}.")
        if pd.notna(g("Avg GPA", np.nan)):
            parts.append(f"Average GPA: {g('Avg GPA')}.")
        if pd.notna(g("Difficulty", np.nan)):
            parts.append(f"Difficulty: {g('Difficulty')}/5.")
        return " ".join(p for p in parts if p and p != " ")

    @staticmethod
    def prepare_metadata(row: pd.Series) -> Dict[str, Any]:
        # Convert pandas NaN -> None, numpy scalars -> native, and normalise keys
        d = row.where(pd.notna(row), None).to_dict()
        d = {str(k): to_native(v) for k, v in d.items()}
        d = normalize_metadata_keys(d)
        return d


class NDJSONWriter:
    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: Iterable[VectorRecord]) -> int:
        count = 0
        with self.out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_json_line() + "\n")
                count += 1
        LOG.info("Wrote %d vectors to %s", count, self.out_path)
        return count


class Pipeline:
    def __init__(self, cfg: Config, csv_path: Path, out_path: Path, index: str, model: str = DEFAULT_MODEL, batch_size: int = MAX_BATCH, sleep_s: float = 0.0):
        self.cfg = cfg
        self.csv = CSVProcessor(csv_path)
        self.out = NDJSONWriter(out_path)
        self.idx_name = index
        self.client = CloudflareAIClient(cfg, model=model, batch_size=batch_size, sleep_s=sleep_s)

    def build(self) -> Tuple[int, int]:
        df = self.csv.load()
        LOG.info("Loaded %d rows from %s", len(df), self.csv.csv_path)
        documents = [self.csv.build_document(row) for _, row in df.iterrows()]
        embeddings = self.client.embed(documents)
        dim = len(embeddings[0])
        LOG.info("Embedding dimension detected: %d", dim)

        records: List[VectorRecord] = []
        for (df_idx, row), emb in zip(df.iterrows(), embeddings):
            rec = VectorRecord(
                id=stable_id(row, df_idx),
                values=[float(x) for x in emb],
                metadata=self.csv.prepare_metadata(row),
            )
            records.append(rec)
        written = self.out.write(records)
        return written, dim

    def insert(self) -> str:
        # Insert the out_path NDJSON as raw body with application/x-ndjson
        url = f"{self.cfg.api_base}/vectorize/v2/indexes/{self.idx_name}/insert"
        LOG.info("Inserting NDJSON into index '%s' via %s", self.idx_name, url)
        data = Path(self.out.out_path).read_bytes()
        headers = {
            "Authorization": f"Bearer {self.cfg.api_token}",
            "Content-Type": "application/x-ndjson",
        }
        r = requests.post(url, headers=headers, data=data, timeout=300)
        r.raise_for_status()
        resp = r.json()
        mut = (resp.get("result") or {}).get("mutationId")
        LOG.info("Insert accepted, mutationId=%s", mut)
        return mut or ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NDJSON vectors for Cloudflare Vectorize")
    p.add_argument("--csv", required=True, type=Path, help="Path to input CSV")
    p.add_argument("--out", required=True, type=Path, help="Path to output NDJSON")
    p.add_argument("--index", required=True, help="Vectorize index name (for insert)")
    p.add_argument("--env", default=None, help="Path to .env/.dev.vars with CF creds")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model to use")
    p.add_argument("--batch-size", type=int, default=MAX_BATCH, help="Batch size (<=100)")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between batches")
    p.add_argument("--insert", action="store_true", help="Insert NDJSON after building")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = Config(env_path=args.env)
    pipe = Pipeline(
        cfg=cfg,
        csv_path=args.csv,
        out_path=args.out,
        index=args.index,
        model=args.model,
        batch_size=args.batch_size,
        sleep_s=args.sleep,
    )
    written, dim = pipe.build()
    LOG.info("Build complete: %d records, dimension=%d", written, dim)

    if args.insert:
        mut = pipe.insert()
        if mut:
            LOG.info("Insert queued with mutationId=%s", mut)
        else:
            LOG.warning("Insert response had no mutationId; inspect server response")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        LOG.error("HTTPError %s: %s", e.response.status_code if e.response else "", getattr(e.response, "text", ""))
        sys.exit(2)
    except Exception as e:
        LOG.exception("Fatal error: %s", e)
        sys.exit(1)
