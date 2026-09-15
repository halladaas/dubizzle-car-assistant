"""Hybrid retrieval: structured pandas pre-filter, then FAISS semantic
search restricted to the surviving rows.

Why this order: a pandas boolean mask on price/year/make is free, instant,
and can never hallucinate a nonexistent car or ignore a hard price ceiling
-- both real failure modes of vector-search-only retrieval. FAISS then
ranks the *filtered* subset by semantic similarity to the query, so "rugged
for desert driving" can match on meaning within whatever make/price/year
constraints were already established. See data/cars.csv (built by
scripts/prepare_dataset.py) for where the structured columns come from --
they're either verbatim source data (make/model/year) or grounded in that
listing's own text (price/body_type/...), never invented.

At ~100 rows, IndexFlatIP (exact, brute-force) is sub-millisecond with 100%
recall, so there's no reason to reach for an approximate index.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import litellm
import numpy as np
import pandas as pd

from core.llm import call_with_tool, with_rate_limit_retry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARS_CSV = DATA_DIR / "cars.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBED_IDS_PATH = DATA_DIR / "embedding_ids.npy"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini/gemini-embedding-001")

BODY_TYPES = ["suv", "sedan", "hatchback", "coupe", "convertible", "pickup", "van", "wagon", "other"]

# Order to drop filter fields in when a filter combination yields no rows.
# Price ceiling goes first (the most common reason for zero results is a
# strict budget the majority-null price column can't confirm), then floor,
# then year range, then body_type; make/model/keywords are kept as long as
# possible since dropping them changes what the user actually asked for.
RELAX_ORDER = ["max_price", "min_price", "min_year", "max_year", "body_type", "model"]


# --------------------------------------------------------------------------
# Inventory loading
# --------------------------------------------------------------------------

_inventory_cache: pd.DataFrame | None = None


def load_inventory(force_reload: bool = False) -> pd.DataFrame:
    global _inventory_cache
    if _inventory_cache is None or force_reload:
        df = pd.read_csv(CARS_CSV)
        for col in ("price_aed", "mileage_km", "year", "warranty_years"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _inventory_cache = df
    return _inventory_cache


def get_car(car_id: int) -> dict | None:
    df = load_inventory()
    row = df[df["car_id"] == car_id]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


# --------------------------------------------------------------------------
# Structured filter extraction (small LLM call, strict schema)
# --------------------------------------------------------------------------

FILTER_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_filters",
        "description": (
            "Extract structured car-search filters from the user's latest "
            "message. Only set a field if THIS message states or changes "
            "it; leave it null if this message doesn't mention it (the "
            "caller merges with filters carried over from earlier turns)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "make": {"type": ["string", "null"], "description": "Car brand, e.g. 'toyota'."},
                "model": {"type": ["string", "null"]},
                "min_price": {"type": ["integer", "null"], "description": "Minimum price in AED."},
                "max_price": {"type": ["integer", "null"], "description": "Maximum price in AED."},
                "min_year": {"type": ["integer", "null"]},
                "max_year": {"type": ["integer", "null"]},
                "body_type": {"type": ["string", "null"], "enum": BODY_TYPES + [None]},
                "keywords": {
                    "type": ["string", "null"],
                    "description": (
                        "Free-text descriptive terms not captured by the "
                        "structured fields above, e.g. 'rugged for desert "
                        "driving', 'white', 'low mileage', 'family car'."
                    ),
                },
                "reset_filters": {
                    "type": "boolean",
                    "description": "True if the user is clearly starting a new, unrelated search rather than refining the current one.",
                },
            },
            "required": [
                "make", "model", "min_price", "max_price", "min_year",
                "max_year", "body_type", "keywords", "reset_filters",
            ],
        },
    },
}


def extract_filters(query: str, prior_filters: dict | None = None) -> dict:
    """Pull structured filters from the user's message and merge them with
    filters already established this session. Returns prior_filters (plus a
    keyword fallback) untouched if the LLM call fails, so callers can still
    do a plain keyword search instead of erroring."""
    prior = {k: v for k, v in (prior_filters or {}).items() if not k.startswith("_")}
    messages = [
        {"role": "system", "content": "You extract car search filters from a chat message. Be conservative: only set a field the message actually states."},
        {"role": "user", "content": query},
    ]
    result = call_with_tool(messages, FILTER_TOOL, component="filter_extraction")

    if result is None:
        merged = dict(prior)
        merged["keywords"] = query
        merged["_extraction_failed"] = True
        return merged

    merged = {} if result.pop("reset_filters", False) else dict(prior)
    for k, v in result.items():
        if v is not None:
            merged[k] = v
    merged["_extraction_failed"] = False
    return merged


# --------------------------------------------------------------------------
# Structured pandas pre-filter with progressive relaxation
# --------------------------------------------------------------------------

@dataclass
class FilterResult:
    df: pd.DataFrame
    applied_filters: dict
    dropped_fields: list[str] = field(default_factory=list)
    used_full_corpus_fallback: bool = False


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if filters.get("make"):
        mask &= df["make"].str.contains(str(filters["make"]), case=False, na=False)
    if filters.get("model"):
        mask &= df["model"].str.contains(str(filters["model"]), case=False, na=False)
    if filters.get("min_price") is not None:
        mask &= df["price_aed"].notna() & (df["price_aed"] >= filters["min_price"])
    if filters.get("max_price") is not None:
        mask &= df["price_aed"].notna() & (df["price_aed"] <= filters["max_price"])
    if filters.get("min_year") is not None:
        mask &= df["year"] >= filters["min_year"]
    if filters.get("max_year") is not None:
        mask &= df["year"] <= filters["max_year"]
    if filters.get("body_type"):
        mask &= df["body_type"].str.lower() == str(filters["body_type"]).lower()
    return df[mask]


def filter_with_relaxation(df: pd.DataFrame, filters: dict, min_rows: int = 1) -> FilterResult:
    working = {k: v for k, v in filters.items() if not k.startswith("_")}
    result = apply_filters(df, working)
    dropped: list[str] = []

    for field_name in RELAX_ORDER:
        if len(result) >= min_rows:
            break
        if working.get(field_name) is not None:
            working[field_name] = None
            dropped.append(field_name)
            result = apply_filters(df, working)

    used_fallback = False
    if len(result) == 0:
        result = df
        used_fallback = True

    return FilterResult(df=result, applied_filters=working, dropped_fields=dropped, used_full_corpus_fallback=used_fallback)


# --------------------------------------------------------------------------
# FAISS semantic search
# --------------------------------------------------------------------------

def _row_to_embedding_text(row: pd.Series) -> str:
    parts = [
        str(row.get("year", "")), str(row.get("make", "")), str(row.get("model", "")),
        str(row.get("trim", "")), str(row.get("body_type") or ""), str(row.get("exterior_color") or ""),
        str(row.get("description", ""))[:1500],
    ]
    return " ".join(p for p in parts if p)


def _embed_texts(texts: list[str]) -> np.ndarray:
    """The free-tier embedding quota is per-minute and batch calls count
    each input individually, so bursts of >~20 texts can trip it even
    within a single build_index run -- with_rate_limit_retry handles that."""
    response = with_rate_limit_retry(litellm.embedding, model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item["embedding"] for item in response["data"]], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(force: bool = False) -> None:
    """Embed every car's description once and cache to disk. Only re-runs
    when the cache is missing/forced -- never on the request path."""
    if FAISS_INDEX_PATH.exists() and EMBEDDINGS_PATH.exists() and not force:
        return
    df = load_inventory(force_reload=True)
    texts = [_row_to_embedding_text(row) for _, row in df.iterrows()]

    # Small batches with a pause between them: the free-tier quota is
    # per-minute and counts each item in a batch individually, so this
    # avoids tripping it even though it means more round trips.
    vectors_list = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        vectors_list.append(_embed_texts(texts[i:i + batch_size]))
        time.sleep(1)
    vectors = np.vstack(vectors_list)

    ids = df["car_id"].to_numpy().astype("int64")
    np.save(EMBEDDINGS_PATH, vectors)
    np.save(EMBED_IDS_PATH, ids)

    # IndexIDMap2 (not IndexIDMap) -- it maintains a reverse id->vector map,
    # which index.reconstruct(id) in semantic_search() relies on.
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


_index_cache: faiss.Index | None = None
_embed_ids_cache: np.ndarray | None = None


def _get_index() -> faiss.Index:
    global _index_cache, _embed_ids_cache
    if _index_cache is None:
        build_index()
        _index_cache = faiss.read_index(str(FAISS_INDEX_PATH))
        _embed_ids_cache = np.load(EMBED_IDS_PATH)
    return _index_cache


def semantic_search(query: str, candidate_ids: list[int] | None, top_k: int = 5) -> list[tuple[int, float]]:
    """Search within candidate_ids only (the structured-filter survivors),
    or the whole index when candidate_ids is None (full-corpus fallback)."""
    index = _get_index()
    q_vec = _embed_texts([query])

    if candidate_ids is None:
        k = min(top_k, index.ntotal)
        if k == 0:
            return []
        scores, ids = index.search(q_vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    valid_ids = set(_embed_ids_cache.tolist())
    id_arr = np.array([cid for cid in candidate_ids if cid in valid_ids], dtype="int64")
    if id_arr.size == 0:
        return []

    # Reconstruct just the filtered subset's vectors and search a small
    # scratch index -- avoids relying on FAISS ID-selector search params,
    # and at <=100 vectors this is effectively free.
    vectors = np.vstack([index.reconstruct(int(cid)) for cid in id_arr])
    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)
    k = min(top_k, sub_index.ntotal)
    scores, local_idx = sub_index.search(q_vec, k)
    result_ids = id_arr[local_idx[0]]
    return [(int(i), float(s)) for i, s in zip(result_ids, scores[0])]


# --------------------------------------------------------------------------
# Combined hybrid search
# --------------------------------------------------------------------------

def hybrid_search(query: str, filters: dict, top_k: int = 5) -> dict:
    df = load_inventory()
    filter_result = filter_with_relaxation(df, filters)

    candidate_ids = None if filter_result.used_full_corpus_fallback else filter_result.df["car_id"].tolist()
    hits = semantic_search(query, candidate_ids, top_k=top_k)

    scores = dict(hits)
    result_df = df[df["car_id"].isin(scores.keys())].copy()
    result_df["relevance_score"] = result_df["car_id"].map(scores)
    result_df = result_df.sort_values("relevance_score", ascending=False).reset_index(drop=True)

    return {
        "results": result_df,
        "applied_filters": filter_result.applied_filters,
        "dropped_fields": filter_result.dropped_fields,
        "used_full_corpus_fallback": filter_result.used_full_corpus_fallback,
        "extraction_failed": bool(filters.get("_extraction_failed")),
    }
