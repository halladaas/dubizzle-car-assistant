"""One-time data-prep: turn the provided raw listing sheet into
data/cars.csv, the canonical inventory the rest of the app reads.

The source xlsx has no Price/mileage/body_type columns -- those live
(inconsistently) inside free-text titles/descriptions. We extract them once,
offline, with a single structured (function-calling) Gemini call per row, and
cache the result to CSV. This keeps the runtime app fast (no LLM calls just
to load inventory) and keeps the structured pre-filter step honest: every
value in cars.csv is either taken verbatim from the source or explicitly
grounded in that listing's own text, never invented at query time. Fields the
listing doesn't mention are left null -- the agent must say "not listed"
rather than guess.

Usage:
    uv run python scripts/prepare_dataset.py [--limit N] [--sheet "raw dataset"]

Resumable: rows already present in the output CSV are skipped on re-run, so a
rate-limit failure partway through doesn't lose progress.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.llm import call_with_tool  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_XLSX = DATA_DIR / "Copy_of_sample_cars_dataset.xlsx"
OUTPUT_CSV = DATA_DIR / "cars.csv"

COLUMN_ORDER = [
    "car_id", "make", "model", "trim", "year", "title", "description",
    "photo_url", "price_aed", "mileage_km", "body_type", "exterior_color",
    "transmission", "fuel_type", "condition", "warranty_years", "regional_spec",
]

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_car_attributes",
        "description": (
            "Extract structured attributes explicitly stated in a used-car "
            "listing's title/description. Only fill a field if the text "
            "states it; otherwise return null. Never guess or infer a "
            "number that isn't written in the text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "price_aed": {
                    "type": ["integer", "null"],
                    "description": (
                        "Total asking/cash price in AED, if the text states "
                        "one directly (e.g. 'AED 119,750' or a 'in cash' "
                        "finance-plan line). Do NOT compute this from a "
                        "monthly payment figure -- if only a monthly "
                        "installment is given with no total, return null."
                    ),
                },
                "mileage_km": {
                    "type": ["integer", "null"],
                    "description": "Odometer reading in km if stated.",
                },
                "body_type": {
                    "type": ["string", "null"],
                    "enum": [
                        "suv", "sedan", "hatchback", "coupe", "convertible",
                        "pickup", "van", "wagon", "other", None,
                    ],
                    "description": "Infer from model/description if reasonably obvious, else null.",
                },
                "exterior_color": {"type": ["string", "null"]},
                "transmission": {
                    "type": ["string", "null"],
                    "enum": ["automatic", "manual", None],
                },
                "fuel_type": {
                    "type": ["string", "null"],
                    "enum": ["petrol", "diesel", "hybrid", "electric", None],
                },
                "condition": {
                    "type": ["string", "null"],
                    "enum": ["new", "used", None],
                    "description": "'new' only if explicitly brand new / 0km, else 'used' if stated, else null.",
                },
                "warranty_years": {
                    "type": ["number", "null"],
                    "description": "Warranty length in years if stated.",
                },
                "regional_spec": {
                    "type": ["string", "null"],
                    "description": (
                        "ONLY a regional spec designation such as GCC, US, "
                        "Japanese, Canadian, European, Korean. Never customs/"
                        "paperwork text or any non-English phrase -- if "
                        "unclear or not one of these, return null."
                    ),
                },
            },
            "required": [
                "price_aed", "mileage_km", "body_type", "exterior_color",
                "transmission", "fuel_type", "condition", "warranty_years",
                "regional_spec",
            ],
        },
    },
}

BOILERPLATE_PATTERNS = [
    r"contact us.*", r"office:.*", r"sales:.*", r"tel:.*", r"mobile no.*",
    r"call.*\+?\d[\d\s]{6,}", r"whatsapp.*\+?\d[\d\s]{6,}", r"\+?971[\d\s]{7,}",
    r"instagram[:\-].*", r"facebook[:\-].*", r"linkedin[:\-].*", r"twitter[:\-].*",
    r"pinterest[:\-].*", r"www\.\S+", r"http\S+", r"#\w+",
    r"dd id[:\-]?\s*\S+", r"ref\s*#?\s*\w+",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


def clean_text(raw: str) -> str:
    """Deterministic cleanup: decode HTML entities/tags, strip dealer
    boilerplate (phone numbers, socials, hashtags, refs). Used for the
    embedding text and for a tidier display description."""
    text = html.unescape(str(raw))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    kept = [ln for ln in lines if ln and not BOILERPLATE_RE.search(ln)]
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned or text.strip()


def build_messages(title: str, description: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You extract structured facts from used-car listing text. "
                "Only report a value if it is explicitly stated in the "
                "text. Return null for anything not stated -- never guess."
            ),
        },
        {
            "role": "user",
            "content": f"TITLE: {title}\n\nDESCRIPTION:\n{description[:3000]}",
        },
    ]


BODY_TYPES = {"suv", "sedan", "hatchback", "coupe", "convertible", "pickup", "van", "wagon", "other"}
TRANSMISSIONS = {"automatic", "manual"}
FUEL_TYPES = {"petrol", "diesel", "hybrid", "electric"}
CONDITIONS = {"new", "used"}
CONDITION_ALIASES = {"brand new": "new", "0km": "new", "pre-owned": "used", "pre owned": "used"}


def _normalize_enum(value, allowed: set[str], aliases: dict[str, str] | None = None):
    """Models don't reliably respect JSON-schema enum casing -- normalize
    case/whitespace here rather than trust the raw string for filtering."""
    if value is None or (isinstance(value, float)):
        return None
    v = str(value).strip().lower()
    if aliases and v in aliases:
        v = aliases[v]
    return v if v in allowed else None


def normalize_attrs(attrs: dict) -> dict:
    attrs = dict(attrs)
    attrs["body_type"] = _normalize_enum(attrs.get("body_type"), BODY_TYPES)
    attrs["transmission"] = _normalize_enum(attrs.get("transmission"), TRANSMISSIONS)
    attrs["fuel_type"] = _normalize_enum(attrs.get("fuel_type"), FUEL_TYPES)
    attrs["condition"] = _normalize_enum(attrs.get("condition"), CONDITIONS, CONDITION_ALIASES)
    spec = attrs.get("regional_spec")
    attrs["regional_spec"] = spec.strip() if isinstance(spec, str) and spec.strip() else None
    color = attrs.get("exterior_color")
    attrs["exterior_color"] = color.strip() if isinstance(color, str) and color.strip() else None
    return attrs


def enrich_row(title: str, cleaned_desc: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        result = call_with_tool(
            messages=build_messages(title, cleaned_desc),
            tool_schema=EXTRACT_TOOL,
            component="data_prep_enrichment",
        )
        if result is not None:
            return normalize_attrs(result)
        time.sleep(2 * (attempt + 1))
    return {
        "price_aed": None, "mileage_km": None, "body_type": None,
        "exterior_color": None, "transmission": None, "fuel_type": None,
        "condition": None, "warranty_years": None, "regional_spec": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sheet", default="raw dataset")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between LLM calls (free-tier RPM guard)")
    args = parser.parse_args()

    df = pd.read_excel(SOURCE_XLSX, sheet_name=args.sheet)
    df = df.reset_index(drop=True)
    df["car_id"] = df.index

    if args.limit:
        df = df.head(args.limit)

    existing = pd.DataFrame()
    done_ids: set[int] = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing["car_id"].tolist())
        print(f"Resuming: {len(done_ids)} rows already in {OUTPUT_CSV.name}")

    rows_out = []
    for i, row in df.iterrows():
        car_id = int(row["car_id"])
        if car_id in done_ids:
            continue

        title_clean = clean_text(row["title"])
        desc_clean = clean_text(row["description"])
        attrs = enrich_row(row["title"], desc_clean)

        rows_out.append(
            {
                "car_id": car_id,
                "make": row["make"],
                "model": row["model"],
                "trim": row["trim"],
                "year": int(row["year"]),
                "title": title_clean,
                "description": desc_clean,
                "photo_url": row["photo_url"],
                **attrs,
            }
        )
        print(f"[{i + 1}/{len(df)}] car_id={car_id} {row['make']} {row['model']} "
              f"price={attrs.get('price_aed')} body={attrs.get('body_type')}")

        # Flush every 10 rows so a crash/rate-limit doesn't lose progress.
        if len(rows_out) % 10 == 0:
            existing = _flush(existing, rows_out)
            done_ids.update(r["car_id"] for r in rows_out)
            rows_out = []

        time.sleep(args.sleep)

    existing = _flush(existing, rows_out)
    print(f"Done. Wrote {OUTPUT_CSV}")


def _flush(existing: pd.DataFrame, rows_out: list[dict]) -> pd.DataFrame:
    if not rows_out:
        return existing
    new_df = pd.DataFrame(rows_out)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset="car_id", keep="last").sort_values("car_id")
    combined = combined[COLUMN_ORDER]
    combined.to_csv(OUTPUT_CSV, index=False)
    return combined


if __name__ == "__main__":
    main()
