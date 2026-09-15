"""Retrieval + grounding evaluation harness.

Run: uv run python eval/run_eval.py

Three separate measurements, deliberately kept apart because they need
different kinds of ground truth:

1. Precision@k / Recall@k / MRR for structured + multi-turn queries, run
   against the pure retrieval function (query in, car_ids out) -- fast,
   deterministic, no response generation involved.
2. A relaxed/fallback check for no-match queries: did the system correctly
   broaden the search rather than silently returning nothing (or crashing)?
3. LLM-as-judge (1-5) for fuzzy semantic queries, since there's no single
   "correct" car_id to compute precision/recall against.

Grounding is checked separately, against a handful of real generated
responses (agent.handle_message), by regex-scanning the reply for any
inventory make+model mention that falls outside the cars actually shown
that turn -- a deterministic hallucination check that doesn't need an LLM
judge.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import agent, memory, retrieval  # noqa: E402
from core.llm import SMALL_MODEL, call_with_tool  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


# --------------------------------------------------------------------------
# Pure retrieval (no response generation)
# --------------------------------------------------------------------------

def run_retrieval(query: str, prior_turns: list[str] | None = None, top_k: int = 5) -> dict:
    filters: dict = {}
    for turn in prior_turns or []:
        filters = retrieval.extract_filters(turn, filters)
    filters = retrieval.extract_filters(query, filters)
    search = retrieval.hybrid_search(query, filters, top_k=top_k)
    return {
        "car_ids": search["results"]["car_id"].tolist(),
        "used_full_corpus_fallback": search["used_full_corpus_fallback"],
        "dropped_fields": search["dropped_fields"],
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def precision_at_k(retrieved: list[int], expected: list[int], k: int) -> float | None:
    if not expected:
        return None
    top = retrieved[:k]
    return (sum(1 for c in top if c in expected) / len(top)) if top else 0.0


def recall_at_k(retrieved: list[int], expected: list[int], k: int) -> float | None:
    if not expected:
        return None
    top = retrieved[:k]
    return sum(1 for c in expected if c in top) / len(expected)


def mrr(retrieved: list[int], expected: list[int]) -> float | None:
    if not expected:
        return None
    for rank, car_id in enumerate(retrieved, start=1):
        if car_id in expected:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------
# Grounding check (deterministic, no LLM judge)
# --------------------------------------------------------------------------

_make_model_pairs_cache: list[tuple[str, str, int]] | None = None


def _known_make_model_pairs() -> list[tuple[str, str, int]]:
    global _make_model_pairs_cache
    if _make_model_pairs_cache is None:
        df = retrieval.load_inventory()
        _make_model_pairs_cache = list(
            zip(df["make"].str.lower(), df["model"].str.lower(), df["car_id"])
        )
    return _make_model_pairs_cache


def grounding_check(reply: str, allowed_car_ids: list[int]) -> dict:
    """Flag any inventory car whose make+model is mentioned in the reply but
    wasn't among the cars actually shown this turn -- a real, in-corpus car
    being described out of context is exactly the hallucination pattern the
    structured pre-filter is meant to prevent.

    Mentions of allowed cars are masked out of the text first (longest model
    name first) before checking for other cars. Without this, a shorter
    model name that happens to be a substring of an allowed car's longer
    model name -- e.g. "range rover" inside "range rover evoque" -- would
    falsely flag as a hallucinated mention of a *different* Range Rover.
    """
    allowed_set = set(allowed_car_ids)
    pairs = _known_make_model_pairs()
    allowed_pairs = [p for p in pairs if p[2] in allowed_set]
    other_pairs = [p for p in pairs if p[2] not in allowed_set]

    masked = reply.lower()
    mentioned, violations = [], []

    for pair_group, is_violation in ((allowed_pairs, False), (other_pairs, True)):
        for make, model, car_id in sorted(pair_group, key=lambda p: len(p[1]), reverse=True):
            if make in masked and model in masked:
                mentioned.append(int(car_id))
                if is_violation:
                    violations.append(int(car_id))
                masked = masked.replace(model, " ")

    return {"mentioned_car_ids": mentioned, "violations": violations, "grounded": len(violations) == 0}


# --------------------------------------------------------------------------
# LLM-as-judge for fuzzy/semantic queries
# --------------------------------------------------------------------------

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_relevance",
        "description": "Score how relevant a set of retrieved cars is to a fuzzy/semantic search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "description": "1 (irrelevant) to 5 (excellent match)"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
        },
    },
}


def llm_judge(query: str, cars: list[dict]) -> dict:
    cars_str = "\n".join(
        f"- {c.get('year')} {c.get('make')} {c.get('model')}: {str(c.get('description') or '')[:200]}"
        for c in cars
    ) or "(no results)"
    messages = [
        {"role": "system", "content": "Score 1-5 how well these retrieved cars match the user's query. Be strict -- 5 is reserved for an excellent, clearly on-topic match."},
        {"role": "user", "content": f"QUERY: {query}\n\nRETRIEVED CARS:\n{cars_str}"},
    ]
    result = call_with_tool(messages, JUDGE_TOOL, component="eval_llm_judge", model=SMALL_MODEL)
    return result or {"score": None, "reason": "judge call failed"}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    results = []
    run_id = uuid.uuid4().hex[:8]  # fresh session per eval run -- otherwise a case's
    # conversation history (and session_state) would leak across separate
    # `uv run eval/run_eval.py` invocations that reuse the same case id.

    for case in golden_set:
        case_type = case["type"]
        prior_turns = case.get("prior_turns", [])
        query = case["query"]
        expected = case.get("expected_car_ids")

        retrieval_result = run_retrieval(query, prior_turns, top_k=5)
        retrieved_ids = retrieval_result["car_ids"]
        row = {"id": case["id"], "type": case_type, "query": query, "retrieved_car_ids": retrieved_ids}

        if case_type in ("structured", "multiturn"):
            row["precision_at_5"] = precision_at_k(retrieved_ids, expected, 5)
            row["recall_at_5"] = recall_at_k(retrieved_ids, expected, 5)
            row["mrr"] = mrr(retrieved_ids, expected)

        elif case_type == "no_match":
            row["relaxed_or_fallback"] = bool(retrieval_result["dropped_fields"]) or retrieval_result["used_full_corpus_fallback"]

        elif case_type == "semantic":
            cars = [retrieval.get_car(cid) for cid in retrieved_ids]
            judge = llm_judge(query, [c for c in cars if c])
            row["llm_judge_score"] = judge.get("score")
            row["llm_judge_reason"] = judge.get("reason")

        if case.get("check_grounding"):
            # Both session_id AND user_id are scoped per-case-per-run: reusing
            # one synthetic user_id across cases would let one case's saved
            # preferences leak into another's long-term-memory context via
            # build_user_context -- correct behavior for real returning
            # users, but it contaminates otherwise-independent eval cases.
            session_id = f"eval_{case['id']}_{run_id}"
            user_id = f"eval_user_{case['id']}_{run_id}"
            for turn in prior_turns:
                agent.handle_message(session_id, user_id, turn)
            # A grounded reply may correctly reference cars shown earlier in
            # the session (that's the point of short-term memory) without
            # re-listing them as this turn's cars_shown -- e.g. a message
            # classified as lead_info still legitimately says "the Evoque we
            # just looked at fits your budget." Allow both.
            previously_shown_ids = memory.get_session_state(session_id)["last_shown_car_ids"]
            response = agent.handle_message(session_id, user_id, query)
            allowed_ids = list(set(previously_shown_ids) | {c["car_id"] for c in response["cars_shown"]})
            row["grounding"] = grounding_check(response["reply"], allowed_ids)

        results.append(row)
        print(f"[{case['id']:>4}] {case_type:11s} {query[:55]:55s} -> {retrieved_ids}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    _print_summary(results)
    print(f"\nFull results written to {RESULTS_PATH}")


def _print_summary(results: list[dict]) -> None:
    structured = [r for r in results if r["type"] in ("structured", "multiturn")]
    semantic = [r for r in results if r["type"] == "semantic"]
    no_match = [r for r in results if r["type"] == "no_match"]
    grounded = [r["grounding"]["grounded"] for r in results if "grounding" in r]

    print("\n=== SUMMARY ===")
    if structured:
        vals = [(r["precision_at_5"], r["recall_at_5"], r["mrr"]) for r in structured if r["precision_at_5"] is not None]
        if vals:
            avg_p = sum(v[0] for v in vals) / len(vals)
            avg_r = sum(v[1] for v in vals) / len(vals)
            avg_mrr = sum(v[2] for v in vals) / len(vals)
            print(f"Structured/multi-turn (n={len(structured)}): P@5={avg_p:.2f}  R@5={avg_r:.2f}  MRR={avg_mrr:.2f}")
    if semantic:
        scored = [r["llm_judge_score"] for r in semantic if r.get("llm_judge_score") is not None]
        if scored:
            print(f"Semantic (n={len(semantic)}): avg LLM-judge relevance = {sum(scored)/len(scored):.2f}/5")
    if no_match:
        ok = sum(1 for r in no_match if r["relaxed_or_fallback"])
        print(f"No-match fallback (n={len(no_match)}): {ok}/{len(no_match)} correctly relaxed/broadened search")
    if grounded:
        print(f"Grounding check (n={len(grounded)}): {sum(grounded)}/{len(grounded)} responses fully grounded in shown cars")


if __name__ == "__main__":
    main()
