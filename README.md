# dubizzle cars AI assistant

A FastAPI backend + Streamlit client that lets a user explore a ~100-listing used-car
inventory, ask follow-up questions, book test-drive slots, get qualified as a lead, and be
recognized across sessions — built for the dubizzle ML Intern take-home.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
cd project
uv sync

cp .env.example .env
# put your Google AI Studio key in .env:
#   GEMINI_API_KEY=your_key_here
```

**First-time only** — the source spreadsheet has no Price/mileage/body-type columns (see
[Design decisions](#design-decisions) below for why), so a one-time enrichment pass builds
the inventory the app actually reads:

```bash
uv run python scripts/prepare_dataset.py   # ~6 min for 100 rows on the free tier, resumable
```

This writes `data/cars.csv`. The FAISS index (`data/faiss.index`, `data/embeddings.npy`) is
built automatically on first run of the backend and cached to disk after that.

## Run

```bash
# terminal 1 — backend
uv run uvicorn main:app --reload --port 8000

# terminal 2 — client
uv run streamlit run app.py
```

Open the Streamlit URL it prints, enter any name/ID to start (use the same one again in a
new browser session to test cross-session recall), and chat.

Run tests: `uv run pytest tests/ -q` (24 tests, no network calls — pure filter/memory/booking
logic against fixtures). Run the retrieval eval: `uv run python eval/run_eval.py`.

## Why these choices

**Streamlit over a notebook.** The brief allows either; Streamlit gives a real chat
interface (message history, image cards for listings, a sidebar for session info) that
exercises the backend the way an actual user would, which matters more here than
notebook-native inspectability — and it still stays a thin client: every piece of state
(memory, retrieval, bookings, leads) lives server-side in `core/`, so the backend is fully
usable and testable without the UI at all.

**An explicit control loop over an agent framework (LangChain/LangGraph/CrewAI).**
`core/agent.py` is: classify intent → route to one of a handful of hand-written tool
functions in `core/tools.py` → synthesize a response. No hidden retry loops, no framework
abstraction between a bug and the code causing it. Every step is a plain Python function you
can set a breakpoint in or unit-test in isolation (see `tests/`), and every LLM call is
logged to `data/traces.db` with component, latency, and token counts — that's the
traceability story: nothing happens that isn't a visible, loggable function call.

**Hybrid pandas + FAISS retrieval over pure vector search.** `core/retrieval.py` filters the
inventory DataFrame on structured fields (make/price/year/body_type) *first* — free, instant,
and it can never hallucinate a nonexistent car or ignore a hard price ceiling, which
vector-search-alone is prone to. FAISS (`IndexFlatIP`, exact/brute-force — at ~100 rows there's
no reason to reach for an approximate index) then ranks *within* the filtered survivors by
semantic similarity, so "something rugged for desert driving" still works within whatever
make/price/year the user already stated. If a filter combination returns nothing, fields are
relaxed one at a time (price ceiling first) before falling back to full-corpus semantic search,
and the agent tells the user it broadened the search rather than silently returning empty or
irrelevant results.

**SQLite over an external memory service.** `core/memory.py` covers both memory tiers with
plain tables: short-term is just the last N messages for a `session_id` (no separate
in-process buffer, so it survives a backend restart), summarized into a rolling
`session_summaries` row once a conversation passes 12 messages to keep prompt-token cost
bounded; long-term is `users` / `preferences` / `car_interactions` / `leads` /
`bookings` keyed by `user_id`, condensed into a short natural-language blurb (never raw rows)
injected into the system prompt on a returning user's first turn. It's a single file, needs
no setup, and is fully inspectable with any SQLite browser — appropriate for a ~100-row,
single-instance take-home; a real deployment would swap it for Postgres without touching
`core/agent.py` or `core/tools.py`.

## Design decisions

The dataset shape drove more decisions than the model choice did. The provided spreadsheet
turned out to have **two unrelated 100-row sheets** (`raw dataset`, `cleaned dataset`, almost
no listing overlap) and **no Price, mileage, or body-type columns** at all — those live
inconsistently inside scraped, HTML-laden description text (dealer contact blocks, hashtags,
Arabic listings, monthly finance figures that aren't the actual price). Rather than fabricate
data or skip structured filtering, `scripts/prepare_dataset.py` runs a one-time Gemini
function-calling pass per listing that extracts `price_aed`, `mileage_km`, `body_type`,
`transmission`, `fuel_type`, `condition`, `warranty_years`, and `regional_spec` **only when
the source text states them** (never inferred), caching the result to `data/cars.csv`. A
field the listing doesn't mention stays `null`, and the agent is instructed to say "not
listed" rather than guess — so the structured pre-filter, the leads it produces, and the
grounding guarantee all stay honest about what's actually known. The tradeoff: only ~31% of
listings have a confirmed price, which the relax-and-fallback logic in `retrieval.py` exists
specifically to handle gracefully. Guardrails (`core/agent.py`'s intent classifier) route
out-of-scope and competitor-comparison questions to **fixed canned replies**, not a freeform
LLM response — cheaper and more reliably on-policy than trusting the model to always refuse
correctly. Both Gemini model tiers used (filter extraction/intent classification/booking
&lead extraction vs. the final conversational reply) currently point at the same
`gemini-3.5-flash-lite` tier: the non-lite "flash" models turned out to be thinking models
that spend 100+ reasoning tokens even on a one-word reply by default, which actively fights
the brief's low-latency/low-cost goals for tasks this small, so lite is the default with the
larger tier left as a one-line env var swap (`GEMINI_LARGE_MODEL`) if richer prose is worth
the added latency.

Out of scope for this pass: authentication (a typed name/ID is trusted as-is, matching the
brief's "recognize a user ID or name" framing); a real payments/CRM integration for leads
(the SQLite `leads` table + mirrored `data/leads.csv` simulate that, as asked); voice/RTL
handling for the ~6% of listings with Arabic text (they load and filter fine, just aren't
translated); slot-conflict checking on bookings (the brief only asked for the Mon–Sat
8am–8pm window, not double-booking prevention); and combining the per-turn intent
classification + filter-extraction calls into one to shave latency, which would cut RPM
pressure against the free tier's ceiling (15 requests/minute on `flash-lite`, easily hit by a
fast back-and-forth — handled today with request retry/backoff in `core/llm.py`, not by
avoiding the calls).

## Evaluation

`eval/golden_set.json` has 18 hand-labeled queries across four categories; `eval/run_eval.py`
scores each independently of response generation (pure retrieval function in, car IDs out)
and writes `eval/results.json`. Latest run:

```
Structured/multi-turn (n=10): P@5=1.00  R@5=1.00  MRR=1.00
Semantic (n=5): avg LLM-judge relevance = 4.00/5
No-match fallback (n=3): 3/3 correctly relaxed/broadened search
Grounding check (n=4): 4/4 responses fully grounded in shown cars
```

- **Structured/multi-turn** (exact make/model/price/year lookups, incl. filters carried
  across turns): precision/recall/MRR against hand-verified car IDs.
- **Semantic** (queries with no single correct answer, e.g. "something rugged for desert
  driving"): scored 1–5 by a separate cheap Gemini call acting as judge.
- **No-match** (queries with no valid inventory match): checks the system relaxed filters or
  fell back to full-corpus search rather than returning nothing or erroring.
- **Grounding**: for a sample of real generated replies, every make+model mentioned in the
  text is checked against the cars actually shown that turn (deterministic regex check, no
  LLM judge) — flags a real, in-corpus car being described out of context, i.e. the
  hallucination pattern the structured pre-filter is meant to prevent.

Every LLM call (data-prep enrichment, filter extraction, intent classification, booking/lead
extraction, response generation) is logged to `data/traces.db` with latency and token counts.
A sample, filtered to real (non-retried) calls:

```
component               n     median   typical avg
filter_extraction       150   997ms    1026ms
intent_classification   42    914ms    905ms
response_generation     38    1358ms   1307ms
```

```
2026-09-15T20:15:25  response_generation    gemini/gemini-3.5-flash-lite  1143.9ms  prompt=1344 completion=164
2026-09-15T20:15:23  lead_extraction        gemini/gemini-3.5-flash-lite   913.8ms  prompt=589  completion=45
2026-09-15T20:15:22  intent_classification  gemini/gemini-3.5-flash-lite   701.2ms  prompt=538  completion=18
2026-09-15T20:15:19  filter_extraction      gemini/gemini-3.5-flash-lite   926.9ms  prompt=408  completion=58
```

## Screenshots

**Multi-turn conversation** — inventory search, then a follow-up ("is there a warranty on
it?") resolved via short-term memory without restating the car, then lead qualification:

![Inventory query](docs/screenshots/02_inventory_query.png)
![Follow-up resolved via short-term memory](docs/screenshots/03_followup_memory.png)
![Lead captured](docs/screenshots/04_lead_captured.png)

**Cross-session recall** — a brand-new session (new session ID) for the same returning user,
greeted with their preferences and last lead recalled from a previous session:

![Cross-session recall](docs/screenshots/05_cross_session_recall.png)
