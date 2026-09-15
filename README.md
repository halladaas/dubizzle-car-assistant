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

I chose Streamlit over a notebook for a real chat UI that exercises the backend the way an
actual user would, while keeping all state server-side so the backend stays fully testable on
its own. For the agent, I used an explicit control loop (intent classification → tool routing
→ response) instead of a framework like LangChain, since at this scope it keeps every step a
visible, loggable, unit-testable function rather than hiding logic behind framework
abstractions. For retrieval, I used hybrid pandas filtering + FAISS: structured filters
(price/year/make) are applied first so hard constraints are never violated or hallucinated,
and FAISS then ranks the filtered survivors by semantic similarity — exact/brute-force search
since ~100 rows doesn't need an approximate index. Memory uses SQLite for both short-term
(session messages, summarized past 12 turns) and long-term (preferences, leads, bookings) —
simple, inspectable, and a drop-in swap to Postgres if this went to production.

### Why no agent framework (LangGraph, Google ADK, etc.)

Both are built to solve problems this app doesn't have:

- **LangGraph** is a graph-based state machine — its value is cycles, conditional branching
  across many nodes, parallel/fan-out tool calls, and durable checkpointed state you can pause
  and resume across sessions. This agent's actual control flow is one straight line: classify
  intent → call one of five tool functions → generate a response. There's no branch to model
  as a graph and no long-running state to checkpoint — short/long-term memory is already
  handled explicitly by `core/memory.py`'s SQLite tables, which is what LangGraph's persistence
  layer would otherwise exist to provide.
- **Google ADK** is built around multi-agent hierarchies (agents delegating to sub-agents via
  an agent-to-agent protocol) and ships its own session/state services and Vertex AI-oriented
  deployment path. This is a single agent with no sub-agents to orchestrate, and adopting ADK's
  session service would mean either running it alongside the SQLite memory layer already built
  (duplicated state) or replacing SQLite with ADK's abstraction (unnecessary coupling to a
  Vertex-centric deployment model this take-home doesn't call for).

Concretely, choosing either here would cost more than it buys:

- **Traceability** — the assessment explicitly grades this. With a hand-written loop, a stack
  trace or a breakpoint lands in code you wrote (`core/agent.py`, `core/tools.py`); with a
  framework, it first passes through the framework's own executor/runtime before reaching your
  logic, and `data/traces.db`'s per-call latency/token logging would need to instrument
  *inside* that runtime rather than wrapping a plain function call.
- **Latency/cost** — both frameworks add state-serialization and orchestration overhead between
  steps, which is real cost when a single chat turn is meant to return in ~1–2s against a
  free-tier model with a 15 requests/minute ceiling (`gemini-3.5-flash-lite`). A five-intent
  linear flow doesn't have enough steps for that overhead to be worth paying.
- **Interview defensibility** — every function in the control loop is one I can explain line by
  line; a framework means part of the explanation is the framework's design choices rather than
  mine, which is a weaker position for a take-home meant to demonstrate engineering judgment.

If this grew into a genuinely multi-agent system (e.g. a separate pricing-negotiation agent, a
document-verification agent, human-in-the-loop approval steps), LangGraph's or ADK's
orchestration would start earning its overhead — it doesn't here.

## Architecture

```mermaid
flowchart TD
    U[User message] --> IC["Intent classification (LLM call)"]

    IC -->|inventory_query| FE["Filter extraction (LLM call)"]
    IC -->|booking| BK["book_slot tool<br/>(validate Mon–Sat 8am–8pm)"]
    IC -->|lead_info| LD["save_lead tool"]
    IC -->|chitchat| RG
    IC -->|out_of_scope / competitor| CR["Canned decline reply"]

    FE --> PF["Pandas structured filter<br/>(price / year / make / body type)"]
    PF -->|empty result| RX["Relax filters one field at a time<br/>→ full-corpus fallback"]
    RX --> FS
    PF -->|has rows| FS["FAISS semantic search<br/>(IndexFlatIP, within filtered rows)"]

    FS --> RG["Response generation (LLM call)<br/>grounded in retrieved car rows"]
    BK --> RG
    LD --> RG

    subgraph Memory [SQLite]
        ST["Short-term: session messages<br/>+ rolling summary after 12 turns"]
        LT["Long-term: users / preferences /<br/>car_interactions / leads / bookings"]
    end

    ST <-.-> IC
    ST <-.-> RG
    LT <-.-> RG

    RG --> R[Response to user]

    IC -.->|every LLM call logged| TR[(traces.db:<br/>component, latency, tokens)]
    FE -.-> TR
    RG -.-> TR
```

Every arrow above is a real function call in `core/agent.py`, `core/retrieval.py`,
`core/memory.py`, and `core/tools.py` — no orchestration framework sits between this diagram
and the code.

## Design decisions

The dataset had no structured Price/mileage/body-type fields — those values lived
inconsistently inside scraped description text. A one-time enrichment script extracts them
via LLM function-calling only when explicitly stated in the source text (never inferred), so
retrieval and grounding stay honest about what's actually known, at the cost of ~31% price
coverage — handled by retrieval's filter-relaxation fallback. Guardrails route out-of-scope/
competitor questions to fixed canned replies rather than trusting the LLM to freeform-refuse
correctly.

Out of scope: real authentication (a typed name/ID is trusted as-is), a real payments/CRM
integration for leads (simulated via SQLite + CSV), Arabic/RTL translation, and booking
slot-conflict prevention (only the Mon–Sat 8am–8pm window was required). A next step would be
merging the per-turn intent-classification and filter-extraction LLM calls into one to reduce
latency and free-tier rate-limit pressure.

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
