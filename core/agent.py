"""The explicit control loop: classify intent -> route to a small set of
hand-defined tools -> synthesize a response. No agent framework -- every
step below is a plain Python function you can set a breakpoint in, log, or
unit test in isolation. That's the traceability story: nothing happens
inside a hidden framework retry loop or an opaque planner.
"""
from __future__ import annotations

from datetime import datetime

from core import memory, tools
from core.llm import LARGE_MODEL, call_llm, call_with_tool

# --------------------------------------------------------------------------
# Guardrails: intent classification gates which path runs. Out-of-scope and
# competitor-mention paths return a fixed template -- never a freeform LLM
# reply -- since a canned response is both cheaper and more reliably
# on-policy than trusting the model to always refuse correctly.
# --------------------------------------------------------------------------

INTENT_LABELS = [
    "inventory_query", "booking", "chitchat", "lead_info",
    "out_of_scope", "competitor_mention",
]

INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Classify the user's latest message into exactly one category.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": INTENT_LABELS,
                    "description": (
                        "inventory_query: searching/asking about cars in stock, incl. follow-up "
                        "detail questions about a previously shown car. "
                        "booking: wants to schedule/reschedule/cancel a test drive/viewing. "
                        "chitchat: greetings, thanks, small talk with no car-search intent. "
                        "lead_info: volunteering budget/needs/contact info to be qualified as a lead. "
                        "out_of_scope: anything unrelated to buying a used car here (general knowledge, "
                        "code, unrelated tasks). "
                        "competitor_mention: asks about or wants to compare other car marketplaces/platforms."
                    ),
                },
            },
            "required": ["intent"],
        },
    },
}


def classify_intent(message: str, recent_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Classify the latest user message given the recent conversation for context.",
        },
        {"role": "user", "content": f"RECENT CONTEXT:\n{recent_context}\n\nLATEST MESSAGE:\n{message}"},
    ]
    result = call_with_tool(messages, INTENT_TOOL, component="intent_classification")
    if result is None or result.get("intent") not in INTENT_LABELS:
        return "inventory_query"  # safe default: worst case we search and return broad results
    return result["intent"]


OUT_OF_SCOPE_REPLY = (
    "I'm your dubizzle cars assistant, so I can only help with exploring our car inventory, "
    "answering questions about listings, and booking test drives. I can't help with that request, "
    "but I'd be glad to help you find a car -- what are you looking for?"
)

COMPETITOR_REPLY = (
    "I can only speak to dubizzle's own inventory and can't discuss or compare other car-selling "
    "platforms. Happy to help you find something in our listings instead -- what kind of car are you after?"
)


# --------------------------------------------------------------------------
# Booking parameter extraction
# --------------------------------------------------------------------------

BOOKING_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_booking_request",
        "description": "Extract test-drive booking details from the user's message.",
        "parameters": {
            "type": "object",
            "properties": {
                "car_id": {
                    "type": ["integer", "null"],
                    "description": "The car_id being booked, matched against the 'candidate cars' list by make/model/position ('the first one', 'the Honda'). Null if unclear which car.",
                },
                "day": {
                    "type": ["string", "null"],
                    "description": "Requested date resolved to YYYY-MM-DD using today's date given below. Null if not stated.",
                },
                "time": {
                    "type": ["string", "null"],
                    "description": "Requested time as 24h HH:MM. Null if not stated.",
                },
            },
            "required": ["car_id", "day", "time"],
        },
    },
}


def extract_booking_request(message: str, candidate_cars: list[dict], recent_context: str) -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d (%A)")
    candidates_str = "\n".join(
        f"- car_id={c['car_id']}: {c.get('year')} {c.get('make')} {c.get('model')} {c.get('trim') or ''}"
        for c in candidate_cars
    ) or "(none shown yet)"
    messages = [
        {"role": "system", "content": f"Today is {today_str}. Extract booking details from the user's message."},
        {
            "role": "user",
            "content": f"RECENT CONTEXT:\n{recent_context}\n\nCANDIDATE CARS:\n{candidates_str}\n\nMESSAGE:\n{message}",
        },
    ]
    result = call_with_tool(messages, BOOKING_TOOL, component="booking_extraction")
    return result or {"car_id": None, "day": None, "time": None}


# --------------------------------------------------------------------------
# Lead qualification extraction
# --------------------------------------------------------------------------

LEAD_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_lead_info",
        "description": "Extract lead-qualification details volunteered in the user's message.",
        "parameters": {
            "type": "object",
            "properties": {
                "price_range": {"type": ["string", "null"], "description": "Budget/price range as stated, e.g. '80k-120k AED'."},
                "needs": {"type": ["string", "null"], "description": "What they need it for / must-haves, e.g. 'family SUV, automatic, low mileage'."},
                "car_id": {"type": ["integer", "null"], "description": "A specific car from the candidate list they're interested in, if any."},
            },
            "required": ["price_range", "needs", "car_id"],
        },
    },
}


def extract_lead_info(message: str, candidate_cars: list[dict], recent_context: str) -> dict:
    candidates_str = "\n".join(
        f"- car_id={c['car_id']}: {c.get('year')} {c.get('make')} {c.get('model')}"
        for c in candidate_cars
    ) or "(none shown yet)"
    messages = [
        {"role": "system", "content": "Extract lead-qualification info from the user's message."},
        {
            "role": "user",
            "content": f"RECENT CONTEXT:\n{recent_context}\n\nCANDIDATE CARS:\n{candidates_str}\n\nMESSAGE:\n{message}",
        },
    ]
    result = call_with_tool(messages, LEAD_TOOL, component="lead_extraction")
    return result or {"price_range": None, "needs": None, "car_id": None}


# --------------------------------------------------------------------------
# Response synthesis (grounded in retrieved/shown car data)
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the dubizzle cars AI assistant. You help users explore a used-car inventory, answer questions about specific listings, and guide them toward booking a test drive.

Rules:
- Only state facts that appear in the CAR DATA provided below. Never invent a price, mileage, feature, or car that isn't in that data.
- If a field (e.g. price) is null/missing for a car, say it isn't listed rather than guessing.
- Be concise and conversational. When listing cars, briefly mention make/model/year/price (or "price on request" if null) and one or two standout details.
- If USER CONTEXT below shows this is a returning user with known preferences, you may reference them naturally (e.g. "since you were looking at SUVs under 100k AED before...").
- Stay focused on helping with cars; keep chitchat brief and steer back to how you can help.
"""


def _format_cars_block(cars: list[dict], label: str) -> str:
    if not cars:
        return f"{label}: (none)"
    lines = [label + ":"]
    for c in cars:
        price = f"{int(c['price_aed']):,} AED" if c.get("price_aed") is not None else "price not listed"
        mileage = f"{int(c['mileage_km']):,} km" if c.get("mileage_km") is not None else "mileage not listed"
        lines.append(
            f"- car_id={c['car_id']}: {c.get('year')} {c.get('make')} {c.get('model')} {c.get('trim') or ''} | "
            f"{price} | {mileage} | body={c.get('body_type') or 'n/a'} | color={c.get('exterior_color') or 'n/a'} | "
            f"transmission={c.get('transmission') or 'n/a'} | fuel={c.get('fuel_type') or 'n/a'} | "
            f"condition={c.get('condition') or 'n/a'} | warranty={c.get('warranty_years') or 'n/a'} yrs | "
            f"spec={c.get('regional_spec') or 'n/a'}\n  description: {str(c.get('description') or '')[:400]}"
        )
    return "\n".join(lines)


def generate_response(
    user_message: str,
    recent_context: str,
    session_summary: str | None,
    user_context: str | None,
    current_results: list[dict],
    previously_shown: list[dict],
    extra_note: str | None = None,
) -> str:
    context_parts = []
    if user_context:
        context_parts.append(f"USER CONTEXT (returning user): {user_context}")
    if session_summary:
        context_parts.append(f"EARLIER IN THIS SESSION: {session_summary}")
    context_parts.append(f"RECENT MESSAGES:\n{recent_context}")
    if previously_shown:
        context_parts.append(_format_cars_block(previously_shown, "PREVIOUSLY SHOWN CARS"))
    context_parts.append(_format_cars_block(current_results, "CURRENT SEARCH RESULTS"))
    if extra_note:
        context_parts.append(f"NOTE: {extra_note}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(context_parts) + f"\n\nUSER'S MESSAGE: {user_message}"},
    ]
    response = call_llm(messages=messages, component="response_generation", model=LARGE_MODEL, max_tokens=500)
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------
# Main control loop
# --------------------------------------------------------------------------

def handle_message(session_id: str, user_id: str, user_message: str) -> dict:
    memory.start_or_touch_session(session_id, user_id)
    memory.save_message(session_id, user_id, "user", user_message)

    state = memory.get_session_state(session_id)
    recent = memory.get_recent_messages(session_id, limit=8)
    recent_context = "\n".join(f"{m['role']}: {m['content']}" for m in recent[:-1]) or "(start of conversation)"
    session_summary = memory.get_session_summary(session_id)

    user_row = memory.get_or_create_user(user_id)
    user_context = memory.build_user_context(user_id, is_returning=user_row.get("is_returning", False))

    previously_shown = [
        car for cid in state["last_shown_car_ids"]
        if (car := tools.get_car_details(cid)) is not None
    ]

    intent = classify_intent(user_message, recent_context)
    result: dict = {"intent": intent, "cars_shown": [], "booking": None, "lead": None}

    if intent == "out_of_scope":
        reply = OUT_OF_SCOPE_REPLY

    elif intent == "competitor_mention":
        reply = COMPETITOR_REPLY

    elif intent == "booking":
        candidates = previously_shown
        extraction = extract_booking_request(user_message, candidates, recent_context)
        if extraction.get("car_id") is None or extraction.get("day") is None or extraction.get("time") is None:
            missing = [f for f in ("car_id", "day", "time") if extraction.get(f) is None]
            reply = generate_response(
                user_message, recent_context, session_summary, user_context,
                current_results=[], previously_shown=previously_shown,
                extra_note=f"The user wants to book a test drive but hasn't specified: {', '.join(missing)}. Ask a brief clarifying question for the missing info. Available slots: Mon-Sat, 8am-8pm.",
            )
        else:
            booking_result = tools.book_slot(
                user_id, session_id, extraction["car_id"], extraction["day"], extraction["time"]
            )
            result["booking"] = booking_result
            if booking_result["success"]:
                b = booking_result["booking"]
                car = booking_result["car"]
                reply = (
                    f"Booked! Your test drive for the {car['year']} {car['make']} {car['model']} "
                    f"is confirmed for {b['day']} at {b['time']}. See you then."
                )
            else:
                reply = f"I couldn't book that: {booking_result['error']}"

    elif intent == "lead_info":
        candidates = previously_shown
        extraction = extract_lead_info(user_message, candidates, recent_context)
        lead_state = dict(state["lead_in_progress"])
        for k in ("price_range", "needs", "car_id"):
            if extraction.get(k) is not None:
                lead_state[k] = extraction[k]
        state["lead_in_progress"] = lead_state

        if lead_state.get("price_range") or lead_state.get("needs"):
            lead = tools.save_lead(
                user_id, lead_state.get("price_range"), lead_state.get("needs"), lead_state.get("car_id")
            )
            result["lead"] = lead
            reply = generate_response(
                user_message, recent_context, session_summary, user_context,
                current_results=[], previously_shown=previously_shown,
                extra_note=(
                    f"You just recorded this lead: budget={lead_state.get('price_range')}, "
                    f"needs={lead_state.get('needs')}. Thank the user briefly and confirm what you noted; "
                    "if either budget or needs is still unknown, ask one short follow-up question for it."
                ),
            )
        else:
            reply = generate_response(
                user_message, recent_context, session_summary, user_context,
                current_results=[], previously_shown=previously_shown,
                extra_note="Ask a brief, friendly question to learn the user's budget and what they need the car for.",
            )

    elif intent == "chitchat":
        reply = generate_response(
            user_message, recent_context, session_summary, user_context,
            current_results=[], previously_shown=previously_shown,
        )

    else:  # inventory_query (default/fallback path too)
        search = tools.search_inventory(user_message, state["filters"], top_k=5)
        state["filters"] = search["filters"]
        state["last_shown_car_ids"] = [c["car_id"] for c in search["cars"]]
        result["cars_shown"] = search["cars"]

        note = None
        if search["used_full_corpus_fallback"]:
            note = "No cars matched the stated filters, so these are the closest semantic matches across the whole inventory -- tell the user you broadened the search."
        elif search["dropped_fields"]:
            note = f"No exact matches with all filters, so these results relax: {', '.join(search['dropped_fields'])}. Tell the user you broadened the search on those criteria."
        if search["extraction_failed"]:
            note = (note + " " if note else "") + "Filter extraction failed this turn; treat this as a plain keyword match."

        reply = generate_response(
            user_message, recent_context, session_summary, user_context,
            current_results=search["cars"], previously_shown=previously_shown,
            extra_note=note,
        )

    memory.save_message(session_id, user_id, "assistant", reply)
    memory.set_session_state(session_id, user_id, state)
    memory.maybe_summarize_session(session_id, user_id)

    result["reply"] = reply
    return result
