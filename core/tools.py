"""The small, explicit set of tool functions the control loop can call.
Each one is a plain Python function -- no framework tool-registry magic --
so every call is a regular stack frame you can breakpoint, log, or unit
test directly.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from core import memory, retrieval

OPEN_HOUR = 8   # 8am
CLOSE_HOUR = 20  # 8pm, last bookable start time is 19:xx

CAR_DISPLAY_COLUMNS = [
    "car_id", "make", "model", "trim", "year", "price_aed", "mileage_km",
    "body_type", "exterior_color", "transmission", "fuel_type", "condition",
    "warranty_years", "regional_spec", "title", "photo_url",
]


def _car_row_to_dict(row: pd.Series) -> dict:
    out = {}
    for col in CAR_DISPLAY_COLUMNS:
        val = row.get(col)
        out[col] = None if pd.isna(val) else val
    return out


def search_inventory(query: str, prior_filters: dict, top_k: int = 5) -> dict:
    """Structured pre-filter + FAISS semantic search. Returns grounded
    results (rows straight out of the filtered DataFrame -- never
    model-generated) plus metadata the agent can use to explain itself
    (e.g. 'broadened the search since no exact match')."""
    filters = retrieval.extract_filters(query, prior_filters)
    search = retrieval.hybrid_search(query, filters, top_k=top_k)

    cars = [_car_row_to_dict(row) for _, row in search["results"].iterrows()]
    return {
        "cars": cars,
        "filters": filters,
        "dropped_fields": search["dropped_fields"],
        "used_full_corpus_fallback": search["used_full_corpus_fallback"],
        "extraction_failed": search["extraction_failed"],
    }


def get_car_details(car_id: int) -> dict | None:
    return retrieval.get_car(int(car_id))


def book_slot(user_id: str, session_id: str, car_id: int, day: str, time_slot: str) -> dict:
    """Validate against Mon-Sat 8am-8pm and persist. `day` must be an ISO
    date (YYYY-MM-DD); `time_slot` must be HH:MM 24h. Returns a specific,
    relayable error rather than a generic failure."""
    car = retrieval.get_car(int(car_id))
    if car is None:
        return {"success": False, "error": f"No car with id {car_id} in inventory."}

    try:
        date_obj = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return {"success": False, "error": f"'{day}' isn't a valid date (expected YYYY-MM-DD)."}

    if date_obj.weekday() == 6:  # Sunday
        return {"success": False, "error": "We're closed on Sundays -- please pick a day Monday to Saturday."}

    try:
        time_obj = datetime.strptime(time_slot, "%H:%M")
    except ValueError:
        return {"success": False, "error": f"'{time_slot}' isn't a valid time (expected HH:MM, 24h)."}

    if not (OPEN_HOUR <= time_obj.hour < CLOSE_HOUR):
        return {"success": False, "error": "Test drives are available 8am-8pm -- please pick a time in that window."}

    booking = memory.save_booking(user_id, session_id, int(car_id), day, time_slot)
    memory.record_car_interaction(user_id, int(car_id), "booked")
    return {
        "success": True,
        "booking": booking,
        "car": {"make": car.get("make"), "model": car.get("model"), "year": car.get("year")},
    }


def save_lead(user_id: str, price_range: str | None, needs: str | None, car_id: int | None) -> dict:
    lead = memory.save_lead(user_id, price_range, needs, car_id)
    if price_range:
        memory.save_preference(user_id, "price_range", price_range)
    if needs:
        memory.save_preference(user_id, "needs", needs)
    if car_id is not None:
        memory.record_car_interaction(user_id, int(car_id), "liked")
    return lead
