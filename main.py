"""FastAPI backend: chat processing, state persistence, inventory retrieval.

Thin by design -- every endpoint delegates straight to core/. This file's
only job is the HTTP boundary (request/response schemas, status codes), so
the boundary between API and core logic stays clean and core/ stays
testable without spinning up a server.
"""
from __future__ import annotations

import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import agent, memory, retrieval

app = FastAPI(title="dubizzle cars AI assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SessionRequest(BaseModel):
    user_id: str  # name/handle the user identifies with, e.g. "ahmed" -- used to recognize returning users
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    name: str
    is_returning: bool
    greeting: str


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    intent: str
    cars_shown: list[dict]
    booking: dict | None = None
    lead: dict | None = None


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

@app.post("/session", response_model=SessionResponse)
def create_session(req: SessionRequest) -> SessionResponse:
    user = memory.get_or_create_user(req.user_id)
    session_id = req.session_id or str(uuid.uuid4())
    memory.start_or_touch_session(session_id, req.user_id)

    context = memory.build_user_context(req.user_id, user["is_returning"])
    if user["is_returning"] and context:
        greeting = f"Welcome back, {user['name']}! {context}"
    elif user["is_returning"]:
        greeting = f"Welcome back, {user['name']}!"
    else:
        greeting = (
            f"Hi {user['name']}! I'm your dubizzle cars assistant -- ask me about our inventory, "
            "get details on a listing, or book a test drive."
        )

    return SessionResponse(
        session_id=session_id, user_id=req.user_id, name=user["name"],
        is_returning=user["is_returning"], greeting=greeting,
    )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    result = agent.handle_message(req.session_id, req.user_id, req.message)
    return ChatResponse(**result)


# --------------------------------------------------------------------------
# Inventory (direct pandas filter, no LLM -- for a plain browse view)
# --------------------------------------------------------------------------

def _clean_records(df: pd.DataFrame) -> list[dict]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/inventory")
def list_inventory(
    make: str | None = None,
    model: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    body_type: str | None = None,
    limit: int = 20,
) -> dict:
    df = retrieval.load_inventory()
    filters = {
        "make": make, "model": model, "min_price": min_price, "max_price": max_price,
        "min_year": min_year, "max_year": max_year, "body_type": body_type,
    }
    filtered = retrieval.apply_filters(df, filters)
    return {"count": len(filtered), "cars": _clean_records(filtered.head(limit))}


@app.get("/inventory/{car_id}")
def get_inventory_item(car_id: int) -> dict:
    car = retrieval.get_car(car_id)
    if car is None:
        raise HTTPException(404, f"No car with id {car_id}")
    return car


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
