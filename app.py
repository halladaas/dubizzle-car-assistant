"""Streamlit client. A thin presentation layer over the FastAPI backend --
every piece of state (memory, retrieval, bookings, leads) lives server-side;
this file only renders it and forwards user input via HTTP. That boundary
is deliberate: the backend is fully usable (and testable) without this UI.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="dubizzle cars assistant", page_icon="🚗", layout="centered")


def _api_post(path: str, json: dict) -> dict:
    resp = httpx.post(f"{BACKEND_URL}{path}", json=json, timeout=60)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# Login screen: identify/recognize the user before starting a session
# --------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.title("🚗 dubizzle cars assistant")
    st.caption("Enter a name or ID. Use the same one again later to test cross-session memory.")

    with st.form("login"):
        user_id = st.text_input("Your name or user ID", placeholder="e.g. ahmed")
        submitted = st.form_submit_button("Start chatting")

    if submitted and user_id.strip():
        try:
            session = _api_post("/session", {"user_id": user_id.strip().lower()})
        except httpx.ConnectError:
            st.error(f"Can't reach the backend at {BACKEND_URL}. Is `uvicorn main:app` running?")
            st.stop()

        st.session_state.session_id = session["session_id"]
        st.session_state.user_id = session["user_id"]
        st.session_state.name = session["name"]
        st.session_state.messages = [{"role": "assistant", "content": session["greeting"]}]
        st.rerun()

    st.stop()


# --------------------------------------------------------------------------
# Chat screen
# --------------------------------------------------------------------------

st.title("🚗 dubizzle cars assistant")
st.caption(f"Signed in as **{st.session_state.name}** (session `{st.session_state.session_id[:8]}`)")

with st.sidebar:
    st.subheader("Session")
    st.write(f"User: {st.session_state.name}")
    st.write(f"Session ID: `{st.session_state.session_id}`")
    if st.button("Sign out / switch user"):
        for key in ("session_id", "user_id", "name", "messages"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption(
        "Try: 'show me SUVs under 150k', 'is there a warranty on the first one?', "
        "'book a test drive for it Tuesday at 10am', 'my budget is 80k-120k for a family car'."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for car in msg.get("cars_shown", []):
            price = f"{int(car['price_aed']):,} AED" if car.get("price_aed") is not None else "price not listed"
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if car.get("photo_url"):
                        st.image(car["photo_url"], use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{car.get('year')} {car.get('make', '').title()} {car.get('model', '').title()}**")
                    st.caption(f"{price} · {car.get('body_type') or 'n/a'} · {car.get('regional_spec') or ''}")

if prompt := st.chat_input("Ask about our inventory, or book a test drive..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _api_post(
                    "/chat",
                    {
                        "session_id": st.session_state.session_id,
                        "user_id": st.session_state.user_id,
                        "message": prompt,
                    },
                )
            except httpx.ConnectError:
                st.error(f"Can't reach the backend at {BACKEND_URL}. Is `uvicorn main:app` running?")
                st.stop()

        st.markdown(result["reply"])
        for car in result.get("cars_shown", []):
            price = f"{int(car['price_aed']):,} AED" if car.get("price_aed") is not None else "price not listed"
            with st.container(border=True):
                cols = st.columns([1, 3])
                with cols[0]:
                    if car.get("photo_url"):
                        st.image(car["photo_url"], use_container_width=True)
                with cols[1]:
                    st.markdown(f"**{car.get('year')} {car.get('make', '').title()} {car.get('model', '').title()}**")
                    st.caption(f"{price} · {car.get('body_type') or 'n/a'} · {car.get('regional_spec') or ''}")

        if result.get("booking") and result["booking"].get("success"):
            st.success("Test drive booked ✅")
        if result.get("lead"):
            st.info("Lead recorded for follow-up 📋")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["reply"], "cars_shown": result.get("cars_shown", [])}
    )
