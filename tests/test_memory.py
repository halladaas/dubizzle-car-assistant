"""Unit tests for SQLite-backed memory. No network calls -- session
summarization (the one function that calls an LLM) is tested elsewhere by
inspection, not here."""


def test_get_or_create_user_new_then_returning(memory_module):
    m = memory_module
    user = m.get_or_create_user("ahmed", "Ahmed")
    assert user["is_returning"] is False

    again = m.get_or_create_user("ahmed")
    assert again["is_returning"] is True
    assert again["name"] == "Ahmed"


def test_messages_round_trip_in_order(memory_module):
    m = memory_module
    m.save_message("s1", "u1", "user", "hello")
    m.save_message("s1", "u1", "assistant", "hi there")
    recent = m.get_recent_messages("s1")
    assert [r["role"] for r in recent] == ["user", "assistant"]
    assert [r["content"] for r in recent] == ["hello", "hi there"]


def test_recent_messages_respects_limit(memory_module):
    m = memory_module
    for i in range(5):
        m.save_message("s1", "u1", "user", f"msg{i}")
    recent = m.get_recent_messages("s1", limit=2)
    assert [r["content"] for r in recent] == ["msg3", "msg4"]


def test_preferences_upsert_overwrites(memory_module):
    m = memory_module
    m.save_preference("u1", "max_price", "100000")
    m.save_preference("u1", "max_price", "120000")
    prefs = m.get_preferences("u1")
    assert prefs["max_price"] == "120000"


def test_save_lead_writes_sqlite_and_csv(memory_module):
    m = memory_module
    lead = m.save_lead("u1", "80k-120k AED", "family suv", 4)
    assert lead["lead_id"] == 1

    stored = m.get_leads("u1")
    assert len(stored) == 1
    assert stored[0]["price_range"] == "80k-120k AED"

    assert m.LEADS_CSV_PATH.exists()
    assert "80k-120k AED" in m.LEADS_CSV_PATH.read_text()


def test_session_state_round_trip(memory_module):
    m = memory_module
    state = m.get_session_state("s1")
    assert state == {"filters": {}, "last_shown_car_ids": [], "lead_in_progress": {}}

    state["filters"] = {"make": "toyota", "max_price": 100000}
    state["last_shown_car_ids"] = [1, 2, 3]
    m.set_session_state("s1", "u1", state)

    reloaded = m.get_session_state("s1")
    assert reloaded["filters"] == {"make": "toyota", "max_price": 100000}
    assert reloaded["last_shown_car_ids"] == [1, 2, 3]


def test_bookings_round_trip(memory_module):
    m = memory_module
    booking = m.save_booking("u1", "s1", 4, "2026-09-15", "10:00")
    assert booking["booking_id"] == 1
    bookings = m.get_bookings("u1")
    assert len(bookings) == 1
    assert bookings[0]["car_id"] == 4


def test_build_user_context_returns_none_for_new_user(memory_module):
    m = memory_module
    m.get_or_create_user("new_user")
    assert m.build_user_context("new_user", is_returning=False) is None
