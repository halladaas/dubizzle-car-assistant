"""Unit tests for tools.book_slot's validation logic (Mon-Sat, 8am-8pm),
independent of the real inventory or LLM calls."""
from datetime import datetime, timedelta

import pytest

from core import tools


def _next_weekday(target_weekday: int) -> str:
    """target_weekday: 0=Mon ... 6=Sun. Returns the next such date as YYYY-MM-DD."""
    today = datetime.now()
    days_ahead = (target_weekday - today.weekday()) % 7 or 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


@pytest.fixture
def fake_car(monkeypatch):
    car = {"car_id": 4, "make": "haval", "model": "h9", "year": 2026}
    monkeypatch.setattr(tools.retrieval, "get_car", lambda car_id: car if car_id == 4 else None)
    return car


def test_book_slot_rejects_sunday(memory_module, fake_car):
    sunday = _next_weekday(6)
    result = tools.book_slot("u1", "s1", 4, sunday, "10:00")
    assert result["success"] is False
    assert "Sunday" in result["error"]


def test_book_slot_rejects_before_opening(memory_module, fake_car):
    tuesday = _next_weekday(1)
    result = tools.book_slot("u1", "s1", 4, tuesday, "07:00")
    assert result["success"] is False
    assert "8am-8pm" in result["error"]


def test_book_slot_rejects_at_or_after_closing(memory_module, fake_car):
    tuesday = _next_weekday(1)
    result = tools.book_slot("u1", "s1", 4, tuesday, "20:00")
    assert result["success"] is False
    assert "8am-8pm" in result["error"]


def test_book_slot_accepts_valid_weekday_slot(memory_module, fake_car):
    tuesday = _next_weekday(1)
    result = tools.book_slot("u1", "s1", 4, tuesday, "10:00")
    assert result["success"] is True
    assert result["car"]["make"] == "haval"
    assert result["booking"]["day"] == tuesday
    assert result["booking"]["time"] == "10:00"


def test_book_slot_accepts_last_valid_hour(memory_module, fake_car):
    tuesday = _next_weekday(1)
    result = tools.book_slot("u1", "s1", 4, tuesday, "19:30")
    assert result["success"] is True


def test_book_slot_unknown_car(memory_module):
    result = tools.book_slot("u1", "s1", 999, "2026-09-15", "10:00")
    assert result["success"] is False
    assert "999" in result["error"]


def test_book_slot_invalid_date_format(memory_module, fake_car):
    result = tools.book_slot("u1", "s1", 4, "next tuesday", "10:00")
    assert result["success"] is False


def test_book_slot_invalid_time_format(memory_module, fake_car):
    tuesday = _next_weekday(1)
    result = tools.book_slot("u1", "s1", 4, tuesday, "10am")
    assert result["success"] is False
