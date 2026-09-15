import pytest


@pytest.fixture
def memory_module(tmp_path, monkeypatch):
    """core.memory backed by a throwaway SQLite DB/CSV per test, so tests
    never touch data/memory.db or data/leads.csv."""
    import core.memory as memory

    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test_memory.db")
    monkeypatch.setattr(memory, "LEADS_CSV_PATH", tmp_path / "test_leads.csv")
    memory.init_db()
    return memory
