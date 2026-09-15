"""Unit tests for the deterministic parts of hybrid retrieval: the pandas
pre-filter and its progressive relaxation. No network/LLM calls -- these run
against a small synthetic fixture, not the real (LLM-derived) cars.csv, so
they're fast and don't depend on API availability.
"""
import pandas as pd
import pytest

from core.retrieval import apply_filters, filter_with_relaxation


@pytest.fixture
def cars_df():
    return pd.DataFrame([
        {"car_id": 0, "make": "toyota", "model": "land cruiser", "year": 2022, "price_aed": 280000, "body_type": "suv"},
        {"car_id": 1, "make": "toyota", "model": "corolla", "year": 2020, "price_aed": 55000, "body_type": "sedan"},
        {"car_id": 2, "make": "honda", "model": "civic", "year": 2023, "price_aed": 75000, "body_type": "sedan"},
        {"car_id": 3, "make": "ford", "model": "mustang", "year": 2019, "price_aed": None, "body_type": "coupe"},
        {"car_id": 4, "make": "toyota", "model": "yaris", "year": 2024, "price_aed": None, "body_type": "hatchback"},
    ])


def test_apply_filters_make_and_price(cars_df):
    result = apply_filters(cars_df, {"make": "toyota", "max_price": 100000})
    assert set(result["car_id"]) == {1}


def test_apply_filters_case_insensitive_make(cars_df):
    result = apply_filters(cars_df, {"make": "TOYOTA"})
    assert set(result["car_id"]) == {0, 1, 4}


def test_apply_filters_price_excludes_null_price_rows(cars_df):
    # A price filter can't confirm a null-price row is in budget, so it's excluded.
    result = apply_filters(cars_df, {"max_price": 300000})
    assert 3 not in set(result["car_id"])
    assert 4 not in set(result["car_id"])


def test_apply_filters_body_type(cars_df):
    result = apply_filters(cars_df, {"body_type": "sedan"})
    assert set(result["car_id"]) == {1, 2}


def test_apply_filters_no_filters_returns_all(cars_df):
    result = apply_filters(cars_df, {})
    assert len(result) == len(cars_df)


def test_relaxation_drops_price_ceiling_before_giving_up(cars_df):
    # No toyota is <= 10,000 AED, but relaxing max_price should surface toyotas.
    result = filter_with_relaxation(cars_df, {"make": "toyota", "max_price": 10000})
    assert "max_price" in result.dropped_fields
    assert not result.used_full_corpus_fallback
    assert set(result.df["car_id"]).issubset({0, 1, 4})


def test_relaxation_falls_back_to_full_corpus_when_nothing_survives(cars_df):
    # No such make exists at all -- even dropping every other field won't help,
    # since 'model' is the last thing relaxation drops and there's still no
    # make match. Full-corpus fallback should kick in.
    result = filter_with_relaxation(cars_df, {"make": "bugatti", "model": "chiron", "max_price": 1000})
    assert result.used_full_corpus_fallback is True
    assert len(result.df) == len(cars_df)


def test_relaxation_keeps_make_filter_longest(cars_df):
    # make is never in RELAX_ORDER, so a make match should survive relaxation
    # of everything else.
    result = filter_with_relaxation(cars_df, {"make": "honda", "max_price": 1000, "body_type": "suv"})
    assert set(result.df["car_id"]) == {2}
    assert not result.used_full_corpus_fallback
