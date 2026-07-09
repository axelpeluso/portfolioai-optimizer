"""
Smoke tests for the core (no-secret) PortfolioAI endpoints.

These exercise the ML pipeline end to end against the bundled prices.csv —
they need no external keys (Claude / Supabase are only touched by their own
endpoints, which aren't tested here). Run with:  pytest -q   (from api/)
"""
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "BND", "GLD", "AMZN"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_tickers_universe():
    r = client.get("/tickers")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 250
    assert all("symbol" in t and "name" in t for t in data["tickers"][:5])


def test_optimize_default_portfolio():
    r = client.post("/optimize", json={
        "tickers": DEFAULT_TICKERS,
        "current_holdings": {"AAPL": 5000, "MSFT": 2000},
    })
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    d = data["data"]
    assert d["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= d["risk_score"] <= 1.0
    # weights should be present and sum to ~1
    weights = d["optimization"]["optimal_weights"]
    assert set(weights) <= set(d["tickers"])
    assert abs(sum(weights.values()) - 1.0) < 1e-3


def test_optimize_rejects_single_ticker():
    r = client.post("/optimize", json={"tickers": ["AAPL"]})
    assert r.status_code == 400


def test_optimize_rejects_too_many_tickers():
    r = client.post("/optimize", json={"tickers": [f"T{i}" for i in range(16)]})
    assert r.status_code == 400
