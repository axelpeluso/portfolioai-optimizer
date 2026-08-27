"""
Tests for the SnapTrade read-only import.

No network and no SnapTrade SDK required: reconciliation is pure, and the route
tests only need to prove the auth gate rejects unauthenticated callers before
any brokerage call is attempted. Run with:  pytest -q   (from api/)
"""
from fastapi.testclient import TestClient

import main
import brokerage as st

client = TestClient(main.app)

REAL = "AAPL"      # in tickers.json
REAL2 = "MSFT"


def pos(symbol, units=10, price=100.0, kind="cs", market_value=None):
    """A SnapTrade-shaped position (nested symbol object)."""
    return {
        "symbol": {"symbol": {"symbol": symbol, "type": {"code": kind}}},
        "units": units,
        "price": price,
        "market_value": market_value if market_value is not None else units * price,
    }


# ── reconciliation ────────────────────────────────────────────
def test_supported_and_unsupported_split():
    out = st.reconcile([pos(REAL), pos("VTSAX"), pos(REAL2)])
    assert {p["symbol"] for p in out["supported"]} == {REAL, REAL2}
    assert out["unsupported"][0]["symbol"] == "VTSAX"
    assert out["unsupported"][0]["reason"] == st.REASON_UNIVERSE


def test_sorted_by_value_and_top_n_preselected():
    universe = sorted(st.universe())[:20]
    # Descending value so position i is worth more than position i+1.
    raw = [pos(s, units=1, price=float(100 - i)) for i, s in enumerate(universe)]
    out = st.reconcile(raw)

    values = [p["value"] for p in out["supported"]]
    assert values == sorted(values, reverse=True), "must be ranked by value"

    selected = [p for p in out["supported"] if p["selected"]]
    assert len(selected) == st.MAX_TICKERS
    assert selected == out["supported"][:st.MAX_TICKERS], "top N by value"
    for p in out["supported"][st.MAX_TICKERS:]:
        assert p["selected"] is False
        assert p["reason"] == st.REASON_OVERFLOW


def test_overflow_positions_are_listed_not_dropped():
    """The user must be able to see and swap in what was not pre-selected."""
    universe = sorted(st.universe())[:20]
    out = st.reconcile([pos(s, price=float(100 - i)) for i, s in enumerate(universe)])
    assert len(out["supported"]) == 20, "nothing silently discarded"
    assert any("beyond the top" in n for n in out["notes"])


def test_duplicate_symbols_across_accounts_are_merged():
    out = st.reconcile([pos(REAL, units=10, price=100.0),
                        pos(REAL, units=5,  price=100.0)])
    assert len(out["supported"]) == 1
    assert out["supported"][0]["units"] == 15
    assert out["supported"][0]["value"] == 1500.0


def test_unmodellable_kinds_are_excluded_with_reason():
    out = st.reconcile([pos(REAL), pos("BTC", kind="crypto")])
    reasons = {p["symbol"]: p["reason"] for p in out["unsupported"]}
    assert reasons["BTC"] == st.REASON_KIND


def test_zero_value_position_is_excluded():
    out = st.reconcile([pos(REAL, units=0, price=0.0, market_value=0)])
    assert out["supported"] == []
    assert out["unsupported"][0]["reason"] == st.REASON_NOVALUE


def test_cash_is_reported_but_not_a_position():
    out = st.reconcile([pos(REAL)], cash=2300.0)
    assert out["cash"] == 2300.0
    assert all(p["symbol"] != "CASH" for p in out["supported"])
    assert any("cash" in n.lower() for n in out["notes"])


def test_total_value_counts_supported_only():
    out = st.reconcile([pos(REAL, units=10, price=100.0),      # 1000 supported
                        pos("VTSAX", units=10, price=50.0)],   # 500 unsupported
                       cash=250.0)
    assert out["total_value"] == 1000.0


def test_flat_symbol_shape_is_handled():
    """Some brokerages return a flatter symbol object than others."""
    out = st.reconcile([{"symbol": {"symbol": REAL, "type": {"code": "cs"}},
                         "units": 2, "price": 50.0, "market_value": 100.0}])
    assert out["supported"][0]["symbol"] == REAL


# ── auth gate ─────────────────────────────────────────────────
def test_positions_requires_token():
    assert client.get("/snaptrade/positions").status_code == 401


def test_accounts_requires_token():
    assert client.get("/snaptrade/accounts").status_code == 401


def test_connect_requires_token():
    assert client.post("/snaptrade/connect", json={}).status_code == 401


def test_disconnect_requires_token():
    assert client.delete("/snaptrade/session").status_code == 401


def test_malformed_authorization_header_rejected():
    r = client.get("/snaptrade/positions", headers={"Authorization": "notbearer x"})
    assert r.status_code == 401


def test_status_is_public_and_reports_availability():
    r = client.get("/snaptrade/status")
    assert r.status_code == 200
    body = r.json()
    assert body["max_tickers"] == st.MAX_TICKERS
    assert isinstance(body["enabled"], bool)   # False when unconfigured, as in CI
