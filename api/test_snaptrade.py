"""
Tests for the SnapTrade read-only import.

No network and no SnapTrade SDK required: reconciliation is pure, and the route
tests only need to prove the auth gate rejects unauthenticated callers before
any brokerage call is attempted. Run with:  pytest -q   (from api/)
"""
import pytest
from fastapi.testclient import TestClient

import main
import brokerage as st


# ── SDK contract ──────────────────────────────────────────────
def test_client_construction_matches_the_installed_sdk(monkeypatch):
    """Construir el cliente de verdad, no un mock.

    v13 exige `auth=SnapTradeAuth.commercial_api_key(...)`; pasar
    client_id/consumer_key sueltos lanza TypeError. Todos los tests con mocks
    pasaban igual, y el fallo solo aparecio en la primera ejecucion real.
    """
    pytest.importorskip("snaptrade_client")
    monkeypatch.setenv("SNAPTRADE_CLIENT_ID", "TEST-ID")
    monkeypatch.setenv("SNAPTRADE_CONSUMER_KEY", "k" * 50)
    monkeypatch.setattr(st, "_client", None)      # evitar el cache del modulo

    client = st.client()
    assert hasattr(client, "authentication")
    assert hasattr(client, "account_information")
    # Los metodos que la app llama tienen que existir con estos nombres exactos.
    for name in ("register_snap_trade_user", "delete_snap_trade_user",
                 "login_snap_trade_user"):
        assert hasattr(client.authentication, name), name
    for name in ("list_user_accounts", "get_all_account_positions",
                 "get_user_account_balance"):
        assert hasattr(client.account_information, name), name
    st._client = None

client = TestClient(main.app)

REAL = "AAPL"      # in tickers.json
REAL2 = "MSFT"


def pos(symbol, units=10, price=100.0, kind="cs", market_value=None, account=None):
    """A SnapTrade-shaped position (nested symbol object)."""
    p = {
        "symbol": {"symbol": {"symbol": symbol, "type": {"code": kind}}},
        "units": units,
        "price": price,
        "market_value": market_value if market_value is not None else units * price,
    }
    if account:
        p["_account"] = account
    return p


def acct(aid, name, value=10000.0, currency="USD"):
    return {"id": aid, "name": name, "broker": "TestBroker",
            "value": value, "currency": currency}


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


# ── forma real del payload (capturada del sandbox) ────────────
# Estos cuatro tests corresponden a bugs reales que ningun mock detecto: el
# codigo asumia una lista de posiciones con el ticker bajo `symbol` y el costo
# bajo `average_purchase_price`. Nada de eso era cierto.

LIVE_PAYLOAD = {
    "results": [
        {"instrument": {"kind": "stock", "symbol": "AAPL", "raw_symbol": "AAPL",
                        "description": "Apple Inc.", "currency": "USD"},
         "units": "5", "price": "180.5", "cost_basis": "175", "currency": "USD"},
        {"instrument": {"kind": "stock", "symbol": "MSFT", "raw_symbol": "MSFT"},
         "units": "10", "price": "410", "cost_basis": "380"},
        {"instrument": {"kind": "crypto", "symbol": "BTC", "raw_symbol": "BTC"},
         "units": "0.25", "price": "59000", "cost_basis": "55000"},
    ],
    "data_freshness": {},
}

LIVE_ACCOUNTS = [
    {"id": "acc1", "name": "Individual", "broker": "sandbox", "value": 25000.0,
     "currency": "USD", "raw_type": "Individual", "account_category": "INVESTMENT"},
    {"id": "acc2", "name": "IRA", "broker": "sandbox", "value": 12500.0,
     "currency": "USD", "raw_type": "IRA", "account_category": "INVESTMENT"},
]


def live_positions(account="acc1"):
    return [dict(p, _account=account) for p in st._rows(LIVE_PAYLOAD)]


def test_response_is_a_dict_not_a_list():
    """Iterar el dict devolvia sus claves (strings) y reventaba en .get()."""
    rows = st._rows(LIVE_PAYLOAD)
    assert len(rows) == 3
    assert all(isinstance(r, dict) for r in rows)


def test_ticker_lives_under_instrument_not_symbol():
    """El bug mas silencioso: todo caia a None y no se importaba nada."""
    rows = st._rows(LIVE_PAYLOAD)
    assert [st._symbol_of(r)[0] for r in rows] == ["AAPL", "MSFT", "BTC"]
    assert [st._symbol_of(r)[1] for r in rows] == ["stock", "stock", "crypto"]


def test_cost_basis_is_read_when_average_purchase_price_is_absent():
    """El dato estaba, con otro nombre; buscabamos solo el que no venia."""
    out = st.reconcile(live_positions(), accounts=LIVE_ACCOUNTS)
    aapl = next(p for p in out["supported"] if p["symbol"] == "AAPL")
    assert aapl["tax"]["average_purchase_price"] == 175.0
    assert aapl["value"] == 902.5


def test_crypto_is_excluded_by_instrument_kind():
    out = st.reconcile(live_positions(), accounts=LIVE_ACCOUNTS)
    assert {p["symbol"] for p in out["supported"]} == {"AAPL", "MSFT"}
    assert [(u["symbol"], u["reason"]) for u in out["unsupported"]] == \
           [("BTC", st.REASON_KIND)]


def test_live_shape_produces_a_usable_gain():
    """De punta a punta: payload real -> reconcile -> consecuencia fiscal."""
    import tax
    out = st.reconcile(live_positions(), accounts=LIVE_ACCOUNTS)
    aapl = next(p for p in out["supported"] if p["symbol"] == "AAPL")["tax"]
    c = tax.sale_consequence({"price": aapl["price"], "units": aapl["units"],
                              "average_purchase_price": aapl["average_purchase_price"]},
                             451.25)
    assert c["fidelity"] == "average"        # el sandbox no manda tax_lots
    assert c["gain"] == 13.75                # 2.5 un * (180.5 - 175)
    assert c["short_term"] is None           # sin lotes, sin reparto inventado


def test_real_account_types_classify_correctly():
    """raw_type 'Individual' e 'IRA' salieron de una conexion real."""
    import tax
    individual = tax.classify_account(LIVE_ACCOUNTS[0])
    ira = tax.classify_account(LIVE_ACCOUNTS[1])
    assert (individual["jurisdiction"], individual["sheltered"]) == (tax.US, False)
    assert individual["holding_period_matters"] is True
    assert (ira["jurisdiction"], ira["sheltered"]) == (tax.US, True)


# ── multi-account ─────────────────────────────────────────────
ACCTS = [acct("a1", "Margin", 30000.0), acct("a2", "RRSP", 20000.0)]


def test_split_across_accounts_sums_to_the_total():
    out = st.reconcile([pos(REAL, units=24, price=310.0, account="a1"),
                        pos(REAL, units=16, price=310.0, account="a2")],
                       accounts=ACCTS)
    row = out["supported"][0]
    assert len(row["accounts"]) == 2
    assert round(sum(a["value"] for a in row["accounts"]), 2) == row["value"]
    assert round(sum(a["units"] for a in row["accounts"]), 6) == row["units"]


def test_shares_sum_to_one():
    """Guards the pro-rata display split in the results table."""
    out = st.reconcile([pos(REAL, units=6, price=100.0, account="a1"),
                        pos(REAL, units=4, price=100.0, account="a2")],
                       accounts=ACCTS)
    shares = [a["share"] for a in out["supported"][0]["accounts"]]
    assert round(sum(shares), 6) == 1.0
    assert sorted(shares, reverse=True) == [0.6, 0.4]


def test_accounts_are_named_and_ranked_by_value():
    out = st.reconcile([pos(REAL, units=2, price=100.0, account="a2"),
                        pos(REAL, units=8, price=100.0, account="a1")],
                       accounts=ACCTS)
    names = [a["name"] for a in out["supported"][0]["accounts"]]
    assert names == ["Margin", "RRSP"], "largest holding first"


def test_single_account_still_reports_one_entry():
    out = st.reconcile([pos(REAL, account="a1")], accounts=ACCTS)
    row = out["supported"][0]
    assert len(row["accounts"]) == 1
    assert row["accounts"][0]["share"] == 1.0


def test_account_scoping_excludes_other_accounts():
    out = st.reconcile([pos(REAL, units=10, price=100.0, account="a1"),
                        pos(REAL2, units=10, price=100.0, account="a2")],
                       accounts=ACCTS, account_ids=["a1"])
    assert [p["symbol"] for p in out["supported"]] == [REAL]
    assert out["total_value"] == 1000.0


def test_scoping_recomputes_the_top_n():
    """Narrowing to one account must re-rank, not reuse the union's selection."""
    universe = sorted(st.universe())[:20]
    raw = [pos(s, units=1, price=float(100 - i),
               account="a1" if i >= 5 else "a2")
           for i, s in enumerate(universe)]
    scoped = st.reconcile(raw, accounts=ACCTS, account_ids=["a1"])
    assert len(scoped["supported"]) == 15
    assert all(p["selected"] for p in scoped["supported"]), \
        "15 positions in scope should all fit under the cap"


def test_foreign_currency_account_is_excluded_not_converted():
    mixed = [acct("a1", "USD Margin", 30000.0, "USD"),
             acct("a2", "CAD TFSA",   20000.0, "CAD")]
    out = st.reconcile([pos(REAL,  units=10, price=100.0, account="a1"),
                        pos(REAL2, units=10, price=100.0, account="a2")],
                       accounts=mixed)
    assert out["currency"] == "USD", "base is the largest account's currency"
    assert [p["symbol"] for p in out["supported"]] == [REAL]
    assert out["total_value"] == 1000.0, "CAD value must not be summed in"
    bad = [u for u in out["unsupported"] if u["reason"] == st.REASON_CURRENCY]
    assert [u["symbol"] for u in bad] == [REAL2]
    assert any("not in USD" in n for n in out["notes"])


def test_untagged_positions_still_work():
    """Back-compat: positions with no _account tag (older callers, tests)."""
    out = st.reconcile([pos(REAL), pos(REAL2)])
    assert len(out["supported"]) == 2
    assert all(len(p["accounts"]) == 1 for p in out["supported"])


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
