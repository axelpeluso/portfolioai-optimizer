"""
Tests for tax disclosure.

The properties that matter most here are the negative ones: that we never
invent a holding-period split we cannot know, and never call an account taxable
when we could not identify it. Pure functions, no network. Run: pytest -q
"""
from datetime import date, timedelta

import tax


TODAY = date(2026, 8, 27)


def lot(units, unit_cost, days_ago):
    return {"quantity": units,
            "cost_basis": units * unit_cost,
            "purchased_price": unit_cost,
            "original_purchase_date": (TODAY - timedelta(days=days_ago)).isoformat()}


def position(price=100.0, units=100, lots=None, avg=None, account="a1"):
    p = {"price": price, "units": units, "_account": account}
    if lots is not None:
        p["tax_lots"] = lots
    if avg is not None:
        p["average_purchase_price"] = avg
    return p


# ── fidelity: never dress up what we do not know ──────────────
def test_lots_give_gain_and_holding_split():
    p = position(price=100.0, units=100,
                 lots=[lot(50, 40.0, 800),     # long-term, $60/unit gain
                       lot(50, 90.0, 100)])    # short-term, $10/unit gain
    c = tax.sale_consequence(p, sell_value=10000.0, asof=TODAY)
    assert c["fidelity"] == "lots"
    assert c["gain"] == 3500.0                 # 50*60 + 50*10
    assert c["long_term"] == 3000.0
    assert c["short_term"] == 500.0


def test_average_cost_gives_gain_but_no_split():
    """The important negative: a total without a split must stay without one."""
    c = tax.sale_consequence(position(price=100.0, units=100, avg=60.0),
                             sell_value=5000.0, asof=TODAY)
    assert c["fidelity"] == "average"
    assert c["gain"] == 2000.0
    assert c["short_term"] is None and c["long_term"] is None


def test_no_basis_reports_unavailable_rather_than_guessing():
    c = tax.sale_consequence(position(price=100.0, units=100), sell_value=5000.0)
    assert c["fidelity"] == "none"
    assert "not provided" in c["reason"]
    assert "gain" not in c


def test_one_year_boundary():
    """366 days is long-term; 365 is not."""
    lt = tax.sale_consequence(position(lots=[lot(10, 50.0, 366)]), 1000.0, asof=TODAY)
    st = tax.sale_consequence(position(lots=[lot(10, 50.0, 365)]), 1000.0, asof=TODAY)
    assert lt["long_term"] == 500.0 and lt["short_term"] == 0.0
    assert st["short_term"] == 500.0 and st["long_term"] == 0.0


def test_undated_lots_are_segregated_not_assumed():
    p = position(price=100.0, units=20,
                 lots=[{"quantity": 20, "cost_basis": 1000.0,
                        "original_purchase_date": None}])
    c = tax.sale_consequence(p, 2000.0, asof=TODAY)
    assert c["undated"] == 1000.0
    assert c["short_term"] == 0.0 and c["long_term"] == 0.0


def test_hifo_realizes_less_gain_than_fifo():
    lots = [lot(50, 10.0, 900), lot(50, 90.0, 900)]
    p = position(price=100.0, units=100, lots=lots)
    fifo = tax.sale_consequence(p, 5000.0, asof=TODAY, method="FIFO")
    hifo = tax.sale_consequence(p, 5000.0, asof=TODAY, method="HIFO")
    assert fifo["gain"] == 4500.0     # sells the $10 lot
    assert hifo["gain"] == 500.0      # sells the $90 lot
    assert hifo["gain"] < fifo["gain"]


def test_partial_sale_only_counts_units_sold():
    p = position(price=100.0, units=100, lots=[lot(100, 50.0, 900)])
    c = tax.sale_consequence(p, sell_value=2500.0, asof=TODAY)
    assert c["units_sold"] == 25.0
    assert c["gain"] == 1250.0


# ── lot method: the assumption must be explicit and jurisdiction-correct ──
SPLIT_LOTS = [lot(50, 20.0, 900), lot(50, 90.0, 100)]   # cheap+old, dear+new


def test_method_dominates_a_partial_sale():
    """Same trade, different method, several-fold difference — hence disclosure."""
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)
    fifo = tax.sale_consequence(p, 5000.0, asof=TODAY, method="FIFO")
    lifo = tax.sale_consequence(p, 5000.0, asof=TODAY, method="LIFO")
    hifo = tax.sale_consequence(p, 5000.0, asof=TODAY, method="HIFO")
    assert fifo["gain"] == 4000.0
    assert lifo["gain"] == 500.0
    assert hifo["gain"] == 500.0
    assert fifo["gain"] > hifo["gain"]


def test_every_method_agrees_on_a_full_exit():
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)
    gains = {m: tax.sale_consequence(p, 10000.0, asof=TODAY, method=m)["gain"]
             for m in ("FIFO", "LIFO", "HIFO", "ACB")}
    assert len(set(gains.values())) == 1, gains


def test_canada_is_forced_to_acb():
    """FIFO/LIFO/HIFO are not permitted for Canadian tax purposes."""
    m, forced = tax.resolve_method(tax.CA, "HIFO")
    assert (m, forced) == ("ACB", True)


def test_acb_uses_the_weighted_average():
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)   # avg cost $55
    c = tax.sale_consequence(p, 5000.0, asof=TODAY, method="ACB")
    assert c["gain"] == 2250.0          # 50 * (100 - 55)
    assert c["method"] == "ACB"
    assert c["short_term"] is None, "Canada has no holding-period split"


def test_us_honours_the_requested_method_and_defaults_to_fifo():
    assert tax.resolve_method(tax.US, "HIFO") == ("HIFO", False)
    assert tax.resolve_method(tax.US, None) == ("FIFO", False)
    assert tax.resolve_method(tax.US, "nonsense") == ("FIFO", False)


def test_range_reports_the_spread_where_methods_disagree():
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)
    r = tax.sale_range(p, 5000.0, tax.US, asof=TODAY)
    assert r["low"] == 500.0 and r["high"] == 4000.0
    assert r["high_method"] == "FIFO"


def test_no_range_on_a_full_exit():
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)
    assert tax.sale_range(p, 10000.0, tax.US, asof=TODAY) is None


def test_no_range_for_canada():
    p = position(price=100.0, units=100, lots=SPLIT_LOTS)
    assert tax.sale_range(p, 5000.0, tax.CA, asof=TODAY) is None


def test_portfolio_summary_uses_acb_for_canadian_accounts():
    """The bug this fixes: a FIFO figure summed into a Canadian total."""
    positions = {"AAPL": position(price=100.0, units=100, lots=SPLIT_LOTS)}
    rebal = {"AAPL": {"trade_amount": -5000.0}}
    ca = tax.portfolio_summary(rebal, 10000.0, positions,
                               {"a1": {"raw_type": "CASH", "currency": "CAD"}})
    us = tax.portfolio_summary(rebal, 10000.0, positions,
                               {"a1": {"raw_type": "MARGIN", "currency": "USD"}})
    assert ca["realized_gain"] == 2250.0, "Canada must use ACB"
    assert us["realized_gain"] == 4000.0, "US defaults to FIFO"


# ── account classification: unknown must stay unknown ─────────
def test_us_sheltered():
    c = tax.classify_account({"raw_type": "ROTH IRA", "currency": "USD"})
    assert (c["jurisdiction"], c["sheltered"]) == (tax.US, True)
    assert c["holding_period_matters"] is False


def test_us_taxable_gets_holding_period():
    c = tax.classify_account({"raw_type": "MARGIN", "currency": "USD"})
    assert (c["jurisdiction"], c["sheltered"]) == (tax.US, False)
    assert c["holding_period_matters"] is True


def test_canadian_sheltered():
    c = tax.classify_account({"raw_type": "TFSA", "currency": "CAD"})
    assert (c["jurisdiction"], c["sheltered"]) == (tax.CA, True)


def test_canadian_taxable_has_no_holding_period_distinction():
    """Canada has no short/long split — showing one would be wrong."""
    c = tax.classify_account({"raw_type": "CASH", "currency": "CAD"})
    assert (c["jurisdiction"], c["sheltered"]) == (tax.CA, False)
    assert c["holding_period_matters"] is False


def test_unrecognised_type_never_claims_taxable():
    """The safety property: silence beats a wrong assertion."""
    c = tax.classify_account({"raw_type": "SOMETHING WEIRD", "currency": "USD"})
    assert c["sheltered"] is None
    assert c["label"] == "account type not identified"
    assert c["holding_period_matters"] is False


def test_empty_account_is_unknown():
    c = tax.classify_account({})
    assert c["jurisdiction"] == tax.UNKNOWN and c["sheltered"] is None


# ── asset character ───────────────────────────────────────────
def test_symbols_pattern_matching_missed_are_classified():
    """VNQ/XLRE and LQD/EMB are exactly the ones a name regex got wrong."""
    for s in ("VNQ", "XLRE"):
        assert tax.asset_profile(s)["category"] == "reit"
    for s in ("LQD", "EMB"):
        assert tax.asset_profile(s)["category"] == "bond"


def test_false_positives_stay_equity():
    """MA/QCOM/TXN/UNH matched 'Incorporated' in a naive bond regex."""
    for s in ("MA", "QCOM", "TXN", "UNH"):
        assert tax.asset_profile(s)["category"] == "equity"


def test_gold_is_a_collectible_in_the_us():
    p = tax.asset_profile("GLD", tax.US)
    assert p["category"] == "precious_metal"
    assert "collectible" in p["note"].lower()


def test_unknown_symbol_defaults_to_equity_and_is_flagged_generic():
    p = tax.asset_profile("ZZZZ")
    assert p["category"] == "equity" and p["generic"] is True


def test_no_jurisdiction_note_when_jurisdiction_unknown():
    assert tax.asset_profile("GLD", tax.UNKNOWN)["note"] is None


# ── portfolio summary ─────────────────────────────────────────
REBAL = {"AAPL": {"trade_amount": -5000.0}, "JPM": {"trade_amount": 5000.0}}


def test_turnover_needs_no_tax_data():
    s = tax.portfolio_summary(REBAL, total_value=10000.0)
    assert s["turnover_pct"] == 50.0
    assert s["realized_gain"] is None            # nothing known, nothing claimed


def test_summary_counts_gains_only_in_taxable_accounts():
    positions = {"AAPL": position(price=100.0, units=100,
                                  lots=[lot(100, 50.0, 900)], account="a1")}
    taxable   = {"a1": {"raw_type": "MARGIN", "currency": "USD"}}
    sheltered = {"a1": {"raw_type": "ROTH IRA", "currency": "USD"}}
    s1 = tax.portfolio_summary(REBAL, 10000.0, positions, taxable)
    s2 = tax.portfolio_summary(REBAL, 10000.0, positions, sheltered)
    assert s1["realized_gain"] == 2500.0
    assert s2["realized_gain"] is None, "sheltered account realizes nothing to report"


def test_summary_reports_how_many_positions_lacked_basis():
    positions = {"AAPL": position(price=100.0, units=100, account="a1")}
    s = tax.portfolio_summary(REBAL, 10000.0, positions,
                              {"a1": {"raw_type": "MARGIN", "currency": "USD"}})
    assert s["positions_without_basis"] == 1
    assert s["realized_gain"] is None


# ── estimator ─────────────────────────────────────────────────
def test_no_estimate_without_rates():
    assert tax.estimate_tax(1000.0, 2000.0, None) is None
    assert tax.estimate_tax(1000.0, 2000.0, {}) is None


def test_us_estimate_uses_supplied_rates():
    e = tax.estimate_tax(1000.0, 2000.0,
                         {"short_term_rate": 0.35, "long_term_rate": 0.15})
    assert e["estimate"] == 650.0 and e["illustrative"] is True


def test_canadian_estimate_uses_inclusion_rate():
    e = tax.estimate_tax(0.0, 4000.0,
                         {"inclusion_rate": 0.5, "marginal_rate": 0.40})
    assert e["taxable_amount"] == 2000.0 and e["estimate"] == 800.0


def test_module_ships_no_default_rates():
    """Guards the rule that we never invent a rate for someone."""
    src = open(tax.__file__, encoding="utf-8").read()
    for bad in ("0.15", "0.20", "0.37", "DEFAULT_RATE"):
        assert bad not in src, f"{bad} looks like a hardcoded tax rate"
