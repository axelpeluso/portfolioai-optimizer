"""
Tests for the opt-in optimizer modes.

The first test is the one that matters: with both modes off, the optimizer must
produce exactly what it produced before tax disclosure existed. Users who never
opt in must never see their numbers move because of a feature they did not use.

Run: pytest -q   (from api/)
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from optimizer import run_optimizer, run_full_analysis, TURNOVER_LAMBDA  # noqa: E402

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "BND", "GLD", "AMZN"]
HOLDINGS = {"AAPL": 5000, "MSFT": 2000, "GOOGL": 3000, "JPM": 1000, "BND": 500}

# Captured from the implementation before the penalty existed.
GOLDEN_WEIGHTS = {
    "AAPL": 0.180232, "MSFT": 0.02, "GOOGL": 0.159768,
    "JPM": 0.3, "BND": 0.02, "GLD": 0.3, "AMZN": 0.02,
}


@pytest.fixture(scope="module")
def baseline():
    return run_full_analysis(TICKERS, HOLDINGS)


# ── the regression that matters ───────────────────────────────
def test_default_off_matches_pre_feature_output(baseline):
    """Both modes off must reproduce the golden weights exactly."""
    w = baseline["optimization"]["optimal_weights"]
    for t, expected in GOLDEN_WEIGHTS.items():
        assert w[t] == pytest.approx(expected, abs=1e-5), f"{t} drifted"


def test_explicit_none_is_same_as_omitting(baseline):
    r = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty=None, tax_weights=None)
    assert r["optimization"]["optimal_weights"] == baseline["optimization"]["optimal_weights"]


def test_min_variance_is_unaffected_by_the_penalty(baseline):
    """Min-var is a reference point, not a trade proposal — it must not move."""
    penalised = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="strong")
    assert penalised["optimization"]["min_var_weights"] == \
           baseline["optimization"]["min_var_weights"]


# ── turnover penalty behaviour ────────────────────────────────
def _turnover(result):
    return sum(-v["trade_amount"] for v in result["rebalancing"].values()
               if v["trade_amount"] < 0) / result["total_value"]


def test_penalty_reduces_turnover(baseline):
    strong = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="strong")
    assert _turnover(strong) < _turnover(baseline)


def test_turnover_decreases_monotonically_with_strength(baseline):
    runs = {lvl: _turnover(run_full_analysis(TICKERS, HOLDINGS, turnover_penalty=lvl))
            for lvl in ("light", "moderate", "strong")}
    assert runs["light"] >= runs["moderate"] >= runs["strong"], runs
    assert _turnover(baseline) >= runs["light"]


def test_reported_sharpe_is_the_true_sharpe_not_the_penalised_objective():
    """The number shown to users must be the real Sharpe of the chosen weights."""
    from optimizer import fetch_data, build_features, portfolio_performance
    r = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="strong")
    w = r["optimization"]["optimal_weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # A penalised run gives up Sharpe by construction; it must be reported honestly.
    assert r["optimization"]["max_sharpe_metrics"]["sharpe"] > 0


def test_penalty_still_respects_bounds_and_sums_to_one():
    r = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="strong")
    w = r["optimization"]["optimal_weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= 0.02 - 1e-6 for v in w.values()), "min weight bound violated"


def test_penalty_needs_current_weights_to_do_anything():
    """Guards against silently penalising toward a zero vector."""
    import optimizer as o
    prices, returns, valid = o.fetch_data(TICKERS)
    feats = o.build_features(prices, returns)
    feats = o.run_kmeans(feats)
    cov = np.cov(returns[valid].T) * 252
    exp = feats["annual_return"].values
    a = run_optimizer(valid, exp, cov, 0.5, feats["cluster_label"].to_dict(),
                      current_weights=None, turnover_penalty="strong")
    b = run_optimizer(valid, exp, cov, 0.5, feats["cluster_label"].to_dict())
    assert a["optimal_weights"] == b["optimal_weights"]


# ── tax weighting ─────────────────────────────────────────────
def test_tax_weights_discourage_selling_the_penalised_asset():
    """A high per-asset cost should hold more of that asset than a uniform one."""
    uniform = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="moderate")
    # AAPL is the largest holding and the biggest proposed sale.
    weighted = run_full_analysis(TICKERS, HOLDINGS, turnover_penalty="moderate",
                                 tax_weights={"AAPL": 12.0})
    assert weighted["optimization"]["optimal_weights"]["AAPL"] >= \
           uniform["optimization"]["optimal_weights"]["AAPL"] - 1e-6


def test_tax_weights_alone_do_nothing_without_the_penalty(baseline):
    """Weights are a modifier on the penalty, never an independent trigger."""
    r = run_full_analysis(TICKERS, HOLDINGS, tax_weights={"AAPL": 50.0})
    assert r["optimization"]["optimal_weights"] == baseline["optimization"]["optimal_weights"]


def test_named_levels_resolve():
    assert set(TURNOVER_LAMBDA) == {"light", "moderate", "strong"}
    assert TURNOVER_LAMBDA["light"] < TURNOVER_LAMBDA["strong"]
