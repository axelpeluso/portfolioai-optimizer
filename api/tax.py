"""
tax.py — tax *disclosure* for rebalancing suggestions.

This module computes facts, never advice, and never a tax bill. The distinction
it is built around:

  fact      "this SELL realizes $1,410 of gain; $820 from lots held under a year"
            — derived entirely from broker-supplied data.
  estimate  "you will owe $340"
            — needs bracket, filing status, state/province and residency.

Facts are always available. A dollar estimate is produced only when the caller
supplies their own rates (see `estimate_tax`), and is labelled illustrative.

The hard rule throughout: **never assert a treatment we cannot verify.** An
unrecognised account type reports "not identified", never "taxable".
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROFILES = BASE_DIR / "tax_profiles.json"

LONG_TERM_DAYS = 366        # US: "more than one year"

# Jurisdictions we model. Anything else falls back to neutral facts.
US, CA, UNKNOWN = "US", "CA", "UNKNOWN"

# Account-type fragments, matched case-insensitively against SnapTrade's
# `raw_type` / `account_category`. These strings vary by brokerage, which is why
# an unmatched value must fall through to UNKNOWN rather than a default.
_US_SHELTERED = ("ROTH", "IRA", "401K", "401(K)", "403B", "457", "SEP", "SIMPLE", "HSA")
_CA_SHELTERED = ("TFSA", "RRSP", "RRIF", "RESP", "LIRA", "FHSA", "DPSP")
_TAXABLE_HINT = ("MARGIN", "CASH", "INDIVIDUAL", "JOINT", "TAXABLE", "NON-REGISTERED",
                 "NONREGISTERED", "BROKERAGE", "TRUST", "CORPORATE")

_profiles: dict | None = None


def profiles() -> dict:
    global _profiles
    if _profiles is None:
        _profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    return _profiles


# ── asset character ───────────────────────────────────────────
def asset_profile(symbol: str, jurisdiction: str = UNKNOWN) -> dict:
    """Descriptive tax character of an instrument. Labels only, no arithmetic."""
    p = profiles()
    cat = p["symbols"].get(symbol.upper(), "equity")
    meta = p["categories"][cat]
    note = meta.get(jurisdiction.lower()) if jurisdiction in (US, CA) else None
    return {
        "category": cat,
        "label":    meta["label"],
        "income":   meta["income"],
        "note":     note,           # None when we have nothing jurisdiction-specific
        "generic":  cat == "equity",
    }


# ── account classification ────────────────────────────────────
def classify_account(account: dict) -> dict:
    """Work out jurisdiction and whether an account is tax-sheltered.

    Returns `sheltered=None` when we genuinely cannot tell — the caller must
    render that as "not identified" rather than assuming it is taxable.
    """
    raw = " ".join(str(account.get(k) or "") for k in
                   ("raw_type", "account_category", "type", "name")).upper()
    ccy = (account.get("currency") or "").upper()

    us_shelter = any(k in raw for k in _US_SHELTERED)
    ca_shelter = any(k in raw for k in _CA_SHELTERED)
    taxable_hint = any(k in raw for k in _TAXABLE_HINT)

    if ca_shelter:
        juris, sheltered = CA, True
    elif us_shelter:
        juris, sheltered = US, True
    elif taxable_hint and ccy in ("USD", "CAD"):
        juris, sheltered = (CA if ccy == "CAD" else US), False
    elif ccy in ("USD", "CAD") and raw.strip():
        # Currency tells us where, the type string tells us nothing usable.
        juris, sheltered = (CA if ccy == "CAD" else US), None
    else:
        juris, sheltered = UNKNOWN, None

    return {
        "jurisdiction": juris,
        "sheltered":    sheltered,
        "currency":     ccy or None,
        "label":        _account_label(juris, sheltered),
        # Holding period only changes treatment under US rules. Canada has no
        # short/long distinction, so showing one there would be wrong.
        "holding_period_matters": juris == US and sheltered is False,
    }


def _account_label(juris: str, sheltered: bool | None) -> str:
    if sheltered is True:
        return "tax-sheltered" + (f" ({juris})" if juris != UNKNOWN else "")
    if sheltered is False:
        return f"taxable ({juris})" if juris != UNKNOWN else "taxable"
    return "account type not identified"


# ── holding period / realized gain ────────────────────────────
def _parse_date(v: Any) -> date | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip().replace("Z", "+00:00")
    for parse in (datetime.fromisoformat,):
        try:
            return parse(s).date()
        except (ValueError, TypeError):
            pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def _lots(position: dict) -> list[dict]:
    out = []
    for lot in (position.get("tax_lots") or []):
        qty   = _num(lot.get("quantity"))
        basis = _num(lot.get("cost_basis"))
        price = _num(lot.get("purchased_price"))
        if basis <= 0 and price > 0:
            basis = price * qty
        d = _parse_date(lot.get("original_purchase_date"))
        if qty > 0:
            out.append({"units": qty, "basis": basis, "date": d,
                        "unit_basis": basis / qty if qty else 0.0})
    return out


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def sale_consequence(position: dict, sell_value: float,
                     asof: date | None = None, method: str = "FIFO") -> dict:
    """What selling `sell_value` of this position realizes.

    Fidelity degrades honestly rather than guessing:
      "lots"    tax_lots present            -> gain AND short/long split
      "average" only average_purchase_price -> total gain, NO split
      "none"    neither                     -> nothing, and we say so
    """
    asof  = asof or datetime.now(timezone.utc).date()
    price = _num(position.get("price"))
    units_held = _num(position.get("units") or position.get("fractional_units"))
    if price <= 0 or sell_value <= 0:
        return {"fidelity": "none", "reason": "no price or nothing to sell"}

    units_to_sell = min(sell_value / price, units_held) if units_held else sell_value / price
    lots = _lots(position)

    if lots:
        # FIFO sells oldest first; HIFO sells the highest-cost lots first, which
        # realizes the least gain. We report, we do not instruct a broker.
        lots = sorted(lots, key=lambda l: (l["date"] or date.min)) if method == "FIFO" \
               else sorted(lots, key=lambda l: -l["unit_basis"])
        remaining, gain, short, long_, unknown_age = units_to_sell, 0.0, 0.0, 0.0, 0.0
        for lot in lots:
            if remaining <= 1e-9:
                break
            take = min(lot["units"], remaining)
            remaining -= take
            g = take * (price - lot["unit_basis"])
            gain += g
            if lot["date"] is None:
                unknown_age += g
            elif (asof - lot["date"]).days >= LONG_TERM_DAYS:
                long_ += g
            else:
                short += g
        return {"fidelity": "lots", "method": method,
                "units_sold": round(units_to_sell, 6),
                "gain": round(gain, 2),
                "short_term": round(short, 2),
                "long_term": round(long_, 2),
                "undated": round(unknown_age, 2)}

    avg = _num(position.get("average_purchase_price"))
    if avg > 0:
        return {"fidelity": "average",
                "units_sold": round(units_to_sell, 6),
                "gain": round(units_to_sell * (price - avg), 2),
                # Deliberately absent: without lots we cannot know the split,
                # and inventing one would be the worst outcome here.
                "short_term": None, "long_term": None}

    return {"fidelity": "none",
            "reason": "cost basis not provided by this brokerage"}


# ── portfolio-level summary ───────────────────────────────────
def portfolio_summary(rebalancing: dict, total_value: float,
                      positions: dict | None = None,
                      accounts: dict | None = None) -> dict:
    """Turnover, and realized gains where basis is known.

    Turnover needs no tax data at all, which is why it is always reported: it is
    the number most likely to make someone pause before acting.
    """
    positions = positions or {}
    accounts  = accounts or {}

    sells = sum(-v["trade_amount"] for v in rebalancing.values()
                if v.get("trade_amount", 0) < 0)
    buys  = sum(v["trade_amount"] for v in rebalancing.values()
                if v.get("trade_amount", 0) > 0)

    gain = short = long_ = 0.0
    known = unknown = 0
    sheltered_only = True

    for sym, row in rebalancing.items():
        amt = row.get("trade_amount", 0)
        if amt >= 0:
            continue
        pos = positions.get(sym)
        if not pos:
            unknown += 1
            continue
        acct = accounts.get(pos.get("_account")) or {}
        cls  = classify_account(acct)
        if cls["sheltered"] is not True:
            sheltered_only = False
        if cls["sheltered"] is True:
            continue                     # no realization event to report
        c = sale_consequence(pos, -amt)
        if c["fidelity"] == "none":
            unknown += 1
            continue
        known += 1
        gain += c["gain"]
        if c.get("short_term") is not None:
            short += c["short_term"]
            long_ += c["long_term"]

    return {
        "turnover_pct":  round(sells / total_value * 100, 1) if total_value else 0.0,
        "sell_total":    round(sells, 2),
        "buy_total":     round(buys, 2),
        "realized_gain": round(gain, 2) if known else None,
        "short_term":    round(short, 2) if known and short else None,
        "long_term":     round(long_, 2) if known and long_ else None,
        "positions_with_basis":    known,
        "positions_without_basis": unknown,
        "all_sheltered": sheltered_only and known == 0 and unknown == 0,
    }


# ── opt-in estimator ──────────────────────────────────────────
def estimate_tax(short_term: float, long_term: float, rates: dict | None) -> dict | None:
    """Illustrative figure using the CALLER'S OWN rates.

    Returns None when no rates are supplied. There are deliberately no default
    rates in this module: a rate we invented is the single thing most likely to
    mislead someone into a real financial decision.
    """
    if not rates:
        return None
    st = _num(rates.get("short_term_rate"))
    lt = _num(rates.get("long_term_rate"))
    inclusion = rates.get("inclusion_rate")

    if inclusion is not None:                      # Canada-style
        inc = _num(inclusion)
        marginal = _num(rates.get("marginal_rate"))
        taxable = (short_term + long_term) * inc
        return {"basis": "inclusion", "taxable_amount": round(taxable, 2),
                "estimate": round(taxable * marginal, 2), "illustrative": True}

    return {"basis": "us_split",
            "estimate": round(short_term * st + long_term * lt, 2),
            "illustrative": True}


# ── tax-aware penalty weights ─────────────────────────────────
def penalty_weights(positions: dict, accounts: dict,
                    rates: dict | None, scale: float = 8.0) -> dict:
    """Per-asset multipliers for the optimizer's turnover penalty.

    Selling an asset carrying a large embedded gain in a taxable account should
    cost the optimizer more than selling one at a loss. Returns {} when we lack
    the inputs — which makes tax-aware mode a no-op rather than a guess.

    Weight 1.0 = no different from ordinary turnover. Above 1.0 = discourage
    selling. Below 1.0 = a loss position, mildly encouraged (harvesting).
    """
    if not rates:
        return {}

    st = _num(rates.get("short_term_rate"))
    lt = _num(rates.get("long_term_rate"))
    if rates.get("inclusion_rate") is not None:
        eff = _num(rates.get("inclusion_rate")) * _num(rates.get("marginal_rate"))
        st = lt = eff

    out: dict[str, float] = {}
    for sym, pos in (positions or {}).items():
        acct = (accounts or {}).get(pos.get("_account")) or {}
        if classify_account(acct)["sheltered"] is not False:
            continue                      # sheltered or unknown: no tax cost to model

        value = _num(pos.get("price")) * _num(pos.get("units"))
        if value <= 0:
            continue
        c = sale_consequence(pos, value)          # cost of liquidating it entirely
        if c["fidelity"] == "none":
            continue

        gain = c.get("gain", 0.0)
        if c.get("short_term") is not None:
            cost = c["short_term"] * st + c["long_term"] * lt
        else:
            cost = gain * lt                      # no split known: use the milder rate
        # Normalised by position size so the weight is a rate, not a dollar amount.
        out[sym] = max(0.1, 1.0 + scale * (cost / value))
    return out
