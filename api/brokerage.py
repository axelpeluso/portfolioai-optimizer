"""
brokerage.py — SnapTrade wrapper + position reconciliation.

Read-only. This module registers users, hands back connection-portal URLs, and
reads accounts/positions. It never places a trade.

The SDK is imported lazily (same pattern as the anthropic/supabase helpers in
main.py) so the API still boots and the existing test suite still runs when
snaptrade is not installed.

Reconciliation lives here rather than in the frontend because both constraints
it enforces are server-side facts: the 288-symbol universe in tickers.json, and
the 15-ticker cap on /optimize.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BASE_DIR      = Path(__file__).resolve().parent
TICKERS_JSON  = BASE_DIR / "tickers.json"

MAX_TICKERS   = 15      # must match the cap enforced by POST /optimize

# Why a position could not be optimized. Surfaced verbatim in the UI — a user
# must never be silently optimized against a portfolio that is not theirs.
REASON_UNIVERSE = "not in the instrument universe"
REASON_OVERFLOW = f"below the top {MAX_TICKERS} by value"
REASON_KIND     = "not a modellable equity/ETF position"
REASON_NOVALUE  = "no market value reported"
REASON_CURRENCY = "held in a different currency"


class SnapTradeNotConfigured(RuntimeError):
    """Raised when the SnapTrade credentials or SDK are missing."""


# ── universe ──────────────────────────────────────────────────
_universe: set[str] | None = None


def universe() -> set[str]:
    """Symbols we hold price history for. Cached after first read."""
    global _universe
    if _universe is None:
        data = json.loads(TICKERS_JSON.read_text(encoding="utf-8"))
        _universe = set(data.keys() if isinstance(data, dict) else data)
    return _universe


# ── SDK ───────────────────────────────────────────────────────
_client = None


def client():
    """Return a cached SnapTrade SDK client, or raise SnapTradeNotConfigured."""
    global _client
    if _client is not None:
        return _client

    client_id    = os.environ.get("SNAPTRADE_CLIENT_ID")
    consumer_key = os.environ.get("SNAPTRADE_CONSUMER_KEY")
    if not client_id or not consumer_key:
        raise SnapTradeNotConfigured(
            "SnapTrade not configured (set SNAPTRADE_CLIENT_ID and "
            "SNAPTRADE_CONSUMER_KEY)."
        )
    try:
        from snaptrade_client import SnapTrade, SnapTradeAuth
    except ImportError:
        raise SnapTradeNotConfigured(
            "snaptrade-python-sdk not installed. Add it to requirements.txt."
        )
    # v13 takes an auth object; passing client_id/consumer_key directly raises
    # TypeError (they survive in the signature only as deprecated None-typed
    # placeholders). Caught by the first real run, not by any mocked test.
    _client = SnapTrade(auth=SnapTradeAuth.commercial_api_key(
        client_id=client_id, consumer_key=consumer_key))
    return _client


def is_configured() -> bool:
    """True when a connection could be attempted — used to gate the UI."""
    try:
        client()
        return True
    except SnapTradeNotConfigured:
        return False


# ── user lifecycle ────────────────────────────────────────────
def register_user(user_id: str) -> str:
    """Register a SnapTrade user and return its userSecret."""
    res = client().authentication.register_snap_trade_user(body={"userId": user_id})
    body = _body(res)
    secret = body.get("userSecret")
    if not secret:
        raise RuntimeError("SnapTrade did not return a userSecret")
    return secret


def delete_user(user_id: str) -> None:
    """Delete a SnapTrade user, dropping every brokerage connection it holds.

    Called on session expiry and explicit disconnect. Connections are billed
    per user, so failing to do this costs money, not just tidiness.
    """
    client().authentication.delete_snap_trade_user(user_id=user_id)


def login_url(user_id: str, user_secret: str,
              redirect_uri: str | None = None) -> str:
    """URL of SnapTrade's hosted connection portal for this user."""
    kwargs: dict[str, Any] = {"user_id": user_id, "user_secret": user_secret}
    if redirect_uri:
        kwargs["custom_redirect"] = redirect_uri
    body = _body(client().authentication.login_snap_trade_user(**kwargs))
    url = body.get("redirectURI") or body.get("redirectUri")
    if not url:
        raise RuntimeError("SnapTrade did not return a portal URL")
    return url


# ── reads ─────────────────────────────────────────────────────
def list_accounts(user_id: str, user_secret: str) -> list[dict]:
    res = client().account_information.list_user_accounts(
        user_id=user_id, user_secret=user_secret
    )
    out = []
    for a in _body(res) or []:
        out.append({
            "id":      a.get("id"),
            "name":    a.get("name") or a.get("number") or "Account",
            "broker":  (a.get("institution_name") or a.get("brokerage_authorization")
                        or "Brokerage"),
            "value":   _num((a.get("balance") or {}).get("total", {}).get("amount")),
            "currency": ((a.get("balance") or {}).get("total", {}) or {}).get("currency"),
            # Carried for tax disclosure: these decide jurisdiction and whether
            # the account is sheltered. Unrecognised values must stay
            # unrecognised — see tax.classify_account.
            "raw_type":         a.get("raw_type"),
            "account_category": a.get("account_category"),
        })
    return out


def list_positions(user_id: str, user_secret: str, account_id: str) -> list[dict]:
    """Positions for one account, each tagged with the account it came from.

    The tag is what lets reconcile() say *where* a holding sits. Without it a
    symbol held in two accounts merges into one number and the resulting trade
    instruction has nowhere to point.
    """
    res = client().account_information.get_all_account_positions(
        user_id=user_id, user_secret=user_secret, account_id=account_id
    )
    out = []
    for p in (_body(res) or []):
        p = dict(p)
        p["_account"] = account_id
        out.append(p)
    return out


def account_cash(user_id: str, user_secret: str, account_id: str) -> float:
    """Cash balance, which the optimizer cannot model as a position."""
    try:
        res = client().account_information.get_user_account_balance(
            user_id=user_id, user_secret=user_secret, account_id=account_id
        )
        return sum(_num(b.get("cash")) for b in (_body(res) or []))
    except Exception:      # noqa: BLE001 — cash is informational, never fatal
        return 0.0


# ── normalisation ─────────────────────────────────────────────
def _body(res: Any) -> Any:
    """SDK responses expose .body; dicts/lists pass through (eases testing)."""
    return getattr(res, "body", res)


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _symbol_of(pos: dict) -> tuple[str | None, str | None]:
    """Extract (symbol, type) from SnapTrade's nested symbol shapes.

    The payload nests differently across brokerages, hence the walk rather than
    a fixed path: position.symbol may be the symbol object itself, or wrap
    another under .symbol.
    """
    sym = pos.get("symbol")
    if isinstance(sym, dict):
        inner = sym.get("symbol")
        if isinstance(inner, dict):
            sym = inner
    if not isinstance(sym, dict):
        return (str(sym).upper() if sym else None), None

    raw = sym.get("symbol") or sym.get("raw_symbol") or sym.get("ticker")
    kind = ((sym.get("type") or {}).get("code")
            if isinstance(sym.get("type"), dict) else sym.get("type"))
    return (str(raw).upper().strip() if raw else None), (kind or None)


# Instrument types the price history cannot support.
_UNSUPPORTED_KINDS = {"crypto", "opt", "option", "fx", "forex", "bnd", "bond"}


def _base_currency(accounts: list[dict]) -> str | None:
    """Currency of the largest account, used as the reporting currency.

    We hold no FX rates, and inventing a conversion would be worse than
    declining — a wrong total is harder to notice than a stated exclusion.
    """
    with_ccy = [a for a in accounts if a.get("currency")]
    if not with_ccy:
        return None
    return max(with_ccy, key=lambda a: _num(a.get("value"))).get("currency")


def reconcile(raw_positions: list[dict], cash: float = 0.0,
              accounts: list[dict] | None = None,
              account_ids: list[str] | None = None) -> dict:
    """Split broker positions into what /optimize can and cannot accept.

    Positions are consolidated across accounts because Modern Portfolio Theory
    operates on total exposure — the union of accounts is the real portfolio.
    Each row keeps a per-account breakdown so the resulting trade can still say
    *where* the holding sits, which is what makes the output executable.

    `account_ids` scopes the run to a subset (None = every account).
    """
    known    = universe()
    accounts = accounts or []
    by_id    = {a.get("id"): a for a in accounts if a.get("id")}

    wanted = set(account_ids) if account_ids is not None else None
    scoped = [a for a in accounts
              if wanted is None or a.get("id") in wanted]

    base_ccy  = _base_currency(scoped)
    wrong_ccy = {a["id"] for a in scoped
                 if a.get("currency") and base_ccy and a["currency"] != base_ccy}

    def acct_name(aid):
        a = by_id.get(aid) or {}
        return a.get("name") or a.get("broker") or "Account"

    supported: list[dict] = []
    unsupported: list[dict] = []

    for pos in raw_positions:
        aid = pos.get("_account")
        if wanted is not None and aid is not None and aid not in wanted:
            continue                      # out of scope for this run

        symbol, kind = _symbol_of(pos)
        units = _num(pos.get("units") or pos.get("fractional_units"))
        price = _num(pos.get("price"))
        value = _num(pos.get("market_value")) or (units * price)

        if not symbol:
            continue
        if aid in wrong_ccy:
            unsupported.append({"symbol": symbol, "value": round(value, 2),
                                "reason": REASON_CURRENCY,
                                "account": acct_name(aid)})
        elif kind and str(kind).lower() in _UNSUPPORTED_KINDS:
            unsupported.append({"symbol": symbol, "value": round(value, 2),
                                "reason": REASON_KIND, "account": acct_name(aid)})
        elif symbol not in known:
            unsupported.append({"symbol": symbol, "value": round(value, 2),
                                "reason": REASON_UNIVERSE, "account": acct_name(aid)})
        elif value <= 0:
            unsupported.append({"symbol": symbol, "value": 0.0,
                                "reason": REASON_NOVALUE, "account": acct_name(aid)})
        else:
            supported.append({"symbol": symbol, "units": units,
                              "value": value, "_account": aid,
                              # Cost basis for tax disclosure. Already in the
                              # payload — previously fetched and discarded.
                              "price": price or (value / units if units else 0.0),
                              "average_purchase_price": _num(pos.get("average_purchase_price")),
                              "open_pnl": _num(pos.get("open_pnl")),
                              "tax_lots": pos.get("tax_lots") or []})

    # Merge duplicates, keeping the per-account split rather than discarding it.
    merged: dict[str, dict] = {}
    for p in supported:
        m = merged.setdefault(p["symbol"], {"symbol": p["symbol"], "units": 0.0,
                                            "value": 0.0, "_acct": {}, "_lots": [],
                                            "_basis": 0.0, "_price": 0.0})
        m["units"] += p["units"]
        m["value"] += p["value"]
        m["_price"] = p["price"] or m["_price"]
        # Lots carry their own account so a mixed-shelter holding can be
        # reported as mixed rather than silently attributed to one treatment.
        for lot in p["tax_lots"]:
            m["_lots"].append({**lot, "_account": p["_account"]})
        m["_basis"] += (p["average_purchase_price"] or 0.0) * p["units"]
        aid = p["_account"]
        a = m["_acct"].setdefault(aid, {"id": aid, "name": acct_name(aid),
                                        "units": 0.0, "value": 0.0})
        a["units"] += p["units"]
        a["value"] += p["value"]

    supported = sorted(merged.values(), key=lambda p: p["value"], reverse=True)
    for p in supported:
        total = p["value"] or 1.0
        # Everything the tax module needs, in one place, so the client can hand
        # it straight back to /optimize without reshaping it.
        lots  = p.pop("_lots")
        basis = p.pop("_basis")
        price = p.pop("_price") or (p["value"] / p["units"] if p["units"] else 0.0)
        p["tax"] = {
            "price":    round(price, 6),
            "units":    round(p["units"], 6),
            "tax_lots": lots,
            "average_purchase_price": round(basis / p["units"], 6) if p["units"] and basis else 0.0,
            "accounts": sorted({str(a) for a in p["_acct"]}),
        }
        p["accounts"] = sorted(
            ({"id": a["id"], "name": a["name"],
              "units": round(a["units"], 6), "value": round(a["value"], 2),
              # Share of this holding sitting in this account. A display split
              # of where the position already is — never a recommendation of
              # which account to trade in.
              "share": round(a["value"] / total, 6)}
             for a in p.pop("_acct").values()),
            key=lambda a: a["value"], reverse=True)
        p["units"] = round(p["units"], 6)
        p["value"] = round(p["value"], 2)

    # Pre-select the top N by value; the rest stay listed and unticked so the
    # user can swap them in rather than discover them missing later.
    for i, p in enumerate(supported):
        p["selected"] = i < MAX_TICKERS
        if i >= MAX_TICKERS:
            p["reason"] = REASON_OVERFLOW

    notes: list[str] = []
    overflow = max(0, len(supported) - MAX_TICKERS)
    if overflow:
        notes.append(f"{overflow} position(s) beyond the top {MAX_TICKERS} by "
                     f"value are listed but unticked — swap any of them in.")
    if wrong_ccy:
        names = ", ".join(sorted(acct_name(a) for a in wrong_ccy))
        notes.append(f"{names} not in {base_ccy} — excluded rather than "
                     f"converted, since no exchange rate is applied.")
    if unsupported:
        notes.append(f"{len(unsupported)} position(s) cannot be modelled and are "
                     f"excluded from the totals below.")
    if cash > 0:
        notes.append(f"{cash:,.2f} {base_ccy or ''} in cash is not modelled as a "
                     f"position.".replace("  ", " "))

    return {
        "supported":   supported,
        "unsupported": unsupported,
        "cash":        round(cash, 2),
        "currency":    base_ccy,
        "total_value": round(sum(p["value"] for p in supported), 2),
        "max_tickers": MAX_TICKERS,
        "notes":       notes,
    }
