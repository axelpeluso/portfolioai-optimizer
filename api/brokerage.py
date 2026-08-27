"""
brokerage.py — SnapTrade wrapper + position reconciliation.

Read-only. This module registers users, hands back connection-portal URLs, and
reads accounts/positions. It never places a trade.

The SDK is imported lazily (same pattern as the anthropic/supabase helpers in
main.py) so the API still boots and the existing test suite still runs when
snaptrade is not installed.

Reconciliation lives here rather than in the frontend because both constraints
it enforces are server-side facts: the 289-symbol universe in tickers.json, and
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
REASON_UNIVERSE = "not in the 289-instrument universe"
REASON_OVERFLOW = f"below the top {MAX_TICKERS} by value"
REASON_KIND     = "not a modellable equity/ETF position"
REASON_NOVALUE  = "no market value reported"


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
        from snaptrade_client import SnapTrade
    except ImportError:
        raise SnapTradeNotConfigured(
            "snaptrade-python-sdk not installed. Add it to requirements.txt."
        )
    _client = SnapTrade(client_id=client_id, consumer_key=consumer_key)
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
        })
    return out


def list_positions(user_id: str, user_secret: str, account_id: str) -> list[dict]:
    res = client().account_information.get_all_account_positions(
        user_id=user_id, user_secret=user_secret, account_id=account_id
    )
    return _body(res) or []


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


def reconcile(raw_positions: list[dict], cash: float = 0.0) -> dict:
    """Split broker positions into what /optimize can and cannot accept.

    Returns supported (top 15 by value pre-selected), unsupported (each with a
    reason), the cash figure, and human-readable notes.
    """
    known = universe()
    supported: list[dict] = []
    unsupported: list[dict] = []

    for pos in raw_positions:
        symbol, kind = _symbol_of(pos)
        units = _num(pos.get("units") or pos.get("fractional_units"))
        price = _num(pos.get("price"))
        value = _num(pos.get("market_value")) or (units * price)

        if not symbol:
            continue
        if kind and str(kind).lower() in _UNSUPPORTED_KINDS:
            unsupported.append({"symbol": symbol, "value": round(value, 2),
                                "reason": REASON_KIND})
        elif symbol not in known:
            unsupported.append({"symbol": symbol, "value": round(value, 2),
                                "reason": REASON_UNIVERSE})
        elif value <= 0:
            unsupported.append({"symbol": symbol, "value": 0.0,
                                "reason": REASON_NOVALUE})
        else:
            supported.append({"symbol": symbol, "units": round(units, 6),
                              "value": round(value, 2)})

    # Merge duplicates — the same symbol can appear across several accounts.
    merged: dict[str, dict] = {}
    for p in supported:
        m = merged.setdefault(p["symbol"], {"symbol": p["symbol"], "units": 0.0,
                                            "value": 0.0})
        m["units"] += p["units"]
        m["value"] += p["value"]
    supported = sorted(merged.values(), key=lambda p: p["value"], reverse=True)
    for p in supported:
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
    if unsupported:
        notes.append(f"{len(unsupported)} position(s) cannot be modelled and are "
                     f"excluded from the totals below.")
    if cash > 0:
        notes.append(f"${cash:,.2f} in cash is not modelled as a position.")

    return {
        "supported":   supported,
        "unsupported": unsupported,
        "cash":        round(cash, 2),
        "total_value": round(sum(p["value"] for p in supported), 2),
        "max_tickers": MAX_TICKERS,
        "notes":       notes,
    }
