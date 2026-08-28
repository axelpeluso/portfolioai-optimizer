# ============================================================
# main.py — FastAPI Application
#
# REST + SSE API for the PortfolioAI rebalancing engine:
#   /optimize, /explain, /chat, /tickers, /waitlist, /track, /analytics
# ============================================================

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone
import os
import re
import json
import time
import uuid
import hashlib
import secrets
import logging
import uvicorn

from optimizer import run_full_analysis

# ── APP SETUP ────────────────────────────────────────────────
app = FastAPI(
    title       = "Portfolio Optimizer API",
    description = "AI-powered portfolio rebalancing using ML + MPT",
    version     = "1.0.0"
)

# Allow the HTML frontend to call this API.
#
# Defaults to "*" so existing deployments keep working. Set ALLOWED_ORIGINS to a
# comma-separated list before enabling SnapTrade in production — see the note in
# .env.example about what that does and does not protect against.
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
            if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins  = _origins,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── LÍMITES DE ABUSO ──────────────────────────────────────────
# /optimize cuesta ~40 s de CPU en frio; /chat y /explain facturan a Anthropic.
# Los tres estaban abiertos y sin tope: cualquiera con la URL podia agotar los
# creditos o saturar la instancia. max_tokens limita la SALIDA del modelo, no la
# entrada, que es donde esta el gasto que un atacante controla.
#
# LIMITACION CONOCIDA: el contador vive en la memoria del proceso. Se reinicia en
# cada deploy y no se comparte entre instancias. Es un lomo de burro contra abuso
# casual, no un control frente a un atacante decidido; para eso hace falta un
# contador en Supabase o Redis.

MAX_BODY_BYTES = 256 * 1024        # 256 KB
# Medido sobre cuerpos reales: el peor caso de /optimize (15 posiciones con lotes
# fiscales) pesa 14.9 KB y una conversacion larga de /chat, 10.8 KB. ~17x de
# margen: el uso normal ni se acerca.

RATE_LIMITS = {                    # llamadas por IP por hora
    "optimize":          20,       # ~40 s de CPU cada una
    "llm":               30,       # /chat y /explain comparten cupo: misma factura
    "snaptrade_session": 10,       # crea usuarios facturables en SnapTrade
}

MAX_CHAT_MESSAGES  = 40
MAX_CHAT_CHARS     = 20_000
MAX_PORTFOLIO_JSON = 64 * 1024

_rate_state: dict[str, dict[str, list[float]]] = {}


def _rate_limit(bucket: str, ip: str, max_per_hour: int) -> int | None:
    """Registra una llamada. Devuelve los segundos de espera si excede, o None."""
    now = time.time()
    hits = [t for t in _rate_state.setdefault(bucket, {}).get(ip, []) if now - t < 3600]
    _rate_state[bucket][ip] = hits
    if len(hits) >= max_per_hour:
        return max(1, int(3600 - (now - min(hits))))
    hits.append(now)
    return None


def _client_ip(request: Request) -> str:
    # Railway va detras de un proxy: sin esto, todas las peticiones comparten IP
    # y el limite se agota para todos a la vez.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(bucket: str, max_per_hour: int | None = None):
    """Dependencia de FastAPI que aplica el limite de un bucket."""
    limit = max_per_hour or RATE_LIMITS.get(bucket, 60)

    def dep(request: Request):
        retry = _rate_limit(bucket, _client_ip(request), limit)
        if retry is not None:
            raise HTTPException(
                status_code=429,
                detail=f"Demasiadas peticiones. Reintenta en {retry} s.",
                headers={"Retry-After": str(retry)},
            )
    return dep


@app.middleware("http")
async def _cap_body_size(request: Request, call_next):
    """Rechaza cuerpos desmedidos antes de leerlos.

    Mirar Content-Length evita cargar en memoria algo que igual vamos a
    descartar. Un cliente puede omitir el header, pero entonces el cuerpo se
    lee en trozos y los limites por endpoint siguen aplicando.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=413,
            content={"detail": f"Cuerpo demasiado grande "
                               f"(máximo {MAX_BODY_BYTES // 1024} KB)."},
        )
    return await call_next(request)


def _check_json_size(obj, label: str, limit: int = MAX_PORTFOLIO_JSON) -> None:
    """El costo de un prompt lo fija su tamaño serializado, no el del request."""
    if obj is None:
        return
    size = len(json.dumps(obj, default=str))
    if size > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{label} demasiado grande ({size // 1024} KB, "
                   f"máximo {limit // 1024} KB).")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────
class TaxContext(BaseModel):
    """Optional broker-derived detail used for tax DISCLOSURE only.

    Sent by the client after a brokerage import. Nothing here is persisted, and
    omitting it simply means no tax panel is returned.
    """
    positions  : dict = {}          # symbol -> {price, units, tax_lots, _account, …}
    accounts   : dict = {}          # account id -> {raw_type, currency, …}
    rates      : Optional[dict] = None   # the USER'S own rates; no defaults exist
    lot_method : Optional[str] = None    # FIFO | LIFO | HIFO; ignored where ACB is required


class OptimizeRequest(BaseModel):
    tickers          : list[str]
    current_holdings : Optional[dict[str, float]] = {}
    # Opt-in optimizer modes. Both off => identical output to before they existed.
    minimize_trading : Optional[str] = None      # 'light' | 'moderate' | 'strong'
    tax_aware        : bool = False
    tax_context      : Optional[TaxContext] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "tickers": ["AAPL","MSFT","GOOGL","JPM","BND","GLD","AMZN"],
                "current_holdings": {
                    "AAPL": 5000,
                    "MSFT": 2000,
                    "GOOGL": 3000,
                    "JPM": 1000,
                    "BND": 500,
                    "GLD": 0,
                    "AMZN": 0
                }
            }
        }
    }

class ExplainRequest(BaseModel):
    portfolio_data: dict


class ChatMessage(BaseModel):
    role   : str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages       : list[ChatMessage]
    portfolio_data : Optional[dict] = None


class WaitlistRequest(BaseModel):
    email     : str
    name      : str
    user_type : str
    source    : Optional[str] = "linkedin"


class TrackRequest(BaseModel):
    event_type : str
    session_id : Optional[str] = None
    tickers    : Optional[list[str]] = None
    metadata   : Optional[dict] = None


EXPLAIN_PROMPT = (
    "You are a senior portfolio manager. Analyze this AI-generated rebalancing "
    "and explain it in 3-4 sentences for an investor: (1) what the risk score "
    "means, (2) why the biggest trades make sense, (3) what the investor gains. "
    "Be direct, no bullet points.\n\n"
    "Portfolio data: {portfolio_data}"
)

CHAT_SYSTEM_PROMPT = (
    "You are a senior portfolio manager assistant for PortfolioAI. You have "
    "access to the user's current portfolio optimization results.\n\n"
    "When the user asks about adding a ticker, changing risk, or re-optimizing: "
    "give a brief 2-sentence analysis, then end your response with exactly "
    '"[REOPTIMIZE]" on a new line to trigger a re-run.\n\n'
    "When answering general questions: be direct and specific, under 4 sentences. "
    "Reference actual numbers from their portfolio when available.\n\n"
    "Never use bullet points. Always sound like a confident quant, not a "
    "disclaimer-heavy advisor.\n\n"
    "If the user reports a bug, has a question, or wants to contact support: "
    "acknowledge it warmly, then give them our support email so they can reach "
    "the team directly: hi@axelpeluso.com. Be conversational and brief.\n\n"
    "PORTFOLIO ACTIONS: If the user wants to add or remove tickers, after your "
    "analysis emit a single machine-readable line exactly like:\n"
    '[ACTION]{"add":["TSLA"],"remove":[],"risk":null}\n'
    "Use uppercase ticker symbols; set \"risk\" to \"low\", \"medium\", \"high\", "
    "or null. Include [REOPTIMIZE] whenever an action is emitted. The [ACTION] "
    "and [REOPTIMIZE] tokens must be the very last thing in your reply and are "
    "hidden from the user, so keep your natural-language answer complete without "
    "them."
)

MODEL = "claude-haiku-4-5-20251001"


# ── ANTHROPIC HELPERS ─────────────────────────────────────────
def _anthropic_client():
    """Return an Anthropic client or raise a clear HTTP error."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY not configured on the server."
        )
    try:
        from anthropic import Anthropic
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="anthropic SDK not installed. Add it to requirements.txt."
        )
    return Anthropic(api_key=api_key)


def _stream_sse(client, **kwargs):
    """Stream a Claude message as Server-Sent Events (data: {...}\\n\\n)."""
    def event_stream():
        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── SUPABASE HELPER ───────────────────────────────────────────
_supabase = None


def _supabase_client():
    """Return a cached Supabase client or raise a clear HTTP error."""
    global _supabase
    if _supabase is not None:
        return _supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured (set SUPABASE_URL and SUPABASE_KEY)."
        )
    try:
        from supabase import create_client
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="supabase not installed. Add supabase to requirements.txt."
        )
    _supabase = create_client(url, key)
    return _supabase


# ── SNAPTRADE: PRINCIPALS ─────────────────────────────────────
# A "principal" owns a SnapTrade connection. Today every principal is an
# ephemeral session (no login); Phase 2 adds kind='account' tied to a real user
# without changing any of the routes below.

SNAPTRADE_TABLE       = "snaptrade_principals"
SESSION_TTL_HOURS     = 24


def _fernet():
    """Cipher for the SnapTrade userSecret.

    Encrypted at application level rather than left in plaintext behind the
    Supabase service_role key: that key is broad, and a leaked table dump must
    not hand over access to anyone's brokerage account.
    """
    key = os.environ.get("SNAPTRADE_ENCRYPTION_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="SNAPTRADE_ENCRYPTION_KEY not configured on the server."
        )
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="cryptography not installed. Add it to requirements.txt."
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def _hash_token(token: str) -> str:
    """Only the hash is stored — the token itself lives in the browser alone."""
    return hashlib.sha256(token.encode()).hexdigest()


def _rate_limited(ip: str) -> bool:
    """Compatibilidad: el limitador de sesiones de SnapTrade."""
    return _rate_limit("snaptrade_session", ip, RATE_LIMITS["snaptrade_session"]) is not None


def _principal(authorization: Optional[str] = Header(None)) -> dict:
    """Resolve `Authorization: Bearer <token>` to a live principal.

    Guards every SnapTrade route except session creation. Expired sessions are
    rejected here even if cleanup has not yet deleted them.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(None, 1)[1].strip()

    client = _supabase_client()
    rows = (client.table(SNAPTRADE_TABLE)
            .select("*").eq("token_hash", _hash_token(token))
            .limit(1).execute().data or [])
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    row = rows[0]
    exp = row.get("expires_at")
    if exp:
        expires = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
        if expires <= datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Session expired.")

    row["_secret"] = _fernet().decrypt(row["st_user_secret"].encode()).decode()
    return row


def _st_error(e: Exception) -> HTTPException:
    """Map a SnapTrade failure to an HTTP error without leaking credentials."""
    import brokerage as st
    if isinstance(e, st.SnapTradeNotConfigured):
        return HTTPException(status_code=503, detail=str(e))
    logging.warning(f"SnapTrade call failed: {type(e).__name__}")
    return HTTPException(status_code=502, detail="Brokerage connection failed.")


CONTROL_MARKERS = ("[REOPTIMIZE]", "[ACTION]")


def _visible_cut(buf: str) -> int:
    """
    Index up to which `buf` is safe to stream to the client. Holds back any
    trailing text that has started (or completed) a control marker so tokens
    like [ACTION]{...} and [REOPTIMIZE] never reach the user's screen.
    """
    for i, ch in enumerate(buf):
        if ch != "[":
            continue
        tail = buf[i:]
        for mk in CONTROL_MARKERS:
            if mk.startswith(tail) or tail.startswith(mk):
                return i
    return len(buf)


def _parse_actions(text: str) -> tuple:
    """Extract (reoptimize: bool, action: dict|None) from the full model reply."""
    reoptimize = "[REOPTIMIZE]" in text
    action = None
    m = re.search(r"\[ACTION\]\s*(\{.*?\})", text, re.DOTALL)
    if m:
        try:
            action = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            action = None
    return reoptimize, action


# ── ROUTES ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Portfolio Optimizer API is running ✅",
        "docs"    : "/docs",
        "version" : "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy ✅"}


_ticker_catalog = None


@app.get("/tickers")
def tickers():
    """Return the investable universe (symbol + company name) for the frontend picker."""
    global _ticker_catalog
    if _ticker_catalog is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.json")
        try:
            with open(path, encoding="utf-8") as f:
                names = json.load(f)
        except FileNotFoundError:
            names = {}
        _ticker_catalog = [{"symbol": s, "name": n} for s, n in sorted(names.items())]
    return {"count": len(_ticker_catalog), "tickers": _ticker_catalog,
            **_data_range()}


_range_cache: Optional[dict] = None


def _data_range() -> dict:
    """First and last date in the bundled price history.

    Served so the UI can state the window instead of hardcoding it: the weekly
    refresh moves the end date, and a fixed label goes stale silently — it was
    still claiming "July 2026" with data through August.
    """
    global _range_cache
    if _range_cache is None:
        try:
            import optimizer
            import pandas as pd
            idx = pd.read_csv(optimizer.CSV_PATH, index_col=0, usecols=[0]).index
            _range_cache = {"data_start": str(idx[0])[:10], "data_end": str(idx[-1])[:10]}
        except Exception as e:                  # noqa: BLE001 — cosmetic only
            logging.warning(f"data range unavailable: {type(e).__name__}")
            _range_cache = {"data_start": None, "data_end": None}
    return _range_cache

@app.post("/optimize", dependencies=[Depends(rate_limited("optimize"))])
def optimize(request: OptimizeRequest):
    """
    Main endpoint — runs full ML pipeline and returns
    optimal weights + rebalancing instructions.
    """
    # Validate tickers
    if len(request.tickers) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 2 tickers"
        )
    if len(request.tickers) > 15:
        raise HTTPException(
            status_code=400,
            detail="Maximum 15 tickers allowed"
        )

    # Clean tickers
    tickers = [t.upper().strip() for t in request.tickers]

    if request.minimize_trading not in (None, "light", "moderate", "strong"):
        raise HTTPException(status_code=400,
                            detail="minimize_trading must be light, moderate or strong")

    import tax as taxmod
    ctx  = request.tax_context
    mode = request.minimize_trading

    # Tax-aware selling is a MODIFIER on the turnover penalty, never an
    # independent trigger — and it needs the user's own rates, so without them
    # it stays a no-op instead of guessing.
    tax_weights = None
    if request.tax_aware and ctx and ctx.rates:
        tax_weights = taxmod.penalty_weights(ctx.positions, ctx.accounts, ctx.rates,
                                             lot_method=ctx.lot_method) or None
        if tax_weights and not mode:
            mode = "moderate"     # tax weighting is meaningless with no penalty

    try:
        result = run_full_analysis(tickers, request.current_holdings,
                                   turnover_penalty=mode, tax_weights=tax_weights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Disclosure: always computed when we have the data, regardless of the modes.
    if ctx:
        try:
            result["tax"] = _tax_disclosure(result, ctx, taxmod)
        except Exception as e:                     # noqa: BLE001
            logging.warning(f"tax disclosure failed (omitted): {type(e).__name__}")

    result["modes"] = {"minimize_trading": mode, "tax_aware": bool(tax_weights)}
    return {"success": True, "data": result}


def _tax_disclosure(result: dict, ctx: "TaxContext", taxmod) -> dict:
    """Per-position and portfolio-level tax facts. Never a bill unless rates given."""
    positions, accounts = ctx.positions or {}, ctx.accounts or {}
    requested = ctx.lot_method
    summary = taxmod.portfolio_summary(result["rebalancing"], result["total_value"],
                                       positions, accounts, lot_method=requested)
    per_symbol = {}
    for sym, row in result["rebalancing"].items():
        pos  = positions.get(sym)
        acct = accounts.get((pos or {}).get("_account")) or {}
        cls  = taxmod.classify_account(acct)
        entry = {"account": cls["label"],
                 "sheltered": cls["sheltered"],
                 "holding_period_matters": cls["holding_period_matters"],
                 "profile": taxmod.asset_profile(sym, cls["jurisdiction"])}
        if pos and row.get("trade_amount", 0) < 0 and cls["sheltered"] is False:
            method, forced = taxmod.resolve_method(cls["jurisdiction"], requested)
            entry["sale"] = taxmod.sale_consequence(pos, -row["trade_amount"],
                                                    method=method)
            entry["method"] = method
            # Canada permits only ACB; say so rather than looking like a choice.
            entry["method_forced"] = forced
            # A partial sale has no single answer until lots are picked.
            rng = taxmod.sale_range(pos, -row["trade_amount"], cls["jurisdiction"])
            if rng:
                entry["range"] = rng
        per_symbol[sym] = entry

    est = None
    if ctx.rates and summary.get("realized_gain") is not None:
        est = taxmod.estimate_tax(summary.get("short_term") or 0.0,
                                  summary.get("long_term") or summary["realized_gain"],
                                  ctx.rates)
    return {"summary": summary, "per_symbol": per_symbol, "estimate": est}


@app.post("/explain", dependencies=[Depends(rate_limited("llm"))])
def explain(request: ExplainRequest):
    """
    Stream a plain-language explanation of the rebalancing result from
    Claude via Server-Sent Events (SSE), so text appears word by word.
    """
    # portfolio_data va entero dentro del prompt: su tamaño ES el costo.
    _check_json_size(request.portfolio_data, "portfolio_data")

    client = _anthropic_client()
    prompt = EXPLAIN_PROMPT.format(
        portfolio_data=json.dumps(request.portfolio_data, default=str)
    )
    return _stream_sse(
        client,
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )


@app.post("/chat", dependencies=[Depends(rate_limited("llm"))])
def chat(request: ChatRequest):
    """
    Conversational portfolio assistant. Streams a Claude response (SSE) using
    the PortfolioAI system prompt, with the current optimization results
    injected as context. The model may emit "[REOPTIMIZE]" to signal the
    frontend to re-run the optimization.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # El gasto real de un prompt esta en los tokens de ENTRADA, que max_tokens no
    # limita. Sin estos topes, un solo request puede costar arbitrariamente caro.
    if len(request.messages) > MAX_CHAT_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Demasiados mensajes (máximo {MAX_CHAT_MESSAGES}).")
    total = sum(len(m.content or "") for m in request.messages)
    if total > MAX_CHAT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Conversación demasiado larga ({total} caracteres, "
                   f"máximo {MAX_CHAT_CHARS}).")
    _check_json_size(request.portfolio_data, "portfolio_data")

    client = _anthropic_client()

    system = CHAT_SYSTEM_PROMPT
    if request.portfolio_data:
        system += (
            "\n\nCurrent portfolio optimization results (JSON):\n"
            + json.dumps(request.portfolio_data, default=str)
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    def event_stream():
        full = ""
        sent = 0
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=500,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full += text
                    cut = _visible_cut(full)
                    if cut > sent:
                        yield f"data: {json.dumps({'text': full[sent:cut]})}\n\n"
                        sent = cut

            reoptimize, action = _parse_actions(full)
            meta = {"done": True, "reoptimize": reoptimize}
            if action is not None:
                meta["action"] = action
            yield f"data: {json.dumps(meta)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── WAITLIST + ANALYTICS ──────────────────────────────────────
def _is_duplicate_error(e: Exception) -> bool:
    """Detect a Postgres unique-violation (duplicate email)."""
    code = getattr(e, "code", None)
    if code == "23505":
        return True
    msg = str(e).lower()
    return "duplicate" in msg or "23505" in msg or "already exists" in msg


@app.post("/waitlist")
def waitlist(request: WaitlistRequest):
    """Add an email to the waitlist. Returns {error:'already registered'} on duplicate."""
    client = _supabase_client()
    row = {
        "email"    : request.email.strip().lower(),
        "name"     : request.name.strip(),
        "user_type": request.user_type.strip(),
        "source"   : (request.source or "linkedin").strip(),
    }
    if not row["email"] or not row["name"] or not row["user_type"]:
        raise HTTPException(status_code=400, detail="email, name and user_type are required.")
    try:
        client.table("waitlist").insert(row).execute()
    except Exception as e:
        if _is_duplicate_error(e):
            return {"error": "already registered"}
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True}


@app.post("/track")
def track(request: TrackRequest):
    """Fire-and-forget analytics event. Always returns 200, never blocks the caller."""
    try:
        client = _supabase_client()
        client.table("events").insert({
            "event_type": request.event_type,
            "session_id": request.session_id,
            "tickers"   : request.tickers,
            "metadata"  : request.metadata,
        }).execute()
    except Exception as e:
        logging.warning(f"track event failed (ignored): {e}")
    return {"success": True}


@app.get("/analytics")
def analytics(x_admin_key: Optional[str] = Header(None)):
    """Admin dashboard data. Requires the X-Admin-Key header to match ADMIN_KEY."""
    admin_key = os.environ.get("ADMIN_KEY")
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client = _supabase_client()

    wl_count = client.table("waitlist").select("id", count="exact").execute().count or 0

    events = client.table("events").select("event_type").execute().data or []
    events_by_type = {}
    for e in events:
        t = e.get("event_type", "unknown")
        events_by_type[t] = events_by_type.get(t, 0) + 1

    signups = client.table("waitlist").select("created_at").execute().data or []
    signups_per_day = {}
    for s in signups:
        day = str(s.get("created_at", ""))[:10]
        if day:
            signups_per_day[day] = signups_per_day.get(day, 0) + 1

    return {
        "waitlist_count" : wl_count,
        "events_by_type" : events_by_type,
        "signups_per_day": dict(sorted(signups_per_day.items())),
        "total_events"   : len(events),
    }


# ── SNAPTRADE: READ-ONLY BROKERAGE IMPORT ─────────────────────
# Connect a brokerage, read positions, hand them to the sidebar. No trading:
# the app stays a research tool, and nothing here can place an order.

class ConnectRequest(BaseModel):
    redirect_uri: Optional[str] = None


@app.get("/snaptrade/status")
def snaptrade_status():
    """Whether brokerage import is available, so the UI can hide the button."""
    import brokerage as st
    return {"enabled": st.is_configured(), "max_tickers": st.MAX_TICKERS}


@app.post("/snaptrade/session")
def snaptrade_session(request: Request):
    """Create an ephemeral principal and register it with SnapTrade.

    Returns a bearer token the browser keeps for the session. Only the token's
    hash is stored here; the SnapTrade userSecret is encrypted at rest.
    """
    import brokerage as st

    # _client_ip y no request.client.host: detras del proxy de Railway todas las
    # peticiones comparten IP y el cupo se agotaria para todos a la vez.
    if _rate_limited(_client_ip(request)):
        raise HTTPException(status_code=429,
                            detail="Too many sessions from this address.")

    st_user_id = f"pai_{uuid.uuid4().hex}"
    try:
        user_secret = st.register_user(st_user_id)
    except Exception as e:                      # noqa: BLE001
        raise _st_error(e)

    token   = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)

    # Cifrar antes del insert: si la clave Fernet esta mal formada, el fallo es
    # de configuracion y conviene distinguirlo de un problema de base de datos.
    try:
        encrypted = _fernet().encrypt(user_secret.encode()).decode()
    except HTTPException:
        raise
    except Exception as e:                      # noqa: BLE001
        try:
            st.delete_user(st_user_id)
        except Exception:                       # noqa: BLE001
            logging.warning(f"orphaned SnapTrade user {st_user_id}")
        raise HTTPException(
            status_code=500,
            detail=f"SNAPTRADE_ENCRYPTION_KEY invalida ({type(e).__name__}). "
                   f"Debe ser una clave Fernet: 44 caracteres base64 url-safe.")

    try:
        _supabase_client().table(SNAPTRADE_TABLE).insert({
            "kind":           "session",
            "token_hash":     _hash_token(token),
            "st_user_id":     st_user_id,
            "st_user_secret": encrypted,
            "expires_at":     expires.isoformat(),
        }).execute()
    except Exception as e:                      # noqa: BLE001
        # Never strand a registered SnapTrade user we cannot reach again —
        # those are billed per connection.
        try:
            st.delete_user(st_user_id)
        except Exception:                       # noqa: BLE001
            logging.warning(f"orphaned SnapTrade user {st_user_id}")
        # El tipo y el mensaje del driver, sin secretos: sin esto un fallo de
        # tabla, de permisos o de columna se ven todos igual.
        # El mensaje completo va al log; al cliente solo el tipo. El detalle del
        # driver nombra tablas y permisos, y eso no es asunto de quien llama.
        logging.warning(f"snaptrade session insert failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo guardar la sesión ({type(e).__name__}).")

    return {"token": token, "expires_at": expires.isoformat()}


@app.post("/snaptrade/connect")
def snaptrade_connect(body: ConnectRequest, principal: dict = Depends(_principal)):
    """URL of SnapTrade's hosted portal, where the user links their brokerage."""
    import brokerage as st
    try:
        url = st.login_url(principal["st_user_id"], principal["_secret"],
                           body.redirect_uri)
    except Exception as e:                      # noqa: BLE001
        raise _st_error(e)
    return {"url": url}


@app.get("/snaptrade/accounts")
def snaptrade_accounts(principal: dict = Depends(_principal)):
    import brokerage as st
    try:
        return {"accounts": st.list_accounts(principal["st_user_id"],
                                             principal["_secret"])}
    except Exception as e:                      # noqa: BLE001
        raise _st_error(e)


@app.get("/snaptrade/positions")
def snaptrade_positions(accounts: Optional[str] = None,
                        principal: dict = Depends(_principal)):
    """Positions across the connected accounts, reconciled against the universe.

    `accounts` is an optional comma-separated list of account ids; omitted means
    all of them. Consolidation happens here rather than in the browser so the
    15-ticker cap and the universe check have exactly one implementation.

    Nothing is persisted — positions are fetched, reconciled, returned and
    discarded, which keeps personal financial data out of our storage entirely.
    """
    import brokerage as st
    uid, secret = principal["st_user_id"], principal["_secret"]
    wanted = [a for a in (accounts or "").split(",") if a.strip()] or None

    try:
        all_accounts = st.list_accounts(uid, secret)
        raw, cash = [], {}
        for acct in all_accounts:
            if wanted is not None and acct["id"] not in wanted:
                continue                    # don't pay for accounts we exclude
            raw += st.list_positions(uid, secret, acct["id"])
            for ccy, amount in st.account_cash(uid, secret, acct["id"]).items():
                cash[ccy] = cash.get(ccy, 0.0) + amount
    except Exception as e:                      # noqa: BLE001
        raise _st_error(e)

    result = st.reconcile(raw, cash, accounts=all_accounts, account_ids=wanted)
    result["accounts"] = all_accounts           # always the full list, so the
    result["selected_accounts"] = wanted        # UI can offer the unselected
    return result


@app.delete("/snaptrade/session")
def snaptrade_disconnect(principal: dict = Depends(_principal)):
    """Drop the brokerage connection and the principal row.

    Deletes the SnapTrade user too, not just our row — connections are billed
    per user, so an orphan keeps costing after the session is gone.
    """
    import brokerage as st
    try:
        st.delete_user(principal["st_user_id"])
    except Exception:                           # noqa: BLE001
        logging.warning(f"could not delete SnapTrade user {principal['st_user_id']}")
    _supabase_client().table(SNAPTRADE_TABLE).delete().eq("id", principal["id"]).execute()
    return {"success": True}


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)# deploy trigger
