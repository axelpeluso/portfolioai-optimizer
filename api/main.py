# ============================================================
# main.py — FastAPI Application
# v2 - explain endpoint active
# ============================================================

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
import logging
import requests
import uvicorn

from optimizer import run_full_analysis

# ── APP SETUP ────────────────────────────────────────────────
app = FastAPI(
    title       = "Portfolio Optimizer API",
    description = "AI-powered portfolio rebalancing using ML + MPT",
    version     = "1.0.0"
)

# Allow HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────
class OptimizeRequest(BaseModel):
    tickers          : list[str]
    current_holdings : Optional[dict[str, float]] = {}

    class Config:
        json_schema_extra = {
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


class ContactRequest(BaseModel):
    name    : str = ""
    email   : str = ""
    message : str
    source  : str = "chat"


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


CONTROL_MARKERS = ("[REOPTIMIZE]", "[ACTION]", "[SUPPORT_READY")


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


def _parse_support(text: str):
    """Extract {name, email, issue} from a [SUPPORT_READY: name | email | issue] tag."""
    m = re.search(r"\[SUPPORT_READY:([^\]]*)\]", text, re.DOTALL)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split("|")]
    return {
        "name" : parts[0] if len(parts) > 0 else "",
        "email": parts[1] if len(parts) > 1 else "",
        "issue": parts[2] if len(parts) > 2 else "",
    }


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
    return {"count": len(_ticker_catalog), "tickers": _ticker_catalog}

@app.post("/optimize")
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

    try:
        result = run_full_analysis(tickers, request.current_holdings)
        return {
            "success" : True,
            "data"    : result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(request: ExplainRequest):
    """
    Stream a plain-language explanation of the rebalancing result from
    Claude via Server-Sent Events (SSE), so text appears word by word.
    """
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


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Conversational portfolio assistant. Streams a Claude response (SSE) using
    the PortfolioAI system prompt, with the current optimization results
    injected as context. The model may emit "[REOPTIMIZE]" to signal the
    frontend to re-run the optimization.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

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
            support = _parse_support(full)
            meta = {"done": True, "reoptimize": reoptimize}
            if action is not None:
                meta["action"] = action
            if support is not None:
                meta["support"] = support
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


# ── SUPPORT / CONTACT ─────────────────────────────────────────
def send_support_email(name, email, message, source):
    """Send a support notification via the Resend HTTP API (Railway blocks outbound
    SMTP, so raw smtplib hangs). No-op if RESEND_API_KEY is unset. Self-contained
    (swallows its own errors) so it is safe to run as a background task."""
    api_key = os.getenv("RESEND_API_KEY")
    to_addr = os.getenv("SUPPORT_EMAIL_TO")
    if not api_key or not to_addr:
        return

    # Resend requires a verified domain to send from; onboarding@resend.dev works
    # out of the box (to the account owner) for testing until the domain is verified.
    sender = os.getenv("SUPPORT_EMAIL_FROM") or "onboarding@resend.dev"

    body = f"""New support request from PortfolioAI

Name:    {name or 'Not provided'}
Email:   {email or 'Not provided'}
Source:  {source}

Message:
{message}

---
Reply directly to {email} to follow up.
"""
    payload = {
        "from"    : f"PortfolioAI <{sender}>",
        "to"      : [to_addr],
        "subject" : f"[PortfolioAI Support] {message[:50]}...",
        "text"    : body,
        "reply_to": email or sender,
    }
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
        if r.status_code >= 400:
            logging.warning(f"Resend email failed {r.status_code}: {r.text[:300]}")
    except Exception as e:
        logging.warning(f"Support email failed: {e}")


@app.post("/contact")
def contact(request: ContactRequest, background_tasks: BackgroundTasks):
    """Log a support request to Supabase and notify the team by email. Best-effort:
    the Supabase row is the durable record; the email is sent in the background so a
    slow or blocked SMTP port never delays (or hangs) the response."""
    try:
        client = _supabase_client()
        client.table("support").insert({
            "name"   : request.name,
            "email"  : request.email,
            "message": request.message,
            "source" : request.source,
        }).execute()
    except Exception as e:
        logging.warning(f"Supabase contact insert failed: {e}")

    background_tasks.add_task(
        send_support_email, request.name, request.email, request.message, request.source
    )
    return {"success": True}


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)# deploy trigger
