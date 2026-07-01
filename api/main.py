# ============================================================
# main.py — FastAPI Application
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
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


EXPLAIN_PROMPT = (
    "You are a senior portfolio manager. Analyze this AI-generated rebalancing "
    "and explain it in 3-4 sentences for an investor: (1) what the risk score "
    "means, (2) why the biggest trades make sense, (3) what the investor gains. "
    "Be direct, no bullet points.\n\n"
    "Portfolio data: {portfolio_data}"
)

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

    client = Anthropic(api_key=api_key)
    prompt = EXPLAIN_PROMPT.format(
        portfolio_data=json.dumps(request.portfolio_data, default=str)
    )

    def event_stream():
        try:
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)