# ============================================================
# main.py — FastAPI Application
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)