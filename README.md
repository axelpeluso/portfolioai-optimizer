# 📈 PortfolioAI — AI-Powered Portfolio Rebalancing

> An end-to-end machine-learning pipeline that profiles each asset, predicts forward returns, scores market risk, and emits actionable BUY / SELL / HOLD instructions in dollars — with a **Claude AI** assistant that explains and re-optimizes your portfolio in plain English. Served through a FastAPI backend and a TradingView-style web UI.

![CI](https://github.com/axelpeluso/portfolioai-optimizer/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-D97757?logo=anthropic&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-waitlist_%2B_analytics-3ECF8E?logo=supabase&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**🔗 Landing page:** https://portfolioai-optimizer.vercel.app · **the tool itself** lives at `/app` (unlisted — access is granted after a demo)

> ⚠️ **Testing Beta — not investment advice.** For research and educational purposes only.

---

## ✨ Highlights

- **Five-model engine** — K-Means + Random Forest + MLP + Markowitz MPT + a rebalancing translator.
- **Claude AI layer** — a one-click "Explain with Claude" analysis and a conversational assistant that can **add/remove tickers and re-optimize** on request.
- **288-instrument universe** — stocks, sector/factor/bond/commodity ETFs, and crypto (BTC/ETH ETFs).
- **Scenario compare** — every run is snapshotted; flip between them or view a side-by-side delta table.
- **Ticker autocomplete** — search 288 instruments by symbol *or* company name.
- **Deployed** — backend on Railway, frontend on Vercel, waitlist + analytics on Supabase.

> Actual Sharpe / volatility / risk figures depend on the tickers, holdings, and data window you run — the app reports them live per optimization.

---

## 🧠 What it does

PortfolioAI combines **unsupervised learning**, **supervised learning**, **deep learning**, **classical financial theory**, and a **large language model** into a single rebalancing experience:

1. **K-Means Clustering** profiles each asset as `Growth` / `Moderate` / `Defensive` from return / volatility / Sharpe.
2. **Random Forest Regressor** (per-ticker) predicts the next-quarter return from a rolling-window feature set.
3. **MLP Neural Network** evaluates the *current* portfolio's market regime and emits a 0–1 **risk score** (sigmoid-squashed so it never hard-pins to 1.0).
4. **Modern Portfolio Theory** (Markowitz, SLSQP-solved) finds the Max-Sharpe and Min-Variance weights — with bounds that adapt to the MLP's risk score, and Ledoit-Wolf covariance shrinkage.
5. **Rebalancing Engine** translates optimal weights into concrete `BUY $X` / `SELL $Y` / `HOLD` actions vs. your current holdings.
6. **Claude (Haiku 4.5)** streams a plain-English explanation of the result, and powers a chat assistant that can re-optimize with structured ticker actions.

---

## 🏗️ Architecture

```
        ┌──────────────────────────────┐        ┌──────────────────┐
        │  Frontend (Vercel)           │        │  Anthropic API   │
        │  frontend/index.html         │──────▶ │  Claude Haiku    │
        │  Chart.js + vanilla JS       │  SSE   │  /explain /chat  │
        └──────────────┬───────────────┘        └──────────────────┘
                       │ REST / SSE
                       ▼
        ┌──────────────────────────────┐        ┌──────────────────┐
        │  Backend (Railway)           │──────▶ │  Supabase        │
        │  api/main.py — FastAPI       │        │  waitlist/events │
        └──────────────┬───────────────┘        └──────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  ML + MPT engine             │
        │  api/optimizer.py            │
        │  KMeans · RF · MLP · SLSQP   │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  api/prices.csv (288 tickers)│  ← bundled adjusted-close history
        │  api/tickers.json (names)    │     (generated offline via yfinance)
        └──────────────────────────────┘
```

---

## 📂 Project structure

```
Final Project/
├── api/
│   ├── main.py                 ← FastAPI app + routes (optimize, explain, chat, waitlist, …)
│   ├── optimizer.py            ← ML + MPT pipeline
│   ├── brokerage.py            ← SnapTrade wrapper + position reconciliation
│   ├── tax.py                  ← tax disclosure (facts, never a bill)
│   ├── tax_profiles.json       ← hand-verified asset tax character (REIT/bond/…)
│   ├── refresh_prices.py       ← rebuild prices.csv from Yahoo (see workflow)
│   ├── test_api.py             ← pytest smoke tests for the core endpoints
│   ├── test_snaptrade.py       ← reconciliation + auth-gate tests
│   ├── test_tax.py             ← tax disclosure tests
│   ├── test_optimizer_modes.py ← opt-in modes + default-off equivalence
│   ├── test_security.py        ← límites de tamaño y frecuencia
│   ├── prices.csv              ← bundled adjusted-close history (288 tickers)
│   ├── tickers.json            ← symbol → company name map (for autocomplete)
│   ├── requirements.txt        ← Python deps (deployment)
│   ├── railway.json            ← Railway deploy config
│   ├── Procfile / runtime.txt  ← process + Python version
├── frontend/
│   ├── landing.html            ← public landing page (demo request form)
│   └── index.html              ← TradingView-style UI (Chart.js, floating Claude chat)
├── notebook/
│   └── portfolio_optimizer.ipynb   ← full ML walk-through
├── docs/
│   ├── whitepaper.md           ← documento técnico completo (modelos, validación, límites)
│   ├── executive-summary.md    ← qué es, cómo funciona, estado medido, límites
│   ├── demo-script.md          ← guion de demo paso a paso + plan de contingencia
│   ├── snaptrade-setup.md      ← brokerage import: credentials, table, cleanup
│   └── dormant-support-flow.md ← parked in-app support (Supabase + email) design
├── .github/workflows/
│   ├── ci.yml                  ← GitHub Actions (pytest on push / PR)
│   └── refresh-prices.yml      ← weekly price refresh → commits prices.csv
├── requirements-notebook.txt   ← notebook-only extras (matplotlib, yfinance)
├── vercel.json                 ← routes / → landing.html, /app → index.html
├── .env.example                ← required env var names (no secrets)
├── .gitignore
└── README.md
```

---

## 🚀 Installation

> Requires **Python 3.11+** (developed on 3.13).

```bash
git clone https://github.com/axelpeluso/portfolioai-optimizer.git
cd portfolioai-optimizer

python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r api/requirements.txt
```

### Configuration

Copy `.env.example` and fill in your own values — **never commit real secrets**:

```bash
cp .env.example .env
```

| Variable | Required for | Notes |
|----------|--------------|-------|
| `ANTHROPIC_API_KEY` | `/explain`, `/chat` | Claude API key |
| `SUPABASE_URL` | `/waitlist`, `/track`, `/analytics` | Supabase project URL |
| `SUPABASE_KEY` | `/waitlist`, `/track`, `/analytics` | **service_role** key |
| `ADMIN_KEY` | `/analytics` | secret; sent as the `X-Admin-Key` header |
| `SNAPTRADE_CLIENT_ID`, `SNAPTRADE_CONSUMER_KEY` | `/snaptrade/*` | brokerage import; unset = feature hidden |
| `SNAPTRADE_ENCRYPTION_KEY` | `/snaptrade/*` | Fernet key encrypting each stored `userSecret` |
| `ALLOWED_ORIGINS` | CORS | comma-separated allowlist; defaults to `*` |
| `RESEND_API_KEY`, `SUPPORT_EMAIL_TO`, `SUPPORT_EMAIL_FROM` | (dormant support flow) | see [`docs/dormant-support-flow.md`](docs/dormant-support-flow.md) |

The core `/optimize` pipeline needs **no** secrets — the AI, waitlist, and analytics features degrade gracefully (clear 500 / no-op) when their keys are unset. In production these are set in the Railway service's **Variables** tab.

---

## ▶️ How to run

### 1. The notebook (full ML walk-through)
The notebook needs a couple of extra libraries (plotting + live data) beyond the API:
```bash
pip install -r requirements-notebook.txt
jupyter notebook notebook/portfolio_optimizer.ipynb
```

### 2. The API
```bash
cd api
uvicorn main:app --reload --port 8000
```
- 🌐 **API root** → http://127.0.0.1:8000/
- 📚 **Swagger UI** → http://127.0.0.1:8000/docs

### 3. The frontend
Open `frontend/index.html` in your browser. By default it calls the **deployed Railway API** (`const API` at the top of the `<script>`). To point it at a local backend, change that constant to `http://127.0.0.1:8000`.

---

## 🔌 API endpoints

| Method | Path | Description |
|-------:|------|-------------|
| `GET`  | `/`          | Health banner + version |
| `GET`  | `/health`    | Liveness probe |
| `GET`  | `/tickers`   | The 288-instrument universe (symbol + company name) for the picker |
| `POST` | `/optimize`  | Run the full ML pipeline + return rebalancing |
| `POST` | `/explain`   | **SSE** — Claude's plain-English analysis of a result |
| `POST` | `/chat`      | **SSE** — conversational assistant (may emit re-optimize / ticker actions) |
| `POST` | `/waitlist`  | Add an email to the Supabase waitlist |
| `GET`  | `/snaptrade/status`    | Whether brokerage import is configured |
| `POST` | `/snaptrade/session`   | Start an ephemeral session, returns a bearer token |
| `POST` | `/snaptrade/connect`   | URL of SnapTrade's hosted connection portal |
| `GET`  | `/snaptrade/accounts`  | Connected brokerage accounts |
| `GET`  | `/snaptrade/positions` | Positions reconciled against the 288-symbol universe |
| `DELETE` | `/snaptrade/session` | Disconnect and delete the SnapTrade user |
| `POST` | `/track`     | Fire-and-forget analytics event (always 200) |
| `GET`  | `/analytics` | Admin dashboard data (requires `X-Admin-Key`) |

### `POST /optimize` — request
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "BND", "GLD", "AMZN"],
  "current_holdings": {
    "AAPL": 5000, "MSFT": 2000, "GOOGL": 3000,
    "JPM":  1000, "BND":   500, "GLD":  0, "AMZN": 0
  }
}
```
Constraints: 2 ≤ `len(tickers)` ≤ 15. `current_holdings` optional (defaults to `{}`; empty → assumes a **$10,000** notional for sizing).

### `POST /optimize` — response shape
```json
{
  "success": true,
  "data": {
    "tickers": ["AAPL", "..."],
    "risk_score": 0.88,
    "risk_level": "HIGH",
    "cluster_map": { "AAPL": "Growth" },
    "expected_returns": { "AAPL": 0.18 },
    "rebalancing": {
      "AAPL": {
        "current_value": 5000, "current_weight": 0.45, "optimal_weight": 0.30,
        "target_value": 3300.00, "trade_amount": -1700.00,
        "action": "SELL", "rf_signal": "Bullish", "cluster": "Growth"
      }
    },
    "total_value": 11500,
    "optimization": {
      "optimal_weights": { "AAPL": 0.30 },
      "min_var_weights": { "AAPL": 0.10 },
      "max_sharpe_metrics": { "return": 0.21, "volatility": 0.17, "sharpe": 1.19 },
      "min_var_metrics": { "return": 0.09, "volatility": 0.07, "sharpe": 0.55 }
    },
    "current_metrics": { "return": 0.20, "volatility": 0.18, "sharpe": 0.77 }
  }
}
```

---

## 🔗 Brokerage import (SnapTrade)

Rather than typing every position, users can connect a brokerage and have the
sidebar fill itself — tickers, dollar holdings and total value.

**Read-only by design.** Nothing in the integration can place a trade. Routing
model-generated BUY/SELL instructions into a live account would make this a
different product with different obligations; it deliberately stops at "here is
what the models suggest".

Positions are reconciled server-side against the 288-symbol universe and the
15-ticker cap, and the review modal shows *every* position with a reason when it
cannot be used — not in the universe, below the top 15 by value, or not
modellable at all (cash, options). Silent truncation would mean optimizing
against a portfolio that isn't the user's.

Positions are never stored: fetched, reconciled, returned, discarded. Only an
encrypted SnapTrade `userSecret` and a token hash are persisted.

The feature is **off unless configured** — with the env vars unset the button is
hidden and the app behaves exactly as before. Setup (credentials, Supabase table,
expiry cleanup): [`docs/snaptrade-setup.md`](docs/snaptrade-setup.md).

> ⏱️ Every imported portfolio is a unique ticker set, so it always misses the
> model cache — expect ~40s for a 15-ticker run.

---

## 🧾 Tax disclosure

The rebalancing engine had no tax awareness at all: it compared target dollars to
current dollars and emitted BUY/SELL. That was fine for typed-in hypotheticals
and not fine once the app started reading real brokerage accounts.

The line drawn throughout is **facts vs. estimates**:

- *Fact* — "this SELL realizes $1,410 of gain, $820 from lots held under a year."
  Computed from broker-supplied cost basis. Shown by default.
- *Estimate* — "you owe $340." Needs bracket, filing status and residency.
  Only ever produced from rates **the user types in**, and labelled illustrative.
  There are no default rates anywhere in the codebase.

### Which lots get sold

On a **partial** sale the lot-selection method dominates the answer — the same
$5,000 trade can realize $500 or $4,000 depending on which lots are consumed.
Rebalancing produces almost nothing but partial sales, so the assumption is
stated on screen rather than hidden:

- **US accounts** default to FIFO, with LIFO and HIFO selectable, because we
  cannot see what a brokerage is actually configured to do. Every figure is
  labelled "assuming FIFO" (or whichever is chosen).
- **Canadian accounts** are forced to **adjusted cost base** (a weighted
  average). FIFO/LIFO/HIFO are not permitted for Canadian tax purposes, so the
  selector is hidden and the reason given.
- Where methods disagree, the **range** is shown too — "$500–$4,000 depending on
  lots sold" is more honest than any single number.

Safety properties that are enforced by tests, not convention:

- An unrecognised account type reports "not identified" — **never** "taxable".
- Without tax lots we report the gain but **no** short/long split, rather than
  inventing one.
- Holding-period language only appears where it applies: US taxable accounts.
  Canadian accounts get inclusion-rate wording, since Canada has no
  short/long distinction.

Turnover is always shown, because it needs no tax data and is the number most
likely to make someone pause — the default portfolio turns over **53%**.

### Opt-in optimizer modes

Both **off by default**, and with both off the optimizer is provably identical to
before the feature existed (`test_optimizer_modes.py` pins it to golden weights).

| Mode | Effect |
|---|---|
| **Minimize trading** | `maximize Sharpe − λ·Σ(wᵢ − wᵢ_current)²`. Named for what it does; needs no tax data. |
| **Tax-aware selling** | Same mechanism, per-asset weights from embedded gain. Requires the user's own rates, so it stays disabled until they are entered. |

Measured on the default portfolio:

| Mode | Turnover | Sharpe |
|---|---:|---:|
| off | 53.3% | 0.969 |
| light | 48.8% | 0.956 |
| moderate | 26.7% | 0.799 |
| strong | 15.1% | 0.694 |

The trade-off is disclosed in both directions — a lower theoretical Sharpe is the
price of trading less.

**Not in scope:** recommending which account to trade in, automatic lot selection,
and state/provincial/NIIT/AMT treatment. Those are named as excluded rather than
approximated.

---

## 🤖 Claude AI features

- **Explain with Claude** — streams a 3–4 sentence explanation of the rebalancing (what the risk score means, why the biggest trades make sense, what you gain).
- **Ask PortfolioAI** — a floating chat assistant. Ask general questions ("what's my Sharpe?") or give commands ("add TSLA", "make it more aggressive"); it emits hidden structured actions, applies them to the ticker chips, re-optimizes, and reports the resulting allocation. Bug reports get pointed to support at **hi@axelpeluso.com**.

Both use **Claude Haiku 4.5** and stream over Server-Sent Events.

---

## 📊 Models cheat-sheet

| Model | Type | Library | Role |
|-------|------|---------|------|
| K-Means | Unsupervised | scikit-learn | Cluster assets by behavioral profile |
| Random Forest | Supervised | scikit-learn | Forecast forward returns per ticker |
| MLP | Deep learning | scikit-learn | Score current market risk (0–1) |
| Markowitz / MPT | Optimization | SciPy (SLSQP) | Solve Max-Sharpe + Min-Variance |
| Claude Haiku 4.5 | LLM | Anthropic API | Explain results + conversational re-optimization |

**Universe:** 288 instruments (stocks · sector/factor/bond/commodity ETFs · crypto ETFs)
**Window:** 2021-11-10 → latest row in `prices.csv` · **Risk-free rate:** 5%
**Data:** bundled `api/prices.csv` (adjusted close, built offline via yfinance)

---

## ☁️ Deployment

- **Backend** → **Railway** (root directory `api/`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`, healthcheck `/health`). Auto-deploys from `main`.
- **Frontend** → **Vercel** (`vercel.json` routes `/` → `frontend/landing.html` and `/app` → `frontend/index.html`). Auto-deploys from `main`.
- Set all secrets in the Railway **Variables** tab — see `.env.example` for the list.

---

## 🛠️ Tech stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **ML:** scikit-learn (KMeans, RandomForestRegressor, MLPRegressor)
- **Math / optimization:** NumPy, SciPy (`scipy.optimize.minimize`, SLSQP), Ledoit-Wolf covariance
- **AI:** Anthropic Claude (Haiku 4.5)
- **Data:** pandas, yfinance (offline), joblib (model cache)
- **Backend services:** Supabase (waitlist + analytics)
- **Frontend:** HTML / CSS / vanilla JS, Chart.js
- **Deploy:** Railway (API), Vercel (UI)

---

## 📜 License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

---

## 🎓 Disclaimer

This project is intended for **educational and research purposes only** and **does not constitute financial advice**. Predictions, risk scores, and rebalancing recommendations are produced by statistical models trained on historical data and may be wrong. Do **not** use this software to make real investment decisions without consulting a qualified financial professional.

Questions or issues? Contact **hi@axelpeluso.com**.
