# 📈 PortfolioAI — AI-Powered Portfolio Rebalancing

> An end-to-end machine-learning pipeline that ingests live market data, profiles each asset, predicts forward returns, scores market risk, and emits actionable BUY / SELL / HOLD instructions in dollars — all served through a FastAPI backend and a TradingView-style web UI.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6?logo=scipy&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Highlights

| Metric                        | Before  | After   | Δ           |
|-------------------------------|---------|---------|-------------|
| **Sharpe Ratio**              | 0.766   | 1.187   | **+0.421**  |
| **Annualized Volatility**     | —       | **−8.09%** | reduced  |
| **Annualized Return**         | ≈ same  | ≈ same  | preserved   |
| **MLP Risk Score (0–1)**      | —       | **0.22 (LOW)** | — |

> Result: portfolio risk-adjusted performance improved by **~55%** while keeping headline return essentially flat.

---

## 🧠 What it does

PortfolioAI combines **unsupervised learning**, **supervised learning**, **deep learning**, and **classical financial theory** into a single rebalancing engine:

1. **K-Means Clustering** profiles each asset as `🚀 Growth` / `⚖️ Moderate` / `🛡️ Defensive` based on return / volatility / Sharpe.
2. **Random Forest Regressor** (per-ticker) predicts the next-quarter return from a rolling-window feature set.
3. **MLP Neural Network** evaluates the *current* portfolio's market regime and emits a 0–1 **risk score**.
4. **Modern Portfolio Theory** (Markowitz, SLSQP-solved) finds the Max-Sharpe and Min-Variance weights — with bounds that adapt to the MLP's risk score.
5. **Rebalancing Engine** translates optimal weights into concrete `BUY $X` / `SELL $Y` / `HOLD` actions vs. current holdings.

---

## 🏗️ Architecture

```
                        ┌────────────────────────┐
                        │   Browser (UI)         │
                        │   frontend/index.html  │
                        │   Chart.js + vanilla JS│
                        └────────────┬───────────┘
                                     │  POST /optimize
                                     ▼
                        ┌────────────────────────┐
                        │   FastAPI server       │
                        │   api/main.py          │
                        │   CORS · pydantic v2   │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   ML + MPT engine      │
                        │   api/optimizer.py     │
                        └────────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
       ┌─────────────┐       ┌──────────────┐       ┌──────────────┐
       │   yfinance  │       │ scikit-learn │       │    SciPy     │
       │  (prices)   │       │ KMeans / RF  │       │ optimize     │
       │             │       │     MLP      │       │   (SLSQP)    │
       └─────────────┘       └──────────────┘       └──────────────┘
```

---

## 📂 Project structure

```
Final Project/
├── api/
│   ├── main.py                ← FastAPI app + routes
│   ├── optimizer.py           ← ML + MPT pipeline
│   └── requirements.txt       ← Python deps
├── frontend/
│   └── index.html             ← TradingView-style UI (Chart.js)
├── portolio_optimizer.ipynb   ← Full notebook walk-through
├── .gitignore
└── README.md
```

---

## 🖼️ Screenshots

> _Replace these placeholders with real screenshots after committing._

| View | File |
|------|------|
| Dashboard          | `docs/screenshot-dashboard.png` |
| Optimization chart | `docs/screenshot-optimizer.png` |
| API Swagger docs   | `docs/screenshot-swagger.png`   |

---

## 🚀 Installation

> Requires **Python 3.11+** (developed on 3.13).

```bash
# Clone
git clone https://github.com/<your-username>/portfolio-ai.git
cd portfolio-ai

# Create + activate a virtual env
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt
```

---

## ▶️ How to run

### 1. The notebook (full ML walk-through)

```bash
jupyter notebook portolio_optimizer.ipynb
```

### 2. The API

```bash
cd api
uvicorn main:app --reload --port 8000
# or, if 8000 is taken on Windows:
uvicorn main:app --reload --port 8080
```

Then open:

- 🌐 **API root** → http://127.0.0.1:8000/
- 📚 **Swagger UI** → http://127.0.0.1:8000/docs
- 📘 **ReDoc** → http://127.0.0.1:8000/redoc

### 3. The frontend

Open `frontend/index.html` in your browser (double-click, or use VS Code's _Live Server_). Make sure the API is running first — the page will call it directly via CORS.

---

## 🔌 API endpoints

| Method | Path        | Description                                    |
|-------:|-------------|------------------------------------------------|
| `GET`  | `/`         | Health banner + version + link to docs         |
| `GET`  | `/health`   | Liveness probe                                 |
| `POST` | `/optimize` | Run the full ML pipeline + return rebalancing  |

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

Constraints:
- 2 ≤ `len(tickers)` ≤ 15
- `current_holdings` is optional (defaults to `{}`); when empty, the engine assumes a **$10,000** notional portfolio for sizing trades.

### `POST /optimize` — response shape

```json
{
  "success": true,
  "data": {
    "tickers":          ["AAPL", "..."],
    "risk_score":       0.22,
    "risk_level":       "LOW",
    "cluster_map":      { "AAPL": "🚀 Growth", "...": "..." },
    "expected_returns": { "AAPL": 0.18, "...": "..." },
    "rebalancing": {
      "AAPL": {
        "current_value":  5000,
        "current_weight": 0.45,
        "optimal_weight": 0.30,
        "target_value":   3300.00,
        "trade_amount":  -1700.00,
        "action":        "SELL",
        "rf_signal":     "Bullish",
        "cluster":       "🚀 Growth"
      }
    },
    "total_value":     11500,
    "optimization": {
      "optimal_weights":    { "...": 0.30 },
      "min_var_weights":    { "...": 0.10 },
      "max_sharpe_metrics": { "return": 0.21, "volatility": 0.17, "sharpe": 1.19 },
      "min_var_metrics":    { "return": 0.09, "volatility": 0.07, "sharpe": 0.55 }
    },
    "current_metrics": { "return": 0.20, "volatility": 0.18, "sharpe": 0.77 }
  }
}
```

---

## 📊 Models cheat-sheet

| Model            | Type           | Library              | Role                                  |
|------------------|----------------|----------------------|---------------------------------------|
| K-Means          | Unsupervised   | scikit-learn         | Cluster assets by behavioral profile  |
| Random Forest    | Supervised     | scikit-learn         | Forecast forward returns per ticker   |
| MLP              | Deep learning  | scikit-learn         | Score current market risk (0–1)       |
| Markowitz / MPT  | Optimization   | SciPy (SLSQP)        | Solve for Max-Sharpe + Min-Variance   |

**Universe:** AAPL · MSFT · GOOGL · JPM · BND · GLD · AMZN
**Window:** 2021-01-01 → 2025-12-31
**Risk-free rate:** 5%

---

## 🛠️ Tech stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **ML:** scikit-learn (KMeans, RandomForestRegressor, MLPRegressor)
- **Math / optimization:** NumPy, SciPy (`scipy.optimize.minimize`, SLSQP)
- **Data:** yfinance, pandas
- **Frontend:** HTML / CSS / vanilla JS, Chart.js
- **Tooling:** Jupyter, VS Code

---

## 📜 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

---

## 🎓 Academic disclaimer

This project was developed for an **academic course** (MAI500 — Atlantis University, Winter 2026 term).
It is intended for **educational and research purposes only** and **does not constitute financial advice**. Predictions, risk scores, and rebalancing recommendations are produced by statistical models trained on historical data and may be wrong. Do **not** use this software to make real investment decisions without consulting a qualified financial professional.
