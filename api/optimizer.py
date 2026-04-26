# ============================================================
# optimizer.py — Core ML + MPT Logic (Fixed)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ── CONSTANTS ────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"
RISK_FREE  = 0.05

# ── CACHE + RETRY ────────────────────────────────────────────
import time
_cache = {}
CACHE_TTL    = 3600  # 1 hour
MAX_RETRIES  = 4
BASE_BACKOFF = 5     # seconds


def _download_with_retry(tickers: list) -> pd.DataFrame:
    """yf.download with exponential backoff on rate-limit errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(
                tickers, start=START_DATE, end=END_DATE,
                progress=False, auto_adjust=True, threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
            last_err = RuntimeError("yfinance returned empty frame")
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate" not in msg and "too many" not in msg and attempt > 0:
                break
        wait = BASE_BACKOFF * (2 ** attempt)
        print(f"⏳ yfinance retry {attempt + 1}/{MAX_RETRIES} in {wait}s ({last_err})")
        time.sleep(wait)
    raise RuntimeError(
        "Yahoo Finance rate-limited or unreachable after retries. "
        "Please wait a few minutes and try again."
    ) from last_err


# ── 1. DATA ──────────────────────────────────────────────────
def fetch_data(tickers: list) -> tuple:
    """Download price data with caching + retry to avoid rate limits."""
    cache_key = ",".join(sorted(tickers))
    now = time.time()

    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            print("📦 Using cached data")
            return cached_data

    raw = _download_with_retry(tickers)

    # Handle both single and multi-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(axis=1, how='all').ffill().dropna()

    valid_tickers = list(prices.columns)
    if len(valid_tickers) < 2:
        raise ValueError(
            f"Only {len(valid_tickers)} ticker(s) returned valid data. "
            "Please check your tickers and try again."
        )

    returns = prices.pct_change().dropna()
    result  = (prices, returns, valid_tickers)

    _cache[cache_key] = (now, result)
    return result


# ── 2. FEATURES ──────────────────────────────────────────────
def build_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per stock."""
    feats = pd.DataFrame(index=prices.columns)
    feats['annual_return']     = returns.mean() * 252
    feats['annual_volatility'] = returns.std()  * np.sqrt(252)
    feats['sharpe_ratio']      = (feats['annual_return'] /
                                  feats['annual_volatility'].replace(0, np.nan)).fillna(0)

    def max_drawdown(r):
        try:
            cum = (1 + r).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax()
            return dd.min()
        except Exception:
            return 0.0

    feats['max_drawdown'] = returns.apply(max_drawdown)

    try:
        n = min(126, len(prices) - 2)
        feats['momentum_6m'] = prices.iloc[-n:].pct_change(n - 1).iloc[-1]
    except Exception:
        feats['momentum_6m'] = 0.0

    feats['skewness'] = returns.skew()
    return feats.fillna(0)


# ── 3. K-MEANS ───────────────────────────────────────────────
def run_kmeans(features: pd.DataFrame) -> pd.DataFrame:
    """Cluster stocks into Growth / Moderate / Defensive."""
    features = features.copy()

    cols = [c for c in ['annual_return', 'annual_volatility', 'sharpe_ratio']
            if c in features.columns]
    data = features[cols].fillna(0)

    n_stocks   = len(data)
    n_clusters = min(3, n_stocks)

    if n_stocks < 2:
        features['cluster_label'] = '🚀 Growth'
        return features

    # StandardScaler needs variance — add tiny noise if all identical
    if data.std().max() == 0:
        features['cluster_label'] = '🚀 Growth'
        return features

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features['cluster'] = km.fit_predict(scaled)

    median_ret = features['annual_return'].median()
    label_map  = {}

    for c in features['cluster'].unique():
        grp     = features[features['cluster'] == c]
        avg_vol = grp['annual_volatility'].mean()
        avg_ret = grp['annual_return'].mean()
        if avg_vol < 0.10:
            label_map[c] = '🛡️ Defensive'
        elif avg_ret >= median_ret:
            label_map[c] = '🚀 Growth'
        else:
            label_map[c] = '⚖️ Moderate'

    features['cluster_label'] = features['cluster'].map(label_map)
    return features


# ── 4. RANDOM FOREST ─────────────────────────────────────────
def build_stock_features(stock_returns: pd.Series,
                          window: int = 60, forward: int = 63):
    X, y = [], []
    for i in range(window, len(stock_returns) - forward):
        w  = stock_returns.iloc[i - window:i]
        fr = (1 + stock_returns.iloc[i:i + forward]).prod() - 1
        X.append([
            w.mean() * 252,
            w.std()  * np.sqrt(252),
            w.iloc[-5:].mean()  * 252,
            w.iloc[-20:].mean() * 252,
            w.iloc[-20:].std()  * np.sqrt(252),
            (w > 0).sum() / window,
            w.iloc[-1] / (w.mean() + 1e-9) - 1,
        ])
        y.append(fr)

    if not X:
        return np.empty((0, 7)), np.empty(0)

    X, y = np.array(X), np.array(y)
    if len(y) < 4:
        return X, y
    mask = np.abs(y - y.mean()) < 2 * y.std()
    return X[mask], y[mask]


def run_random_forest(returns: pd.DataFrame) -> tuple:
    """Return predicted annual returns + R² scores per ticker."""
    rf_predictions, rf_scores = {}, {}

    for ticker in returns.columns:
        try:
            sr   = returns[ticker].dropna()
            X, y = build_stock_features(sr)

            if len(X) < 50:
                continue

            split = int(len(X) * 0.8)
            Xtr, Xte = X[:split], X[split:]
            ytr, yte = y[:split], y[split:]

            sc     = StandardScaler()
            Xtr_sc = sc.fit_transform(Xtr)
            Xte_sc = sc.transform(Xte)

            rf = RandomForestRegressor(
                n_estimators=200, max_depth=4,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
            rf.fit(Xtr_sc, ytr)
            rf_scores[ticker] = r2_score(yte, rf.predict(Xte_sc))

            last = returns[ticker].iloc[-60:]
            lX   = np.array([[
                last.mean() * 252,
                last.std()  * np.sqrt(252),
                last.iloc[-5:].mean()  * 252,
                last.iloc[-20:].mean() * 252,
                last.iloc[-20:].std()  * np.sqrt(252),
                (last > 0).sum() / 60,
                last.iloc[-1] / (last.mean() + 1e-9) - 1,
            ]])
            rf_predictions[ticker] = rf.predict(sc.transform(lX))[0] * 4

        except Exception:
            continue

    return rf_predictions, rf_scores


# ── 5. MLP ───────────────────────────────────────────────────
def build_portfolio_features(returns: pd.DataFrame,
                              window: int = 60, forward: int = 21):
    X, y    = [], []
    tickers = returns.columns.tolist()

    for i in range(window, len(returns) - forward):
        wd  = returns.iloc[i - window:i]
        row = []
        for t in tickers:
            c = wd[t]
            row.extend([
                c.mean() * 252,
                c.std()  * np.sqrt(252),
                (c > 0).sum() / window,
                c.iloc[-5:].mean() * 252,
            ])
        corr = wd.corr()
        up   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        row.append(float(up.stack().mean()))

        port_vol = returns.iloc[i:i + forward].mean(axis=1).std() * np.sqrt(252)
        X.append(row)
        y.append(port_vol)

    if not X:
        return np.empty((0, len(tickers) * 4 + 1)), np.empty(0)

    X, y = np.array(X), np.array(y)
    mask = np.abs(y - y.mean()) < 2.5 * y.std()
    return X[mask], y[mask]


def run_mlp(returns: pd.DataFrame) -> float:
    """Return current portfolio risk score 0–1."""
    try:
        X, y = build_portfolio_features(returns)

        if len(X) < 60:
            return 0.3  # default if not enough data

        y_min, y_max = y.min(), y.max()
        yn = (y - y_min) / (y_max - y_min + 1e-9)

        split  = int(len(X) * 0.8)
        sc     = StandardScaler()
        Xtr_sc = sc.fit_transform(X[:split])

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation='relu',
            solver='adam', learning_rate='adaptive', max_iter=500,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20
        )
        mlp.fit(Xtr_sc, yn[:split])

        last = returns.iloc[-60:]
        lX   = []
        for t in returns.columns:
            c = last[t]
            lX.extend([
                c.mean() * 252,
                c.std()  * np.sqrt(252),
                (c > 0).sum() / 60,
                c.iloc[-5:].mean() * 252,
            ])
        corr = last.corr()
        up   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        lX.append(float(up.stack().mean()))

        score = mlp.predict(sc.transform([lX]))[0]
        return float(np.clip(score, 0, 1))

    except Exception:
        return 0.3


# ── 6. MPT OPTIMIZER ─────────────────────────────────────────
def portfolio_performance(w, exp_ret, cov):
    r   = np.dot(w, exp_ret)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    vol = max(vol, 1e-9)
    return r, vol, (r - RISK_FREE) / vol


def run_optimizer(tickers, expected_returns, cov_matrix,
                  risk_score, cluster_map) -> dict:
    n         = len(tickers)
    max_w     = 0.40 if risk_score < 0.35 else (0.30 if risk_score < 0.65 else 0.20)
    min_def   = 0.10 if risk_score > 0.50 else 0.02
    defensive = [t for t in tickers if '🛡️' in cluster_map.get(t, '')]

    bounds      = [(min_def if t in defensive else 0.02, max_w) for t in tickers]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n

    # Max Sharpe
    res   = minimize(
        lambda w: -portfolio_performance(w, expected_returns, cov_matrix)[2],
        w0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    opt_w              = res.x
    opt_r, opt_v, opt_s = portfolio_performance(opt_w, expected_returns, cov_matrix)

    # Min Variance
    res_mv             = minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        w0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    mv_w               = res_mv.x
    mv_r, mv_v, mv_s   = portfolio_performance(mv_w, expected_returns, cov_matrix)

    return {
        'optimal_weights'    : dict(zip(tickers, opt_w.tolist())),
        'min_var_weights'    : dict(zip(tickers, mv_w.tolist())),
        'max_sharpe_metrics' : {'return': opt_r, 'volatility': opt_v, 'sharpe': opt_s},
        'min_var_metrics'    : {'return': mv_r, 'volatility': mv_v, 'sharpe': mv_s},
    }


# ── 7. BLEND RETURNS ─────────────────────────────────────────
def blend_returns(ticker, rf_pred, hist_ret, r2, blend=0.3):
    weight = max(0.0, min(1.0, r2)) * blend
    return weight * rf_pred + (1 - weight) * hist_ret


# ── 8. MASTER FUNCTION ───────────────────────────────────────
def run_full_analysis(tickers: list, current_holdings: dict) -> dict:
    # 1. Data
    prices, returns, tickers = fetch_data(tickers)

    if prices.empty or len(prices) < 100:
        raise ValueError("Not enough historical data. Check your tickers.")

    # 2. Features
    features = build_features(prices, returns)

    # 3. K-Means
    features    = run_kmeans(features)
    cluster_map = features['cluster_label'].to_dict()

    # 4. Random Forest
    rf_preds, rf_scores = run_random_forest(returns)

    # 5. MLP
    risk_score = run_mlp(returns)

    # 6. Blend returns
    exp_returns = np.array([
        blend_returns(
            t,
            rf_preds.get(t, features.loc[t, 'annual_return']),
            features.loc[t, 'annual_return'],
            rf_scores.get(t, 0)
        )
        for t in tickers
    ])

    # 7. Covariance
    cov_matrix = returns[tickers].cov() * 252

    # 8. Optimize
    opt_result  = run_optimizer(tickers, exp_returns, cov_matrix,
                                risk_score, cluster_map)
    opt_weights = opt_result['optimal_weights']

    # 9. Rebalancing
    total_value = sum(current_holdings.get(t, 0) for t in tickers)
    if total_value == 0:
        total_value = 10000

    rebalancing = {}
    for t in tickers:
        curr_val = float(current_holdings.get(t, 0))
        opt_val  = opt_weights[t] * total_value
        diff     = opt_val - curr_val
        rebalancing[t] = {
            'current_value'  : curr_val,
            'current_weight' : curr_val / total_value,
            'optimal_weight' : opt_weights[t],
            'target_value'   : round(opt_val, 2),
            'trade_amount'   : round(diff, 2),
            'action'         : ('BUY'  if diff >  50 else
                                'SELL' if diff < -50 else 'HOLD'),
            'rf_signal'      : ('Bullish' if rf_preds.get(t, 0) >  0.10 else
                                'Bearish' if rf_preds.get(t, 0) <  0    else 'Neutral'),
            'cluster'        : cluster_map.get(t, ''),
        }

    # 10. Current portfolio metrics
    curr_w               = np.array([current_holdings.get(t, 0) / total_value
                                     for t in tickers])
    curr_r, curr_v, curr_s = portfolio_performance(curr_w, exp_returns, cov_matrix)

    return {
        'tickers'         : tickers,
        'risk_score'      : risk_score,
        'risk_level'      : ('LOW'    if risk_score < 0.35 else
                             'MEDIUM' if risk_score < 0.65 else 'HIGH'),
        'cluster_map'     : cluster_map,
        'expected_returns': dict(zip(tickers, exp_returns.tolist())),
        'rebalancing'     : rebalancing,
        'total_value'     : total_value,
        'optimization'    : opt_result,
        'current_metrics' : {'return': curr_r, 'volatility': curr_v, 'sharpe': curr_s},
    }
