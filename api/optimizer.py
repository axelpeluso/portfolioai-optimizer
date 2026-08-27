# ============================================================
# optimizer.py — Core ML + MPT Logic (Fixed)
# ============================================================

import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import scipy.special
import warnings
import logging
import hashlib
import joblib
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".model_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _data_version() -> str:
    """Fingerprint of the bundled price file (size + mtime).

    Folded into the cache key so a refreshed prices.csv invalidates every
    cached RF/MLP result — otherwise new data would silently serve old
    predictions for any ticker set that had been optimized before.
    """
    try:
        st = os.stat(CSV_PATH)
        return f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "nofile"


def _cache_key(tickers: list) -> str:
    key = "_".join(sorted(tickers)) + "|" + _data_version()
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _save_cache(key, kmeans_feats, rf_preds, rf_scores, risk_score):
    joblib.dump({
        'kmeans_feats': kmeans_feats,
        'rf_preds':     rf_preds,
        'rf_scores':    rf_scores,
        'risk_score':   risk_score,
    }, CACHE_DIR / f"{key}.pkl")


def _load_cache(key):
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


logging.basicConfig(
    filename='portfolio.log',
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s'
)
warnings.filterwarnings('ignore', category=UserWarning)  # solo sklearn verbosity

# ── CONSTANTS ────────────────────────────────────────────────
START_DATE = "2021-11-10"
END_DATE   = None          # None → run to whatever the CSV ends at, so a
                           # refreshed prices.csv is picked up automatically
RISK_FREE  = 0.05

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.csv")


# ── 1. DATA ──────────────────────────────────────────────────
def fetch_data(tickers: list) -> tuple:
    """Load price data from bundled CSV and compute daily returns."""
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    available = [t for t in tickers if t in df.columns]
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"Warning: tickers not in CSV (skipping): {missing}")
    prices = df[available].loc[START_DATE:END_DATE]
    prices = prices.dropna(axis=1, how="all")
    valid_tickers = list(prices.columns)
    returns = prices.pct_change().dropna()
    return prices, returns, valid_tickers


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
        features['cluster_label'] = 'Growth'
        return features

    # StandardScaler needs variance — add tiny noise if all identical
    if data.std().max() == 0:
        features['cluster_label'] = 'Growth'
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
            label_map[c] = 'Defensive'
        elif avg_ret >= median_ret:
            label_map[c] = 'Growth'
        else:
            label_map[c] = 'Moderate'

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
            quarterly = rf.predict(sc.transform(lX))[0]
            rf_predictions[ticker] = (1 + quarterly) ** 4 - 1

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

        p5, p95 = np.percentile(y, 5), np.percentile(y, 95)
        yn = (y - p5) / (p95 - p5 + 1e-9)
        yn = scipy.special.expit(2.5 * (yn - 0.5))  # sigmoid centered at 0.5, gentle slope

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

        raw = float(mlp.predict(sc.transform([lX]))[0])
        return float(scipy.special.expit(2.5 * (raw - 0.5)))

    except Exception:
        return 0.3


# ── 6. MPT OPTIMIZER ─────────────────────────────────────────
def portfolio_performance(w, exp_ret, cov):
    r   = np.dot(w, exp_ret)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    vol = max(vol, 1e-9)
    return r, vol, (r - RISK_FREE) / vol


# Penalty strength for the opt-in "minimize trading" mode. Exposed as named
# levels rather than a raw number — a lambda is not a user-facing concept.
TURNOVER_LAMBDA = {'light': 0.5, 'moderate': 2.0, 'strong': 6.0}


def run_optimizer(tickers, expected_returns, cov_matrix,
                  risk_score, cluster_map,
                  current_weights=None, turnover_penalty=None,
                  tax_weights=None) -> dict:
    """Solve Max-Sharpe and Min-Variance.

    With `turnover_penalty` unset this is exactly the original solver — same
    objective, same starting point, same result. That equivalence is asserted by
    test_optimizer_modes.py and must not be broken casually: users who never
    opt in should see the numbers they have always seen.

    When set, the objective becomes

        maximize  Sharpe − λ · Σ pᵢ·(wᵢ − wᵢ_current)²

    Quadratic rather than |Δw| so the objective stays smooth for SLSQP. `pᵢ` is
    1 everywhere for plain "minimize trading"; `tax_weights` replaces it with a
    per-asset cost so that selling a large embedded gain is discouraged more
    than selling a loss.
    """
    n         = len(tickers)
    max_w     = 0.40 if risk_score < 0.35 else (0.30 if risk_score < 0.65 else 0.20)
    min_def   = 0.10 if risk_score > 0.50 else 0.02
    defensive = [t for t in tickers if 'Defensive' in cluster_map.get(t, '')]

    bounds      = [(min_def if t in defensive else 0.02, max_w) for t in tickers]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n

    lam = TURNOVER_LAMBDA.get(turnover_penalty) if isinstance(turnover_penalty, str) \
          else turnover_penalty
    use_penalty = bool(lam) and current_weights is not None
    if use_penalty:
        cw = np.array([float(current_weights.get(t, 0.0)) for t in tickers])
        pw = np.array([float((tax_weights or {}).get(t, 1.0)) for t in tickers])
        # Start from where the portfolio already is: better conditioned for a
        # penalised objective, and it keeps the solver near a low-turnover basin.
        w0 = cw if cw.sum() > 0 else w0

    def neg_sharpe(w):
        s = -portfolio_performance(w, expected_returns, cov_matrix)[2]
        if use_penalty:
            s += lam * float(np.sum(pw * (w - cw) ** 2))
        return s

# Max Sharpe
    res = minimize(
        neg_sharpe,
        w0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    if not res.success:
        import logging
        logging.warning(f"Max-Sharpe optimizer did not converge: {res.message}")
    opt_w = res.x
    opt_r, opt_v, opt_s = portfolio_performance(opt_w, expected_returns, cov_matrix)

    # Min Variance — deliberately unpenalised and started from equal weights.
    # It is a reference point ("what is the lowest-risk mix available"), not a
    # proposal to trade toward, so turnover cost does not belong in it.
    res_mv = minimize(
        lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
        np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=constraints
    )
    if not res_mv.success:
        import logging
        logging.warning(f"Min-Var optimizer did not converge: {res_mv.message}")
    mv_w = res_mv.x
    mv_r, mv_v, mv_s = portfolio_performance(mv_w, expected_returns, cov_matrix)

    return {
        'optimal_weights'    : {t: float(w) for t, w in zip(tickers, opt_w)},
        'max_sharpe_metrics' : {'return': float(opt_r),
                                'volatility': float(opt_v),
                                'sharpe': float(opt_s)},
        'min_var_weights'    : {t: float(w) for t, w in zip(tickers, mv_w)},
        'min_var_metrics'    : {'return': float(mv_r),
                                'volatility': float(mv_v),
                                'sharpe': float(mv_s)},
    }


# ── 7. BLEND RETURNS ─────────────────────────────────────────
def blend_returns(ticker, rf_pred, hist_ret, r2, blend=0.3):
    weight = max(0.0, min(1.0, r2)) * blend
    return weight * rf_pred + (1 - weight) * hist_ret


# ── 8. MASTER FUNCTION ───────────────────────────────────────
def run_full_analysis(tickers: list, current_holdings: dict,
                      turnover_penalty=None, tax_weights=None) -> dict:
    """Full pipeline. With both optional arguments unset the result is identical
    to the pre-tax-disclosure implementation — see test_optimizer_modes.py."""
    # 1. Data
    prices, returns, tickers = fetch_data(tickers)

    if prices.empty or len(prices) < 100:
        raise ValueError("Not enough historical data. Check your tickers.")

    # 2. Features
    features = build_features(prices, returns)

    # 3. K-Means
    features    = run_kmeans(features)
    cluster_map = features['cluster_label'].to_dict()

    # 4–5. RF + MLP — use cache if available
    cache_key = _cache_key(tickers)
    cached    = _load_cache(cache_key)

    if cached:
        logging.warning(f"Cache hit for {cache_key} — skipping RF+MLP training")
        rf_preds   = cached['rf_preds']
        rf_scores  = cached['rf_scores']
        risk_score = cached['risk_score']
    else:
        logging.warning(f"Cache miss for {cache_key} — training RF+MLP")
        rf_preds, rf_scores = run_random_forest(returns)
        risk_score          = run_mlp(returns)
        _save_cache(cache_key, features, rf_preds, rf_scores, risk_score)

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

    # 7. Covariance con Ledoit-Wolf shrinkage
    from sklearn.covariance import LedoitWolf
    lw         = LedoitWolf().fit(returns[tickers])
    cov_matrix = lw.covariance_ * 252

    # 8. Optimize
    #    Current weights are needed before the solve for the opt-in turnover
    #    penalty, so the portfolio total moves up from step 9.
    total_value = sum(current_holdings.get(t, 0) for t in tickers)
    if total_value == 0:
        total_value = 10000
    current_weights = {t: current_holdings.get(t, 0) / total_value for t in tickers}

    opt_result  = run_optimizer(tickers, exp_returns, cov_matrix,
                                risk_score, cluster_map,
                                current_weights=current_weights,
                                turnover_penalty=turnover_penalty,
                                tax_weights=tax_weights)
    opt_weights = opt_result['optimal_weights']

    # 9. Rebalancing

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
