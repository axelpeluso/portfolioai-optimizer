"""
refresh_prices.py — rebuild / extend api/prices.csv from Yahoo Finance.

Merge semantics
---------------
Freshly downloaded values win; anything the download does not supply falls back
to what is already on disk. A ticker that fails to download therefore never
wipes the history already committed — the worst case is that it stays stale.

By default this pulls the FULL history for every ticker rather than only the
missing tail. It is slower, but it self-heals the ragged tails that appear when
an earlier run partially failed (which is exactly how prices.csv ended up with
266 of 289 tickers frozen at 2026-04-24 while 23 kept updating).

Usage
-----
    python refresh_prices.py                  # merge-refresh in place
    python refresh_prices.py --dry-run        # report, write nothing
    python refresh_prices.py --full           # ignore existing file entirely
    python refresh_prices.py --since 2026-01-01   # only pull the recent tail

Exit codes: 0 = wrote (or dry-run OK) · 1 = validation failed, file untouched
            2 = nothing to do (already current)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent
TICKERS_JSON = BASE_DIR / "tickers.json"
PRICES_CSV   = BASE_DIR / "prices.csv"
DEFAULT_START = "2021-01-01"


# ── helpers ───────────────────────────────────────────────────
def log(msg: str) -> None:
    print(msg, flush=True)


def load_tickers(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.keys() if isinstance(data, dict) else data)


def preflight() -> int:
    """Check reachability before grinding through 300 downloads.

    yfinance authenticates against fc.yahoo.com before fetching anything, so
    when that host fails to resolve EVERY download fails with a DNS error that
    looks like the tickers are bad. It is a legacy host that does not resolve
    on plenty of ISPs and resolvers, while the actual data hosts are fine.

    Returns 0 = good, 1 = no connectivity at all, 2 = data hosts fine but the
    auth host is unresolvable.
    """
    import socket

    def resolves(host: str) -> bool:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            return False

    data_ok = resolves("query1.finance.yahoo.com") or resolves("query2.finance.yahoo.com")
    auth_ok = resolves("fc.yahoo.com")

    log(f"  query*.finance.yahoo.com : {'ok' if data_ok else 'UNRESOLVABLE'}")
    log(f"  fc.yahoo.com (auth)      : {'ok' if auth_ok else 'UNRESOLVABLE'}")

    if not data_ok:
        log("\n  No route to Yahoo's data hosts — this machine has no working DNS")
        log("  or internet access right now. Nothing to do but fix that first.")
        return 1
    if not auth_ok:
        log("\n  Data hosts reachable, but yfinance's auth host is not. This is a")
        log("  known yfinance issue, not a problem with your tickers. Options:")
        log("    1. pip install -U yfinance     (newer versions avoid this host)")
        log("    2. switch DNS to 1.1.1.1 or 8.8.8.8")
        return 2
    return 0


# ── source A: Yahoo's chart endpoint, no auth handshake ───────
# yfinance authenticates against fc.yahoo.com before every download. That host
# is unresolvable on many networks, and when it fails yfinance reports it as if
# the tickers were bad. This endpoint is what yfinance ultimately calls anyway,
# and it needs no cookie or crumb — so it works wherever the data hosts resolve.

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def download_chart_one(sym: str, p1: int, p2: int,
                       retries: int, pause: float) -> pd.Series | None:
    """Adjusted daily closes for one symbol. None means 'no data for this one'."""
    from urllib.error import HTTPError, URLError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sym)}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")

    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": _UA}), timeout=30) as r:
                payload = json.load(r)

            chart = payload.get("chart") or {}
            if chart.get("error"):
                raise RuntimeError(chart["error"].get("description", "api error"))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise RuntimeError("empty result")

            ts  = result.get("timestamp") or []
            ind = result.get("indicators") or {}
            # Prefer the split/dividend-adjusted series; fall back to raw close.
            adj = ((ind.get("adjclose") or [{}])[0] or {}).get("adjclose")
            if adj is None:
                adj = ((ind.get("quote") or [{}])[0] or {}).get("close")
            if not ts or not adj:
                raise RuntimeError("no price series")

            idx = pd.to_datetime([datetime.fromtimestamp(t, timezone.utc).date()
                                  for t in ts])
            s = pd.Series(adj, index=idx, name=sym, dtype="float64")
            return s[~s.index.duplicated(keep="last")]

        except HTTPError as e:
            if e.code in (404, 422):
                return None                       # delisted / unknown — do not retry
            if e.code == 429 and attempt < retries:
                time.sleep(pause * 4 * attempt)   # rate limited: back off hard
                continue
            if attempt >= retries:
                raise
            time.sleep(pause * attempt)
        except (URLError, TimeoutError, RuntimeError, ValueError, KeyError):
            if attempt >= retries:
                raise
            time.sleep(pause * attempt)
    return None


def fetch_all_chart(tickers: list[str], start: str, end: str, workers: int,
                    retries: int, pause: float) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    series: dict[str, pd.Series] = {}
    empty:  list[str] = []
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_chart_one, t, p1, p2, retries, pause): t
                   for t in tickers}
        for n, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                s = fut.result()
                if s is None or s.empty:
                    empty.append(t)
                else:
                    series[t] = s
            except Exception as e:                # noqa: BLE001 — collected below
                errors[t] = f"{type(e).__name__}: {e}"
            if n % 25 == 0 or n == len(tickers):
                log(f"  {n}/{len(tickers)} fetched "
                    f"({len(series)} ok, {len(empty)} empty, {len(errors)} error)")

    if empty:
        log(f"\n  No data returned (likely delisted/renamed) — {len(empty)}: "
            f"{', '.join(sorted(empty))}")
    if errors:
        log(f"\n  Errored — {len(errors)}:")
        for t, e in sorted(errors.items())[:15]:
            log(f"    {t}: {e}")
        if len(errors) > 15:
            log(f"    … and {len(errors) - 15} more")
    if not series:
        return pd.DataFrame()

    out = pd.concat(series.values(), axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


# ── source B: yfinance ────────────────────────────────────────
def download_batch(symbols: list[str], start: str, end: str,
                   retries: int, pause: float) -> pd.DataFrame:
    """Download adjusted closes for one batch, retrying transient failures."""
    import yfinance as yf

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                symbols, start=start, end=end,
                auto_adjust=True,     # 'Close' is then the adjusted close
                progress=False, threads=True, group_by="column",
            )
            if raw is None or raw.empty:
                raise RuntimeError("empty response")

            # Single ticker -> flat columns; multiple -> MultiIndex (field, ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": symbols[0]})
            return close.dropna(axis=1, how="all")

        except Exception as e:                      # noqa: BLE001 — report and retry
            last_err = e
            if attempt < retries:
                wait = pause * attempt
                log(f"    retry {attempt}/{retries - 1} in {wait:.0f}s — {e}")
                time.sleep(wait)

    log(f"    BATCH FAILED after {retries} attempts — {last_err}")
    return pd.DataFrame()


def fetch_all(tickers: list[str], start: str, end: str, batch_size: int,
              retries: int, pause: float) -> pd.DataFrame:
    frames, failed = [], []
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for i, batch in enumerate(batches, 1):
        log(f"  [{i}/{len(batches)}] {len(batch)} tickers: {batch[0]}…{batch[-1]}")
        got = download_batch(batch, start, end, retries, pause)
        if got.empty:
            failed.extend(batch)
        else:
            missing = [t for t in batch if t not in got.columns]
            if missing:
                failed.extend(missing)
                log(f"    no data for: {', '.join(missing)}")
            frames.append(got)
        if i < len(batches):
            time.sleep(pause)          # be polite to the API between batches

    if failed:
        log(f"\n  {len(failed)} ticker(s) returned nothing: {', '.join(sorted(failed))}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.sort_index()


def validate(new: pd.DataFrame, old: pd.DataFrame | None,
             expected: list[str], min_coverage: float) -> list[str]:
    """Return a list of problems. Empty list means the frame is safe to write."""
    errs: list[str] = []

    if new.empty:
        return ["merged frame is empty"]

    missing = sorted(set(expected) - set(new.columns))
    if missing:
        errs.append(f"{len(missing)} ticker(s) absent from output: {', '.join(missing[:8])}"
                    + ("…" if len(missing) > 8 else ""))

    # Never accept a frame that lost history relative to what we already had.
    if old is not None and not old.empty:
        if len(new) < len(old):
            errs.append(f"row count shrank: {len(old)} -> {len(new)}")
        lost = sorted(set(old.columns) - set(new.columns))
        if lost:
            errs.append(f"columns lost: {', '.join(lost[:8])}")

    # The whole point is a fresh tail — insist most tickers actually reach it.
    tail = new.tail(1)
    cov  = float(tail.notna().sum(axis=1).iloc[0]) / max(1, len(new.columns))
    if cov < min_coverage:
        errs.append(f"last row covers only {cov:.1%} of tickers "
                    f"(need {min_coverage:.0%}) — partial download")

    bad = new[(new <= 0)].notna().sum().sum()
    if bad:
        errs.append(f"{int(bad)} non-positive price value(s)")

    return errs


def coverage_report(df: pd.DataFrame, label: str) -> None:
    cov = df.notna().sum(axis=1)
    full = cov[cov >= len(df.columns) * 0.98]
    log(f"  {label}: {len(df)} rows x {len(df.columns)} tickers | "
        f"{df.index[0].date()} -> {df.index[-1].date()}")
    if len(full):
        log(f"    last date with >=98% coverage: {full.index[-1].date()}")


# ── main ──────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Refresh the bundled price history.")
    p.add_argument("--out", type=Path, default=PRICES_CSV)
    p.add_argument("--tickers-file", type=Path, default=TICKERS_JSON)
    p.add_argument("--start", default=DEFAULT_START,
                   help="first date to pull when rebuilding (default %(default)s)")
    p.add_argument("--since", default=None,
                   help="only pull from this date (faster; does not heal older gaps)")
    p.add_argument("--full", action="store_true",
                   help="ignore the existing CSV instead of merging into it")
    p.add_argument("--source", choices=("chart", "yfinance"), default="chart",
                   help="chart = Yahoo's chart endpoint (no auth host, default); "
                        "yfinance = the library, which needs fc.yahoo.com to resolve")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel requests for --source chart")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--pause", type=float, default=2.0,
                   help="seconds between batches / retry backoff unit")
    p.add_argument("--round", type=int, default=4, dest="round_dp",
                   help="decimal places to store (default 4 — halves the file size)")
    p.add_argument("--min-coverage", type=float, default=0.90)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="only run the connectivity preflight, then exit")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    if not args.skip_preflight:
        log("Connectivity check…")
        pf = preflight()
        # The chart endpoint needs no auth host, so pf == 2 is irrelevant to it.
        fatal = (pf == 1) or (pf == 2 and args.source == "yfinance")
        if args.check:
            return 1 if fatal else 0
        if fatal:
            if pf == 2:
                log("\n  --source chart avoids this host entirely (it is the default).")
            return 1
        if pf == 2:
            log("  (irrelevant for --source chart — continuing)\n")
    elif args.check:
        return 0

    tickers = load_tickers(args.tickers_file)
    log(f"Tickers: {len(tickers)} from {args.tickers_file.name}")

    old = None
    if args.out.exists() and not args.full:
        old = pd.read_csv(args.out, index_col=0, parse_dates=True)
        old.index = pd.to_datetime(old.index).tz_localize(None).normalize()
        coverage_report(old, "existing")

    start = args.since or args.start
    end   = (date.today() + timedelta(days=1)).isoformat()   # end is exclusive
    log(f"\nDownloading {start} -> {end} via {args.source} …")

    if args.source == "chart":
        fresh = fetch_all_chart(tickers, start, end, args.workers,
                                args.retries, args.pause)
    else:
        fresh = fetch_all(tickers, start, end, args.batch_size,
                          args.retries, args.pause)
    if fresh.empty:
        log("\nFAILED: no data downloaded at all. Existing file left untouched.")
        return 1
    coverage_report(fresh, "downloaded")

    # Fresh values win; fall back to what we already had.
    merged = fresh.combine_first(old) if old is not None else fresh
    merged = merged.reindex(columns=[t for t in tickers if t in merged.columns])
    merged = merged.sort_index()
    merged.index.name = "Date"

    log("")
    coverage_report(merged, "merged")

    errs = validate(merged, old, tickers, args.min_coverage)
    if errs:
        log("\nVALIDATION FAILED — existing file left untouched:")
        for e in errs:
            log(f"  - {e}")
        return 1

    if old is not None:
        added = len(merged) - len(old)
        log(f"\n  +{added} new row(s); last date "
            f"{old.index[-1].date()} -> {merged.index[-1].date()}")
        if added == 0 and merged.index[-1] == old.index[-1]:
            # Still worth writing if we healed gaps in existing rows.
            filled = int(old.isna().sum().sum() - merged.isna().sum().sum())
            if filled <= 0:
                log("  Already current — nothing to write.")
                return 2
            log(f"  Backfilled {filled} previously-missing value(s).")

    if args.dry_run:
        log("\nDry run — nothing written.")
        return 0

    out = merged.round(args.round_dp)
    tmp = args.out.with_suffix(".csv.tmp")
    out.to_csv(tmp)
    os.replace(tmp, args.out)          # atomic: never leave a half-written CSV
    size = args.out.stat().st_size / 1e6
    log(f"\nWrote {args.out.name} ({size:.2f} MB, {len(out)} rows x {len(out.columns)} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
