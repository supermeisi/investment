import concurrent.futures
import csv
import itertools
import json
import os
import random
import sqlite3
import threading
import time
import urllib.request
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Thread-sicherer, fensterloser Backend

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
import yfinance as yf
from joblib import dump, load

DATABASE_PATH = "ranking.db"
PLOTS_DIR = "plots"

# Local cache. Override with CACHE_DIR=/some/path if desired.
CACHE_DIR = Path(os.getenv("CACHE_DIR", "data"))
PRICE_CACHE_DIR = CACHE_DIR / "prices"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
FUNDAMENTAL_CACHE_DIR = CACHE_DIR / "fundamentals"
TICKER_MAP_CACHE_DIR = CACHE_DIR / "ticker_map"
INSTRUMENT_CACHE_FILE = CACHE_DIR / "xetra_isins.parquet"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_BUNDLE_FILE = MODELS_DIR / "production_models.joblib"

# Cache behavior:
#   FORCE_REFRESH=1 -> redownload master data, fundamentals and full price histories.
#   OFFLINE=1       -> use local cache only; never call Yahoo/Xetra.
FORCE_REFRESH = os.getenv("FORCE_REFRESH", "0").lower() in {"1", "true", "yes", "on"}
OFFLINE = os.getenv("OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "24"))
PRICE_OVERLAP_DAYS = int(os.getenv("PRICE_OVERLAP_DAYS", "7"))
FEATURE_RECOMPUTE_ROWS = int(os.getenv("FEATURE_RECOMPUTE_ROWS", "450"))

DB_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()

COMPLETED_COUNT = 0
TOTAL_ITEMS = 0
START_TIME = 0.0

FORWARD_DAYS = 126
TARGET_RETURN_THRESHOLD = 0.10
MIN_TRAIN_DATES = 252
WALK_FORWARD_TEST_DAYS = 126
WALK_FORWARD_FOLDS = 3

# Random-forest parallelism. Default to 1 because some Python 3.14 /
# scikit-learn combinations emit repeated parallel-config warnings with
# multi-worker RandomForest execution. Set RF_N_JOBS=4 (or another value)
# explicitly if you want to test parallel execution on your environment.
RF_N_JOBS = int(os.getenv("RF_N_JOBS", "1"))
if RF_N_JOBS == 0 or RF_N_JOBS < -1:
    raise ValueError("RF_N_JOBS must be -1 or a positive integer.")

# Training progress. Random forests are built in small warm-start batches so the
# terminal can show real tree-level progress instead of appearing frozen during
# a long .fit() call. Set RF_PROGRESS=0 to disable the progress bar or change
# RF_PROGRESS_BATCH to control how many trees are added per update.
RF_PROGRESS = os.getenv("RF_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}
RF_PROGRESS_BATCH = max(1, int(os.getenv("RF_PROGRESS_BATCH", "10")))
RF_PROGRESS_WIDTH = max(10, int(os.getenv("RF_PROGRESS_WIDTH", "30")))

# This exact warning can be emitted internally by scikit-learn on some
# Python 3.14 builds. Our code does not call sklearn.utils.parallel.delayed
# directly, so suppress only this specific noisy warning; all other warnings
# remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


class CalibratedRFClassifier:
    """Random forest plus an optional monotonic probability calibrator."""

    def __init__(self, model: RandomForestClassifier, calibrator: IsotonicRegression | None = None):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            prob = self.calibrator.predict(raw)
        else:
            prob = raw
        prob = np.clip(prob, 0.0, 1.0)
        return np.column_stack([1.0 - prob, prob])


def initialize_database(db_path: str = DATABASE_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio (
                isin TEXT PRIMARY KEY,
                ticker TEXT,
                sector TEXT,
                industry TEXT,
                status TEXT NOT NULL,
                error TEXT,
                price REAL,
                div_yield REAL,
                payout_ratio REAL,
                pe_ratio REAL,
                debt_to_equity REAL,
                positive_cashflow INTEGER,
                one_year_return REAL,
                five_year_return REAL,
                max_drawdown REAL,
                sharpe REAL,
                trend_sma200 REAL,
                ai_prob REAL,
                score REAL
            )
            """
        )
        # Lightweight schema migration for databases created by older versions.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(portfolio)")}
        for column, sql_type in {
            "expected_return": "REAL",
            "history_years": "REAL",
        }.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE portfolio ADD COLUMN {column} {sql_type}")
        conn.commit()


def save_instrument(
    isin: str,
    ticker: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    metrics: dict | None = None,
    status: str = "processed",
    error: str | None = None,
    db_path: str = DATABASE_PATH,
) -> None:
    pos_cf = None
    if metrics and metrics.get("Positive_CashFlow") is not None:
        pos_cf = 1 if metrics.get("Positive_CashFlow") else 0

    with DB_LOCK:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolio (
                    isin, ticker, sector, industry, status, error, price,
                    div_yield, payout_ratio, pe_ratio, debt_to_equity,
                    positive_cashflow, one_year_return, five_year_return,
                    max_drawdown, sharpe, trend_sma200, ai_prob, expected_return,
                    history_years, score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(isin) DO UPDATE SET
                    ticker = excluded.ticker,
                    sector = excluded.sector,
                    industry = excluded.industry,
                    status = excluded.status,
                    error = excluded.error,
                    price = excluded.price,
                    div_yield = excluded.div_yield,
                    payout_ratio = excluded.payout_ratio,
                    pe_ratio = excluded.pe_ratio,
                    debt_to_equity = excluded.debt_to_equity,
                    positive_cashflow = excluded.positive_cashflow,
                    one_year_return = excluded.one_year_return,
                    five_year_return = excluded.five_year_return,
                    max_drawdown = excluded.max_drawdown,
                    sharpe = excluded.sharpe,
                    trend_sma200 = excluded.trend_sma200,
                    ai_prob = excluded.ai_prob,
                    expected_return = excluded.expected_return,
                    history_years = excluded.history_years,
                    score = NULL
                """,
                (
                    isin,
                    ticker,
                    sector,
                    industry,
                    status,
                    error,
                    metrics.get("Price") if metrics else None,
                    metrics.get("Div_Yield_%") if metrics else None,
                    metrics.get("Payout_Ratio") if metrics else None,
                    metrics.get("PE_Ratio") if metrics else None,
                    metrics.get("Debt_To_Equity") if metrics else None,
                    pos_cf,
                    metrics.get("1Y_Return_%") if metrics else None,
                    metrics.get("5Y_Return_%") if metrics else None,
                    metrics.get("Max_DD_%") if metrics else None,
                    metrics.get("Sharpe") if metrics else None,
                    metrics.get("Trend_SMA200_%") if metrics else None,
                    metrics.get("AI_Prob") if metrics else None,
                    metrics.get("Expected_Return_%") if metrics else None,
                    metrics.get("History_Years") if metrics else None,
                ),
            )
            conn.commit()


def save_scores(ranked_table: pd.DataFrame, db_path: str = DATABASE_PATH) -> None:
    if ranked_table.empty:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE portfolio SET score = ? WHERE isin = ?",
            ranked_table[["Score", "ISIN"]].itertuples(index=False, name=None),
        )
        conn.commit()


def _ensure_cache_dirs() -> None:
    for path in [CACHE_DIR, PRICE_CACHE_DIR, FEATURE_CACHE_DIR, FUNDAMENTAL_CACHE_DIR, TICKER_MAP_CACHE_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _cache_is_fresh(path: Path, max_age_hours: float = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= max_age_hours * 3600


def _safe_name(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in value)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(None)
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out


def get_isin_list(url: str, limit: int | None = None) -> list[str]:
    _ensure_cache_dirs()

    if INSTRUMENT_CACHE_FILE.exists() and (OFFLINE or (not FORCE_REFRESH and _cache_is_fresh(INSTRUMENT_CACHE_FILE))):
        cached = pd.read_parquet(INSTRUMENT_CACHE_FILE)
        all_isins = sorted(cached['isin'].dropna().astype(str).unique().tolist())
        print(f"✓ Xetra-Instrumentenliste aus Cache geladen ({len(all_isins):,} ISINs).")
    else:
        if OFFLINE:
            raise RuntimeError(f"OFFLINE=1, aber kein Instrumenten-Cache vorhanden: {INSTRUMENT_CACHE_FILE}")

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            lines = [line.decode("utf-8", errors="ignore") for line in response.readlines()]

        reader = csv.reader(lines, delimiter=";")
        next(reader, None)
        next(reader, None)
        next(reader, None)

        isin_set = set()
        for row in reader:
            if len(row) > 3 and row[3].strip():
                isin = row[3].strip()
                if len(isin) == 12:
                    isin_set.add(isin)

        all_isins = sorted(isin_set)
        pd.DataFrame({'isin': all_isins}).to_parquet(INSTRUMENT_CACHE_FILE, index=False)
        print(f"✓ Xetra-Instrumentenliste heruntergeladen und gecacht ({len(all_isins):,} ISINs).")

    if limit is not None and limit < len(all_isins):
        rng = random.Random(42)
        return sorted(rng.sample(all_isins, limit))
    return all_isins


def _resolve_ticker_by_isin(isin: str, preferred_exchange: str | None = None) -> str:
    _ensure_cache_dirs()
    map_file = TICKER_MAP_CACHE_DIR / f"{_safe_name(isin)}.json"
    cached = _read_json(map_file) if map_file.exists() else None

    if cached and cached.get('ticker') and not FORCE_REFRESH:
        return str(cached['ticker'])

    if OFFLINE:
        raise RuntimeError(f"OFFLINE=1 und kein Ticker-Mapping im Cache für {isin}")

    search = yf.Search(isin, max_results=5)
    if not search.quotes:
        raise ValueError(f"Kein Ticker gefunden für ISIN: {isin}")

    selected_quote = None
    if preferred_exchange:
        for quote in search.quotes:
            if quote.get('exchange', '').upper() == preferred_exchange.upper():
                selected_quote = quote
                break
    if not selected_quote:
        selected_quote = search.quotes[0]

    ticker_symbol = selected_quote['symbol']
    _write_json_atomic(map_file, {
        'isin': isin,
        'ticker': ticker_symbol,
        'exchange': selected_quote.get('exchange'),
        'cached_at': pd.Timestamp.now("UTC").isoformat(),
    })
    return ticker_symbol


def _load_or_update_price_history(ticker_symbol: str, ticker: yf.Ticker) -> pd.DataFrame:
    _ensure_cache_dirs()
    cache_file = PRICE_CACHE_DIR / f"{_safe_name(ticker_symbol)}.parquet"

    cached = pd.DataFrame()
    if cache_file.exists():
        try:
            cached = _normalize_history(pd.read_parquet(cache_file))
        except Exception as exc:
            print(f"⚠ Preis-Cache für {ticker_symbol} konnte nicht gelesen werden: {exc}")

    if OFFLINE:
        if cached.empty:
            raise RuntimeError(f"OFFLINE=1, aber kein Preis-Cache für {ticker_symbol}")
        return cached

    if FORCE_REFRESH or cached.empty:
        fresh = _normalize_history(ticker.history(period='5y'))
        if fresh.empty:
            if not cached.empty:
                return cached
            raise ValueError(f"Keine Kursdaten für {ticker_symbol}")
        fresh.to_parquet(cache_file)
        return fresh

    # Incremental refresh. Re-download a small overlap because vendors can correct recent candles.
    last_date = cached.index.max()
    start_date = (last_date - pd.Timedelta(days=PRICE_OVERLAP_DAYS)).strftime('%Y-%m-%d')
    fresh = _normalize_history(ticker.history(start=start_date))

    if fresh.empty:
        return cached

    combined = pd.concat([cached, fresh])
    combined = _normalize_history(combined)
    # Keep roughly the same horizon as the original program.
    cutoff = pd.Timestamp.now("UTC").tz_localize(None) - pd.DateOffset(years=5, days=30)
    combined = combined[combined.index >= cutoff]
    combined.to_parquet(cache_file)
    return combined


def get_history_by_isin(
    isin: str,
    preferred_exchange: str | None = None,
    max_retries: int = 4,
    base_backoff: float = 2.0,
) -> tuple[str, yf.Ticker, pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            if not OFFLINE:
                time.sleep(random.uniform(0.1, 0.3))

            ticker_symbol = _resolve_ticker_by_isin(isin, preferred_exchange)
            ticker = yf.Ticker(ticker_symbol)
            df = _load_or_update_price_history(ticker_symbol, ticker)
            return ticker_symbol, ticker, df

        except Exception as e:
            err = str(e).lower()
            if not OFFLINE and ('too many requests' in err or 'rate limit' in err or '429' in err):
                wait_time = (base_backoff ** (attempt + 1)) + random.uniform(1.0, 3.0)
                print(f"⏳ Rate-Limit bei {isin}. Warte {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Maximale Versuche für ISIN {isin} überschritten.")


def _get_cached_fundamentals(ticker_symbol: str, ticker: yf.Ticker) -> dict:
    _ensure_cache_dirs()
    cache_file = FUNDAMENTAL_CACHE_DIR / f"{_safe_name(ticker_symbol)}.json"
    cached = _read_json(cache_file) if cache_file.exists() else None

    if cached and (OFFLINE or (not FORCE_REFRESH and _cache_is_fresh(cache_file))):
        return cached

    if OFFLINE:
        return cached or {}

    info = ticker.info or {}
    try:
        fast_info = ticker.fast_info
        raw_yield = fast_info.get('dividend_yield')
    except Exception:
        raw_yield = None

    payload = {
        'dividend_yield': raw_yield if raw_yield is not None else info.get('dividendYield'),
        'payout_ratio': info.get('payoutRatio'),
        'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
        'debt_to_equity': info.get('debtToEquity'),
        'operating_cashflow': info.get('operatingCashflow'),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'cached_at': pd.Timestamp.now("UTC").isoformat(),
    }
    _write_json_atomic(cache_file, payload)
    return payload


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet rollierende Momentum-, Trend- und Volatilitätsindikatoren."""
    data = pd.DataFrame(index=df.index)
    close = df["Close"]

    data["ret_21d"] = close.pct_change(21)
    data["ret_63d"] = close.pct_change(63)
    data["ret_126d"] = close.pct_change(126)
    data["ret_252d"] = close.pct_change(252)

    data["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
    data["sma_50"] = (close / close.rolling(50).mean()) - 1
    data["sma_200"] = (close / close.rolling(200).mean()) - 1

    rolling_max = close.rolling(252).max()
    data["drawdown"] = (close - rolling_max) / rolling_max

    return data


def build_ml_frame_cached(ticker_symbol: str, df_history: pd.DataFrame) -> pd.DataFrame:
    """Cache technical features and forward targets per ticker.

    If prices are unchanged, the cached frame is reused entirely. If new prices arrived,
    only the recent tail is recomputed; older rows are kept from cache.
    """
    _ensure_cache_dirs()
    cache_file = FEATURE_CACHE_DIR / f"{_safe_name(ticker_symbol)}.parquet"
    history = _normalize_history(df_history)
    if history.empty:
        return pd.DataFrame()

    last_price_date = history.index.max()
    cached = pd.DataFrame()
    if cache_file.exists() and not FORCE_REFRESH:
        try:
            cached = pd.read_parquet(cache_file)
            cached.index = pd.to_datetime(cached.index)
            cached_last = cached.attrs.get('source_last_date')
            # Parquet does not reliably preserve attrs across engines; fall back to a column.
            if '_source_last_date' in cached.columns and not cached.empty:
                cached_last = str(cached['_source_last_date'].iloc[-1])
            if cached_last and pd.Timestamp(cached_last) == pd.Timestamp(last_price_date):
                return cached.drop(columns=['_source_last_date'], errors='ignore')
        except Exception:
            cached = pd.DataFrame()

    def compute_frame(hist: pd.DataFrame) -> pd.DataFrame:
        features = extract_features(hist)
        close = hist['Close'].dropna()
        future_return = close.shift(-FORWARD_DAYS) / close - 1
        future_end_date = pd.Series(close.index, index=close.index).shift(-FORWARD_DAYS)
        out = features.copy()
        out['future_return'] = future_return.reindex(out.index)
        out['future_end_date'] = pd.to_datetime(future_end_date.reindex(out.index), errors='coerce')
        return out

    if cached.empty or len(history) <= FEATURE_RECOMPUTE_ROWS:
        result = compute_frame(history)
    else:
        cached = cached.drop(columns=['_source_last_date'], errors='ignore')
        recalc_pos = max(0, len(history) - FEATURE_RECOMPUTE_ROWS)
        recalc_start = history.index[recalc_pos]
        context_pos = max(0, recalc_pos - 260)
        context = history.iloc[context_pos:]
        recomputed = compute_frame(context)
        recomputed = recomputed[recomputed.index >= recalc_start]
        old = cached[cached.index < recalc_start]
        result = pd.concat([old, recomputed]).sort_index()
        result = result[~result.index.duplicated(keep='last')]

    save_df = result.copy()
    save_df['_source_last_date'] = str(last_price_date)
    save_df.to_parquet(cache_file)
    return result


def sanitize_dividend_yield(raw_yield: float | None) -> float:
    if raw_yield is None or np.isnan(raw_yield) or raw_yield <= 0:
        return 0.0
    if raw_yield < 0.25:
        normalized = raw_yield * 100
    elif raw_yield <= 20.0:
        normalized = raw_yield
    else:
        normalized = 0.0
    return round(normalized, 2)


def calculate_metrics_base(df: pd.DataFrame, fundamentals: dict) -> dict | None:
    """Berechnet fundamentale und technische Standardmetriken ohne lokale ML-Vorhersage."""
    if df is None or df.empty or "Close" not in df:
        return None

    close = df["Close"].dropna()
    if len(close) < 252:
        return None

    current_price = close.iloc[-1]
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    trend_pct = ((current_price / sma_200) - 1) * 100

    return_1y = ((current_price / close.iloc[-253]) - 1) * 100 if len(close) >= 253 else np.nan
    history_years = (close.index[-1] - close.index[0]).days / 365.25
    # Do not label a shorter history as a five-year return.
    return_5y = ((current_price / close.iloc[0]) - 1) * 100 if history_years >= 4.5 else np.nan

    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() * 100

    daily_returns = close.pct_change().dropna()
    rf_daily = 0.03 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    raw_yield = fundamentals.get("dividend_yield")
    div_yield = sanitize_dividend_yield(raw_yield)

    payout_ratio = fundamentals.get("payout_ratio")
    sustainable_div = True
    if payout_ratio is not None:
        if payout_ratio > 0.85 or payout_ratio < 0.0:
            sustainable_div = False
    elif div_yield > 6.0:
        sustainable_div = False

    pe_ratio = fundamentals.get("pe_ratio")
    debt_to_equity = fundamentals.get("debt_to_equity")
    # Yahoo commonly reports this field in percentage points (e.g. 75 == 0.75x).
    # Normalize only clearly percentage-like values and leave missing values untouched.
    if debt_to_equity is not None and debt_to_equity > 20:
        debt_to_equity = debt_to_equity / 100.0

    operating_cashflow = fundamentals.get("operating_cashflow")
    positive_cashflow = (operating_cashflow > 0) if operating_cashflow is not None else None

    sector = fundamentals.get("sector", "Unknown")
    industry = fundamentals.get("industry", "Unknown")

    return {
        "Price": round(current_price, 2),
        "Sector": sector,
        "Industry": industry,
        "Div_Yield_%": div_yield,
        "Payout_Ratio": round(payout_ratio, 2) if payout_ratio is not None else None,
        "Sustainable_Div": sustainable_div,
        "PE_Ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
        "Debt_To_Equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
        "Positive_CashFlow": positive_cashflow,
        "1Y_Return_%": round(return_1y, 2) if np.isfinite(return_1y) else np.nan,
        "5Y_Return_%": round(return_5y, 2) if np.isfinite(return_5y) else np.nan,
        "History_Years": round(history_years, 2),
        "Max_DD_%": round(max_drawdown, 2),
        "Sharpe": round(sharpe, 2),
        "Trend_SMA200_%": round(trend_pct, 2),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def get_progress_prefix() -> str:
    global COMPLETED_COUNT, TOTAL_ITEMS, START_TIME
    with PROGRESS_LOCK:
        COMPLETED_COUNT += 1
        done = COMPLETED_COUNT
        total = TOTAL_ITEMS
        elapsed = time.time() - START_TIME

    pct = (done / total) * 100 if total > 0 else 0.0
    left = total - done
    if done > 0 and left > 0:
        rate = elapsed / done
        eta_sec = rate * left
        eta_str = f"ETA: {format_duration(eta_sec)}"
    elif left == 0:
        eta_str = "Fertig"
    else:
        eta_str = "ETA: --"

    return f"[{done}/{total} | {pct:4.1f}% | {left} übrig | {eta_str}]"


def process_fetch_isin(isin: str) -> dict | None:
    """Worker Phase 1: cached prices/fundamentals + cached technical feature frame."""
    try:
        ticker, ticker_obj, df_history = get_history_by_isin(isin)
        fundamentals = _get_cached_fundamentals(ticker, ticker_obj)
        metrics = calculate_metrics_base(df_history, fundamentals)
        prog = get_progress_prefix()

        if metrics and len(df_history) >= 252:
            metrics["ISIN"] = isin
            metrics["Ticker"] = ticker

            ml_frame = build_ml_frame_cached(ticker, df_history)
            feature_cols = list(extract_features(df_history).columns)
            valid_mask = (
                ml_frame['future_return'].notna()
                & np.isfinite(ml_frame['future_return'])
                & ml_frame['future_end_date'].notna()
                & ml_frame[feature_cols].notna().all(axis=1)
            )

            X_history = ml_frame.loc[valid_mask, feature_cols].copy()
            y_reg_history = ml_frame.loc[valid_mask, 'future_return'].copy()
            y_clf_history = (y_reg_history > TARGET_RETURN_THRESHOLD).astype(int)
            target_end_dates = pd.to_datetime(ml_frame.loc[valid_mask, 'future_end_date'].values, utc=True).tz_convert(None)
            latest_feature = ml_frame[feature_cols].iloc[[-1]].copy()

            source_label = "Lokaler Cache" if OFFLINE else "Cache aktualisiert / Daten geladen"
            print(f"{prog} ✓ {isin} ({ticker}) — {source_label}")
            return {
                "isin": isin,
                "ticker": ticker,
                "metrics": metrics,
                "history": df_history,
                "X_train": X_history,
                "y_clf": y_clf_history,
                "y_reg": y_reg_history,
                "target_end_date": pd.Series(target_end_dates, index=X_history.index),
                "latest_feature": latest_feature,
            }
        else:
            save_instrument(isin=isin, ticker=ticker, status="skipped")
            print(f"{prog} ⚠ {isin} ({ticker}) — Übersprungen: Historie < 1 Jahr")
            return None

    except Exception as e:
        prog = get_progress_prefix()
        save_instrument(isin=isin, status="failed", error=str(e))
        print(f"{prog} ✗ {isin} — Fehler: {e}")
        return None


def _build_training_frame(pooled_records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in pooled_records:
        if r["X_train"].empty:
            continue
        part = r["X_train"].copy()
        part["sample_date"] = pd.to_datetime(part.index, utc=True).tz_convert(None)
        part["target_end_date"] = pd.to_datetime(r["target_end_date"].values, utc=True).tz_convert(None)
        part["isin"] = r["isin"]
        part["ticker"] = r["ticker"]
        part["target_clf"] = r["y_clf"].values
        part["target_reg"] = r["y_reg"].values
        rows.append(part.reset_index(drop=True))
    if not rows:
        raise ValueError("Keine gültigen Trainingsdaten für das globale Modell vorhanden.")
    return pd.concat(rows, ignore_index=True).sort_values("sample_date").reset_index(drop=True)


def _new_models():
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=RF_N_JOBS,
    )

    reg = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=RF_N_JOBS,
    )

    return clf, reg


def _fit_forest_with_progress(model, X, y, label: str):
    """Fit a RandomForest in warm-start batches and show tree-level progress.

    Using the same training data and fixed random_state, incremental warm-start
    fitting produces the same forest as a one-shot fit with the same final
    n_estimators. The batching is only used to expose progress to the user.
    """
    total_trees = int(model.n_estimators)
    if total_trees <= 0:
        raise ValueError("RandomForest n_estimators must be positive.")

    if not RF_PROGRESS:
        model.fit(X, y)
        return model

    batch_size = min(RF_PROGRESS_BATCH, total_trees)
    original_warm_start = bool(getattr(model, "warm_start", False))
    start = time.time()
    completed = 0

    # Keep the same dataset for every batch. For the classifier this makes the
    # sklearn warm-start/class_weight warning inapplicable, so suppress only
    # that warning locally.
    model.set_params(warm_start=True)

    while completed < total_trees:
        completed = min(completed + batch_size, total_trees)
        model.set_params(n_estimators=completed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r'class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start.*',
                category=UserWarning,
            )
            model.fit(X, y)

        elapsed = max(time.time() - start, 1e-9)
        rate = completed / elapsed
        remaining = total_trees - completed
        eta = remaining / rate if rate > 0 else 0.0
        fraction = completed / total_trees
        filled = min(RF_PROGRESS_WIDTH, int(round(RF_PROGRESS_WIDTH * fraction)))
        bar = "█" * filled + "░" * (RF_PROGRESS_WIDTH - filled)
        print(
            f"\r  {label:<28} [{bar}] {completed:>3}/{total_trees} "
            f"({fraction*100:5.1f}%) | {format_duration(elapsed)} | ETA {format_duration(eta)}",
            end="",
            flush=True,
        )

    print()
    model.set_params(n_estimators=total_trees, warm_start=original_warm_start)
    return model


def walk_forward_validate(dataset: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Expanding-window validation with target-date purging (embargo by construction)."""
    unique_dates = np.array(sorted(pd.to_datetime(dataset["sample_date"].unique())))
    results = []
    if len(unique_dates) < MIN_TRAIN_DATES + WALK_FORWARD_TEST_DAYS:
        print("⚠ Zu wenige unterschiedliche Handelstage für Walk-Forward-Validierung.")
        return pd.DataFrame()

    possible_starts = []
    last_start = len(unique_dates) - WALK_FORWARD_TEST_DAYS
    for fold_back in range(WALK_FORWARD_FOLDS - 1, -1, -1):
        start_idx = last_start - fold_back * WALK_FORWARD_TEST_DAYS
        if start_idx >= MIN_TRAIN_DATES:
            possible_starts.append(start_idx)

    print("\nWalk-Forward-Validierung (Training wird anhand target_end_date vor Testbeginn bereinigt):")
    for fold_no, start_idx in enumerate(possible_starts, 1):
        end_idx = min(start_idx + WALK_FORWARD_TEST_DAYS, len(unique_dates))
        test_start = pd.Timestamp(unique_dates[start_idx])
        test_end = pd.Timestamp(unique_dates[end_idx - 1])

        train = dataset[dataset["target_end_date"] < test_start]
        test = dataset[(dataset["sample_date"] >= test_start) & (dataset["sample_date"] <= test_end)]
        if len(train) < 1000 or len(test) < 100 or train["target_clf"].nunique() < 2 or test["target_clf"].nunique() < 2:
            continue

        clf, reg = _new_models()
        fold_total = len(possible_starts)
        _fit_forest_with_progress(
            clf, train[feature_cols], train["target_clf"],
            f"Fold {fold_no}/{fold_total} classifier",
        )
        _fit_forest_with_progress(
            reg, train[feature_cols], train["target_reg"],
            f"Fold {fold_no}/{fold_total} regressor",
        )
        prob = clf.predict_proba(test[feature_cols])[:, 1]
        pred_ret = reg.predict(test[feature_cols])

        auc = roc_auc_score(test["target_clf"], prob)
        brier = brier_score_loss(test["target_clf"], prob)
        mae = mean_absolute_error(test["target_reg"], pred_ret)
        spearman = pd.Series(pred_ret).corr(pd.Series(test["target_reg"].to_numpy()), method="spearman")

        # Ranking usefulness: compare the top predicted quintile with the whole test universe.
        cutoff = np.nanquantile(pred_ret, 0.80)
        top_mask = pred_ret >= cutoff
        top_mean = float(test.loc[top_mask, "target_reg"].mean()) if top_mask.any() else np.nan
        all_mean = float(test["target_reg"].mean())

        results.append({
            "fold": fold_no,
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "roc_auc": auc,
            "brier": brier,
            "reg_mae": mae,
            "spearman": spearman,
            "top20_actual_return": top_mean,
            "all_actual_return": all_mean,
            "top20_excess": top_mean - all_mean,
        })
        print(
            f"  Fold {fold_no}: {test_start.date()}–{test_end.date()} | "
            f"AUC={auc:.3f} Brier={brier:.3f} MAE={mae:.3f} "
            f"Spearman={spearman:.3f} Top20 excess={top_mean-all_mean:+.3f}"
        )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df.to_csv("walk_forward_metrics.csv", index=False)
        print("✓ Walk-Forward-Metriken in 'walk_forward_metrics.csv' gespeichert.")
    return result_df


def train_global_ai_models(pooled_records: list[dict]) -> tuple[CalibratedRFClassifier, RandomForestRegressor, pd.DataFrame]:
    """Validiert chronologisch und trainiert anschließend Produktionsmodelle mit zeitgerechter Kalibrierung."""
    print(f"\nSammle Trainingsdaten von {len(pooled_records)} Wertpapieren...")
    dataset = _build_training_frame(pooled_records)
    feature_cols = [c for c in extract_features(pooled_records[0]["history"]).columns if c in dataset.columns]

    with sqlite3.connect(DATABASE_PATH) as conn:
        db_dataset = dataset.copy()
        db_dataset["sample_date"] = db_dataset["sample_date"].astype(str)
        db_dataset["target_end_date"] = db_dataset["target_end_date"].astype(str)
        db_dataset.to_sql("training_data", conn, if_exists="replace", index=False)
    print(f"✓ {len(dataset):,} Trainingszeilen in Tabelle 'training_data' in '{DATABASE_PATH}' gespeichert.")

    validation = walk_forward_validate(dataset, feature_cols)

    # Time-aware calibration: reserve the most recent ~126 feature dates for calibration,
    # while purging any fitting labels that extend into that calibration period.
    unique_dates = np.array(sorted(pd.to_datetime(dataset["sample_date"].unique())))
    calibrator = None
    base_clf, reg = _new_models()

    if len(unique_dates) > MIN_TRAIN_DATES + WALK_FORWARD_TEST_DAYS:
        calib_start = pd.Timestamp(unique_dates[-WALK_FORWARD_TEST_DAYS])
        fit = dataset[dataset["target_end_date"] < calib_start]
        calib = dataset[dataset["sample_date"] >= calib_start]
        if len(fit) >= 1000 and len(calib) >= 100 and fit["target_clf"].nunique() == 2 and calib["target_clf"].nunique() == 2:
            _fit_forest_with_progress(
                base_clf, fit[feature_cols], fit["target_clf"], "Production classifier"
            )
            raw_prob = base_clf.predict_proba(calib[feature_cols])[:, 1]

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_prob, calib["target_clf"].to_numpy())
            print(f"✓ Wahrscheinlichkeitskalibrierung auf {len(calib):,} jüngsten, zeitlich getrennten Zeilen angepasst.")
        else:
            _fit_forest_with_progress(
                base_clf, dataset[feature_cols], dataset["target_clf"], "Production classifier"
            )
    else:
        _fit_forest_with_progress(
            base_clf, dataset[feature_cols], dataset["target_clf"], "Production classifier"
        )

    # Regression is trained on all labelled history after validation; current inference is beyond all labels.
    _fit_forest_with_progress(
        reg, dataset[feature_cols], dataset["target_reg"], "Production regressor"
    )
    print("✓ Globales Produktionsmodell trainiert.\n")
    return CalibratedRFClassifier(base_clf, calibrator), reg, validation


def plot_single_prediction(
    isin: str,
    ticker: str,
    df_history: pd.DataFrame,
    forecast_series: pd.Series | None,
    ai_prob: float,
    expected_ret: float,
    output_dir: str = PLOTS_DIR,
) -> None:
    """Plot history plus a dashed endpoint scenario, not a claimed daily price forecast."""
    if forecast_series is None or df_history.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    history_slice = df_history["Close"].iloc[-380:]
    ax.plot(history_slice.index, history_slice.values, label="Historischer Kurs", lw=2)

    ax.plot(
        [history_slice.index[-1], forecast_series.index[-1]],
        [history_slice.values[-1], forecast_series.values[-1]],
        linestyle="--",
        lw=2,
        label=f"6M-Endpunktszenario ({expected_ret*100:+.1f}%)",
    )
    ax.scatter([forecast_series.index[-1]], [forecast_series.values[-1]], s=45, zorder=4)
    ax.axvline(x=history_slice.index[-1], linestyle=":", label="Prognosebeginn")
    ax.set_title(
        f"{ticker} ({isin}) — 6M-Endpunkt | P(Return > {TARGET_RETURN_THRESHOLD*100:.0f}%)={ai_prob*100:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kurs")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    filename = os.path.join(output_dir, f"{ticker}_{isin}.png")
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def rank_portfolio(records: list[dict], max_per_sector: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(records)
    if df.empty:
        return df, df

    # Missing historical/fundamental values receive a neutral percentile rather than
    # being silently treated as excellent or terrible.
    def pct_rank(series: pd.Series, neutral: float = 0.5) -> pd.Series:
        ranked = series.rank(pct=True)
        return ranked.fillna(neutral)

    df["rank_ret_5y"] = pct_rank(df["5Y_Return_%"])
    df["rank_sharpe"] = pct_rank(df["Sharpe"])
    df["rank_risk"] = pct_rank(df["Max_DD_%"])  # less-negative drawdown ranks higher

    clean_yield = df.apply(lambda r: r["Div_Yield_%"] if r["Sustainable_Div"] else 0.0, axis=1)
    df["rank_dividend"] = pct_rank(clean_yield)

    def pe_score(pe):
        if pe is None or pd.isna(pe) or pe <= 0:
            return 0.2
        if 5 <= pe <= 20:
            return 1.0
        if 20 < pe <= 35:
            return 0.6
        return 0.3

    df["val_score"] = df["PE_Ratio"].apply(pe_score)

    # Reduce double-counting of momentum/trend. The traditional factor block focuses
    # on long-run return, risk, valuation and dividend quality; ML already consumes momentum.
    factor_score = (
        df["rank_ret_5y"] * 0.25
        + df["rank_sharpe"] * 0.25
        + df["val_score"] * 0.20
        + df["rank_dividend"] * 0.15
        + df["rank_risk"] * 0.15
    ) * 100

    # ML block combines calibrated probability, expected return rank and downside-aware risk.
    df["rank_expected_return"] = pct_rank(df["Expected_Return_%"])
    df["rank_ml_risk"] = pct_rank(df["Max_DD_%"])
    ml_score = (
        (df["AI_Prob"].clip(0, 1) * 0.50)
        + (df["rank_expected_return"] * 0.35)
        + (df["rank_ml_risk"] * 0.15)
    ) * 100

    df["Score"] = factor_score * 0.60 + ml_score * 0.40

    df.loc[df["Debt_To_Equity"].fillna(0) > 2.0, "Score"] *= 0.85
    df.loc[df["Positive_CashFlow"] == False, "Score"] *= 0.85
    df["Score"] = df["Score"].round(1)

    display_cols = [
        "ISIN", "Ticker", "Sector", "Score", "AI_Prob", "Expected_Return_%", "Price",
        "PE_Ratio", "Div_Yield_%", "Payout_Ratio", "5Y_Return_%", "History_Years", "Sharpe", "Max_DD_%"
    ]
    full_ranking = df[display_cols].sort_values(by="Score", ascending=False).reset_index(drop=True)

    diversified_rows = []
    sector_counts: dict[str, int] = {}
    for _, row in full_ranking.iterrows():
        sec = row["Sector"]
        if sector_counts.get(sec, 0) < max_per_sector:
            diversified_rows.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    diversified_portfolio = pd.DataFrame(diversified_rows).reset_index(drop=True)
    return full_ranking, diversified_portfolio


def plot_top_predictions_grid(top_picks: list[dict], output_file: str = "top_predictions_grid.png"):
    if not top_picks:
        return

    n_plots = min(6, len(top_picks))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for idx in range(n_plots):
        item = top_picks[idx]
        ax = axes[idx]
        hist = item["history"]["Close"].iloc[-250:]
        f_series = item["forecast"]
        ticker = item["metrics"]["Ticker"]
        isin = item["metrics"]["ISIN"]
        prob = item["metrics"]["AI_Prob"]
        expected = item["metrics"]["Expected_Return_%"]

        ax.plot(hist.index, hist.values, label="Historie")
        if f_series is not None:
            ax.plot(
                [hist.index[-1], f_series.index[-1]],
                [hist.values[-1], f_series.values[-1]],
                linestyle="--",
                label="6M-Endpunkt",
            )
            ax.scatter([f_series.index[-1]], [f_series.values[-1]], s=30)
            ax.axvline(x=hist.index[-1], linestyle=":")

        ax.set_title(
            f"#{idx+1}: {ticker} | Score {item['metrics']['Score']} | "
            f"P>{TARGET_RETURN_THRESHOLD*100:.0f}% {prob*100:.0f}% | E[R] {expected:+.1f}%\n"
            f"ISIN: {isin}",
            fontsize=9,
            fontweight="bold",
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Top-Kandidaten: 6-Monats-Endpunktszenarien", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"✓ Dashboard-Grid gespeichert unter '{output_file}'")


# -----------------------------------------------------------------------------
# Workflow helpers
# -----------------------------------------------------------------------------

def get_cached_isin_list(limit: int | None = None) -> list[str]:
    """Return only ISINs that already have a local ticker mapping and price cache.

    This function never performs network access. It is the preferred universe source for
    analyze/train/rank workflows because those workflows must be reproducible and offline.
    """
    _ensure_cache_dirs()
    candidates: list[str] = []
    for map_file in sorted(TICKER_MAP_CACHE_DIR.glob("*.json")):
        payload = _read_json(map_file)
        if not payload:
            continue
        isin = str(payload.get("isin") or "").strip()
        ticker = str(payload.get("ticker") or "").strip()
        if not isin or not ticker:
            continue
        price_file = PRICE_CACHE_DIR / f"{_safe_name(ticker)}.parquet"
        if price_file.exists():
            candidates.append(isin)

    candidates = sorted(set(candidates))
    if limit is not None and limit > 0 and limit < len(candidates):
        rng = random.Random(42)
        candidates = sorted(rng.sample(candidates, limit))
    return candidates


def collect_records(isin_list: list[str], max_workers: int = 2, phase_name: str = "Daten") -> list[dict]:
    """Load/process a universe using the currently selected online/offline mode."""
    global TOTAL_ITEMS, START_TIME, COMPLETED_COUNT
    TOTAL_ITEMS = len(isin_list)
    COMPLETED_COUNT = 0
    START_TIME = time.time()

    if not isin_list:
        return []

    print(f"{phase_name}: {TOTAL_ITEMS} ISINs werden verarbeitet.\n")
    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_fetch_isin, isin_list)
        for result in results:
            if result is not None:
                records.append(result)

    elapsed = time.time() - START_TIME
    print(f"\n{phase_name} abgeschlossen: {len(records)} gültige Instrumente in {format_duration(elapsed)}.")
    return records


def prepare_training_dataset(
    pooled_records: list[dict],
    save_training_data: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the pooled historical ML dataset from already loaded/cached records."""
    if not pooled_records:
        raise ValueError("Keine gültigen gecachten Instrumente vorhanden.")

    dataset = _build_training_frame(pooled_records)
    feature_cols = [c for c in extract_features(pooled_records[0]["history"]).columns if c in dataset.columns]

    if save_training_data:
        initialize_database()
        with sqlite3.connect(DATABASE_PATH) as conn:
            db_dataset = dataset.copy()
            db_dataset["sample_date"] = db_dataset["sample_date"].astype(str)
            db_dataset["target_end_date"] = db_dataset["target_end_date"].astype(str)
            db_dataset.to_sql("training_data", conn, if_exists="replace", index=False)
        print(f"✓ {len(dataset):,} Trainingszeilen in '{DATABASE_PATH}' gespeichert.")

    return dataset, feature_cols


def analyze_records(pooled_records: list[dict]) -> pd.DataFrame:
    """Run only the chronological out-of-sample analysis; no production training."""
    dataset, feature_cols = prepare_training_dataset(pooled_records, save_training_data=True)
    validation = walk_forward_validate(dataset, feature_cols)
    if not validation.empty:
        numeric_cols = ["roc_auc", "brier", "reg_mae", "spearman", "top20_excess"]
        print("\nDurchschnittliche Out-of-Sample-Metriken:")
        print(validation[numeric_cols].mean(numeric_only=True).round(3).to_string())
    return validation


def train_production_models_from_dataset(
    dataset: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[CalibratedRFClassifier, RandomForestRegressor]:
    """Train production models without running walk-forward validation again."""
    unique_dates = np.array(sorted(pd.to_datetime(dataset["sample_date"].unique())))
    calibrator = None
    base_clf, reg = _new_models()

    if len(unique_dates) > MIN_TRAIN_DATES + WALK_FORWARD_TEST_DAYS:
        calib_start = pd.Timestamp(unique_dates[-WALK_FORWARD_TEST_DAYS])
        fit = dataset[dataset["target_end_date"] < calib_start]
        calib = dataset[dataset["sample_date"] >= calib_start]
        if (
            len(fit) >= 1000
            and len(calib) >= 100
            and fit["target_clf"].nunique() == 2
            and calib["target_clf"].nunique() == 2
        ):
            _fit_forest_with_progress(
                base_clf, fit[feature_cols], fit["target_clf"], "Production classifier"
            )
            raw_prob = base_clf.predict_proba(calib[feature_cols])[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_prob, calib["target_clf"].to_numpy())
            print(
                f"✓ Wahrscheinlichkeitskalibrierung auf {len(calib):,} "
                "jüngsten, zeitlich getrennten Zeilen angepasst."
            )
        else:
            _fit_forest_with_progress(
                base_clf, dataset[feature_cols], dataset["target_clf"], "Production classifier"
            )
    else:
        _fit_forest_with_progress(
            base_clf, dataset[feature_cols], dataset["target_clf"], "Production classifier"
        )

    _fit_forest_with_progress(
        reg, dataset[feature_cols], dataset["target_reg"], "Production regressor"
    )

    print("✓ Produktionsmodelle trainiert.")
    return CalibratedRFClassifier(base_clf, calibrator), reg


def save_production_models(
    classifier: CalibratedRFClassifier,
    regressor: RandomForestRegressor,
    feature_cols: list[str],
    path: Path = MODEL_BUNDLE_FILE,
) -> None:
    """Persist the trained model bundle for later ranking-only runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_cols": feature_cols,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "rf_n_jobs": RF_N_JOBS,
        "forward_days": FORWARD_DAYS,
        "target_return_threshold": TARGET_RETURN_THRESHOLD,
    }
    dump(payload, path)
    print(f"✓ Produktionsmodell gespeichert: '{path}'")


def load_production_models(
    path: Path = MODEL_BUNDLE_FILE,
) -> tuple[CalibratedRFClassifier, RandomForestRegressor, list[str], dict]:
    """Load the saved production model bundle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Kein gespeichertes Produktionsmodell gefunden: {path}. "
            "Bitte zuerst 'python train.py' ausführen."
        )
    payload = load(path)
    return payload["classifier"], payload["regressor"], list(payload["feature_cols"]), payload


def train_and_save_records(pooled_records: list[dict]) -> tuple[CalibratedRFClassifier, RandomForestRegressor, list[str]]:
    """Train production models from cached records and persist them."""
    dataset, feature_cols = prepare_training_dataset(pooled_records, save_training_data=True)
    classifier, regressor = train_production_models_from_dataset(dataset, feature_cols)
    save_production_models(classifier, regressor, feature_cols)
    return classifier, regressor, feature_cols


def generate_ranking(
    fetched_records: list[dict],
    classifier: CalibratedRFClassifier,
    regressor: RandomForestRegressor,
    feature_cols: list[str],
    create_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate today's ranking from cached records and an already-trained model."""
    initialize_database()
    collected_metrics: list[dict] = []
    bundle_store: dict[str, dict] = {}

    print("Erzeuge Ranking mit gespeichertem Produktionsmodell...")
    for item in fetched_records:
        isin = item["isin"]
        ticker = item["ticker"]
        metrics = item["metrics"]
        df_history = item["history"]
        latest_feat = item["latest_feature"].reindex(columns=feature_cols)

        if latest_feat.empty or latest_feat.isna().any(axis=1).iloc[0]:
            prob = 0.5
            pred_ret = 0.0
            forecast_series = None
        else:
            prob = float(classifier.predict_proba(latest_feat)[0][1])
            pred_ret = float(regressor.predict(latest_feat)[0])

            current_price = metrics["Price"]
            last_date = df_history.index[-1]
            endpoint_date = pd.bdate_range(start=last_date, periods=FORWARD_DAYS + 1)[-1]
            target_price = current_price * (1.0 + pred_ret)
            forecast_series = pd.Series([target_price], index=[endpoint_date])

        metrics["AI_Prob"] = round(prob, 3)
        metrics["Expected_Return_%"] = round(pred_ret * 100, 2)
        collected_metrics.append(metrics)

        save_instrument(
            isin=isin,
            ticker=ticker,
            sector=metrics["Sector"],
            industry=metrics["Industry"],
            metrics=metrics,
            status="processed",
        )

        if create_plots:
            plot_single_prediction(
                isin=isin,
                ticker=ticker,
                df_history=df_history,
                forecast_series=forecast_series,
                ai_prob=prob,
                expected_ret=pred_ret,
            )

        bundle_store[isin] = {
            "metrics": metrics,
            "history": df_history,
            "forecast": forecast_series,
            "ret": pred_ret,
        }

    full_ranking, diversified = rank_portfolio(collected_metrics, max_per_sector=2)
    save_scores(full_ranking)

    print("\n" + "=" * 115)
    print("FINALE INVESTMENT-RANGLISTE")
    print("=" * 115)
    print(full_ranking.to_string())

    if create_plots:
        top_candidates = []
        for _, row in full_ranking.head(6).iterrows():
            isin_val = row["ISIN"]
            if isin_val in bundle_store:
                bundle_store[isin_val]["metrics"]["Score"] = row["Score"]
                top_candidates.append(bundle_store[isin_val])
        plot_top_predictions_grid(top_candidates)
        print(f"\nEinzelne Diagramme gespeichert in '{PLOTS_DIR}/'.")

    print(f"Datenbank und Scores aktualisiert in '{DATABASE_PATH}'.")
    return full_ranking, diversified


def print_cache_summary() -> None:
    cached_isins = get_cached_isin_list()
    price_files = list(PRICE_CACHE_DIR.glob("*.parquet"))
    feature_files = list(FEATURE_CACHE_DIR.glob("*.parquet"))
    fundamental_files = list(FUNDAMENTAL_CACHE_DIR.glob("*.json"))
    print(f"Cache-Verzeichnis: {CACHE_DIR.resolve()}")
    print(
        f"Cache: {len(cached_isins):,} verwendbare ISINs | "
        f"{len(price_files):,} Preise | {len(feature_files):,} Features | "
        f"{len(fundamental_files):,} Fundamentals"
    )
    print(f"Random Forest workers: RF_N_JOBS={RF_N_JOBS}")
