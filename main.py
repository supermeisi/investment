import concurrent.futures
import csv
import itertools
import os
import random
import sqlite3
import threading
import time
import urllib.request
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

DATABASE_PATH = "ranking.db"
PLOTS_DIR = "plots"

DB_LOCK = threading.Lock()
PROGRESS_LOCK = threading.Lock()

COMPLETED_COUNT = 0
TOTAL_ITEMS = 0
START_TIME = 0.0

FORWARD_DAYS = 126
TARGET_RETURN_THRESHOLD = 0.10
MIN_TRAIN_DATES = 252
WALK_FORWARD_TEST_DAYS = 126
WALK_FORWARD_FOLDS = 3


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


def get_isin_list(url: str, limit: int | None = None) -> list[str]:
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
    if limit is not None and limit < len(all_isins):
        # Avoid the selection bias of taking only the first rows of the exchange file.
        rng = random.Random(42)
        return sorted(rng.sample(all_isins, limit))
    return all_isins


def get_history_by_isin(
    isin: str,
    preferred_exchange: str | None = None,
    max_retries: int = 4,
    base_backoff: float = 2.0,
) -> tuple[str, yf.Ticker, pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.2, 0.5))

            search = yf.Search(isin, max_results=5)
            if not search.quotes:
                raise ValueError(f"Kein Ticker gefunden für ISIN: {isin}")

            selected_quote = None
            if preferred_exchange:
                for quote in search.quotes:
                    if quote.get("exchange", "").upper() == preferred_exchange.upper():
                        selected_quote = quote
                        break

            if not selected_quote:
                selected_quote = search.quotes[0]

            ticker_symbol = selected_quote["symbol"]
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="5y")

            return ticker_symbol, ticker, df

        except Exception as e:
            err = str(e).lower()
            if "too many requests" in err or "rate limit" in err or "429" in err:
                wait_time = (base_backoff ** (attempt + 1)) + random.uniform(1.0, 3.0)
                print(f"⏳ Rate-Limit bei {isin}. Warte {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError(f"Maximale Versuche für ISIN {isin} überschritten.")


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


def calculate_metrics_base(df: pd.DataFrame, ticker: yf.Ticker) -> dict | None:
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

    info = ticker.info or {}
    raw_yield = ticker.fast_info.get("dividend_yield") or info.get("dividendYield")
    div_yield = sanitize_dividend_yield(raw_yield)

    payout_ratio = info.get("payoutRatio")
    sustainable_div = True
    if payout_ratio is not None:
        if payout_ratio > 0.85 or payout_ratio < 0.0:
            sustainable_div = False
    elif div_yield > 6.0:
        sustainable_div = False

    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    debt_to_equity = info.get("debtToEquity")
    # Yahoo commonly reports this field in percentage points (e.g. 75 == 0.75x).
    # Normalize only clearly percentage-like values and leave missing values untouched.
    if debt_to_equity is not None and debt_to_equity > 20:
        debt_to_equity = debt_to_equity / 100.0

    operating_cashflow = info.get("operatingCashflow")
    positive_cashflow = (operating_cashflow > 0) if operating_cashflow is not None else None

    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")

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
    """Worker Phase 1: Daten, Features und zeitlich korrekt markierte Trainingspaare."""
    try:
        ticker, ticker_obj, df_history = get_history_by_isin(isin)
        metrics = calculate_metrics_base(df_history, ticker_obj)
        prog = get_progress_prefix()

        if metrics and len(df_history) >= 252:
            metrics["ISIN"] = isin
            metrics["Ticker"] = ticker

            features = extract_features(df_history)
            close = df_history["Close"].dropna()

            future_return = close.shift(-FORWARD_DAYS) / close - 1
            future_end_date = pd.Series(close.index, index=close.index).shift(-FORWARD_DAYS)
            valid_mask = (
                future_return.notna()
                & np.isfinite(future_return)
                & future_end_date.notna()
                & features.notna().all(axis=1)
            )

            X_history = features.loc[valid_mask].copy()
            y_reg_history = future_return.loc[valid_mask].copy()
            y_clf_history = (y_reg_history > TARGET_RETURN_THRESHOLD).astype(int)
            target_end_dates = pd.to_datetime(future_end_date.loc[valid_mask].values, utc=True).tz_convert(None)

            latest_feature = features.iloc[[-1]].copy()

            print(f"{prog} ✓ {isin} ({ticker}) — Daten und Features geladen")
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


def _new_models() -> tuple[RandomForestClassifier, RandomForestRegressor]:
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    reg = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1,
    )
    return clf, reg


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
        clf.fit(train[feature_cols], train["target_clf"])
        reg.fit(train[feature_cols], train["target_reg"])
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
            base_clf.fit(fit[feature_cols], fit["target_clf"])
            raw_prob = base_clf.predict_proba(calib[feature_cols])[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_prob, calib["target_clf"].to_numpy())
            print(f"✓ Wahrscheinlichkeitskalibrierung auf {len(calib):,} jüngsten, zeitlich getrennten Zeilen angepasst.")
        else:
            base_clf.fit(dataset[feature_cols], dataset["target_clf"])
    else:
        base_clf.fit(dataset[feature_cols], dataset["target_clf"])

    # Regression is trained on all labelled history after validation; current inference is beyond all labels.
    reg.fit(dataset[feature_cols], dataset["target_reg"])
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
            f"P>{TARGET_RETURN_THRESHOLD*100:.0f}% {prob*100:.0f}% | E[R] {expected:+.1f}%",
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


def main():
    global TOTAL_ITEMS, START_TIME, COMPLETED_COUNT

    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"

    # Set MAX_INSTRUMENTS=0 for the complete Xetra universe.
    # A fixed random sample is used when a limit is set, rather than the first rows in the file.
    env_limit = int(os.getenv("MAX_INSTRUMENTS", "100"))
    limit_count = None if env_limit <= 0 else env_limit
    isin_list = get_isin_list(url=url, limit=limit_count)

    TOTAL_ITEMS = len(isin_list)
    COMPLETED_COUNT = 0
    START_TIME = time.time()

    print(f"Lade Daten für {TOTAL_ITEMS} ISINs. Starte Phase 1 (Paralleler Datenabruf)...\n")
    initialize_database()

    fetched_records = []
    max_workers = 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_fetch_isin, isin_list)
        for res in results:
            if res is not None:
                fetched_records.append(res)

    elapsed_fetch = round(time.time() - START_TIME, 1)
    print(f"\nPhase 1 abgeschlossen: {len(fetched_records)} gültige Instrumente in {format_duration(elapsed_fetch)} geladen.")

    if len(fetched_records) < 5:
        print("Zu wenige gültige Daten für ein verlässliches globales Modell.")
        return

    # Phase 2: Globales Training über alle ISIN-Daten
    clf_global, reg_global, validation_metrics = train_global_ai_models(fetched_records)
    if not validation_metrics.empty:
        numeric_cols = ["roc_auc", "brier", "reg_mae", "spearman", "top20_excess"]
        print("\nDurchschnittliche Out-of-Sample-Metriken:")
        print(validation_metrics[numeric_cols].mean(numeric_only=True).round(3).to_string())

    print("Phase 3: Führe Inferenz für jedes Wertpapier mit dem globalen Modell aus...")
    collected_metrics = []
    bundle_store = {}

    for item in fetched_records:
        isin = item["isin"]
        ticker = item["ticker"]
        metrics = item["metrics"]
        df_history = item["history"]
        latest_feat = item["latest_feature"]

        if latest_feat.isna().any(axis=1).values[0]:
            prob = 0.5
            pred_ret = 0.0
            forecast_series = None
        else:
            prob = float(clf_global.predict_proba(latest_feat)[0][1])
            pred_ret = float(reg_global.predict(latest_feat)[0])

            current_price = metrics["Price"]
            last_date = df_history.index[-1]
            endpoint_date = pd.bdate_range(start=last_date, periods=FORWARD_DAYS + 1)[-1]
            target_price = current_price * (1.0 + pred_ret)
            # One endpoint only: the model predicts a 126-day return, not the daily path.
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

    # Ranking & Diversifikation
    full_ranking, diversified = rank_portfolio(collected_metrics, max_per_sector=2)

    print("\n" + "=" * 115)
    print("FINALE INVESTMENT-RANGLISTE (OUT-OF-SAMPLE VALIDIERTER ML-ANSATZ)")
    print("=" * 115)
    print(full_ranking.to_string())

    save_scores(full_ranking)

    top_candidates = []
    for _, row in full_ranking.head(6).iterrows():
        isin_val = row["ISIN"]
        if isin_val in bundle_store:
            bundle_store[isin_val]["metrics"]["Score"] = row["Score"]
            top_candidates.append(bundle_store[isin_val])

    plot_top_predictions_grid(top_candidates)
    print(f"\nEinzelne Diagramme gespeichert in '{PLOTS_DIR}/'.")
    print(f"Datenbank und Scores aktualisiert in '{DATABASE_PATH}'.")


if __name__ == "__main__":
    main()
