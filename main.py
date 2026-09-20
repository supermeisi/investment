import concurrent.futures
import csv
import itertools
import os
import random
import sqlite3
import threading
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import yfinance as yf

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATABASE_PATH = "ranking.db"
PLOTS_DIR = "plots"
DB_LOCK = threading.Lock()


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
                    max_drawdown, sharpe, trend_sma200, ai_prob, score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
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
    for row in itertools.islice(reader, limit):
        if len(row) > 3 and row[3].strip():
            isin = row[3].strip()
            if len(isin) == 12:
                isin_set.add(isin)

    return sorted(list(isin_set))


def get_history_by_isin(
    isin: str,
    preferred_exchange: str | None = None,
    max_retries: int = 4,
    base_backoff: float = 2.0,
) -> tuple[str, yf.Ticker, pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.3, 0.7))

            search = yf.Search(isin, max_results=5)
            if not search.quotes:
                raise ValueError(f"No ticker found on Yahoo Finance for ISIN: {isin}")

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
                print(f"⏳ Rate-limited on {isin}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError(f"Exceeded max retries on ISIN {isin}")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
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


def train_ai_predictor(df: pd.DataFrame) -> tuple[float, pd.Series | None, float]:
    """Trains both a probability classifier and a forward-trajectory regressor.

    Returns:
        (ai_probability, forward_predicted_prices, expected_return_fraction)
    """
    if len(df) < 500:
        return 0.5, None, 0.0

    features = extract_features(df)
    close = df["Close"].dropna()
    current_price = close.iloc[-1]

    # Target: 6-month (126 trading days) forward return
    future_return = close.shift(-126) / close - 1
    
    # Explicitly filter out NaNs and infs in target and features
    valid_mask = (
        future_return.notna() 
        & np.isfinite(future_return) 
        & features.notna().all(axis=1)
    )

    X_train = features[valid_mask]
    y_reg = future_return[valid_mask]
    y_clf = (y_reg > 0.10).astype(int)

    # Need adequate training samples and at least 2 classes (both 0 and 1) for the classifier
    if len(X_train) < 200 or len(y_clf.unique()) < 2:
        return 0.5, None, 0.0

    # 1. Classification for AI_Prob
    clf = RandomForestClassifier(
        n_estimators=60,
        max_depth=4,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_train, y_clf)

    # 2. Regression for path trajectory projection
    reg = RandomForestRegressor(
        n_estimators=60,
        max_depth=4,
        min_samples_leaf=15,
        random_state=42,
        n_jobs=1,
    )
    reg.fit(X_train, y_reg)

    latest_features = features.iloc[[-1]]
    if latest_features.isna().any(axis=1).values[0]:
        return 0.5, None, 0.0

    prob = float(clf.predict_proba(latest_features)[0][1])
    pred_6m_ret = float(reg.predict(latest_features)[0])

    # Construct synthetic 126-day forward business-day path
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=127)[1:]

    # Smooth compounding trajectory toward target price
    target_price = current_price * (1.0 + pred_6m_ret)
    daily_growth = (target_price / current_price) ** (1 / 126)
    projected_prices = [current_price * (daily_growth ** i) for i in range(1, 127)]
    forecast_series = pd.Series(projected_prices, index=future_dates)

    return prob, forecast_series, pred_6m_ret


def plot_single_prediction(
    isin: str,
    ticker: str,
    df_history: pd.DataFrame,
    forecast_series: pd.Series | None,
    ai_prob: float,
    expected_ret: float,
    output_dir: str = PLOTS_DIR,
) -> None:
    if forecast_series is None or df_history.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Create an isolated figure instance per thread
    fig, ax = plt.subplots(figsize=(10, 5))

    history_slice = df_history["Close"].iloc[-380:]
    ax.plot(history_slice.index, history_slice.values, label="Historical Price", color="#1f77b4", lw=2)

    bridge_dates = [history_slice.index[-1], forecast_series.index[0]]
    bridge_prices = [history_slice.values[-1], forecast_series.values[0]]
    ax.plot(bridge_dates, bridge_prices, color="#ff7f0e", linestyle="--", lw=2)

    ax.plot(forecast_series.index, forecast_series.values, label=f"AI Forecast (+{expected_ret*100:.1f}%)", color="#ff7f0e", lw=2)

    rolling_vol = float(df_history["Close"].pct_change().std() * np.sqrt(252))
    upper_band = forecast_series * (1 + rolling_vol * np.sqrt(np.linspace(0.05, 0.5, len(forecast_series))))
    lower_band = forecast_series * (1 - rolling_vol * np.sqrt(np.linspace(0.05, 0.5, len(forecast_series))))
    ax.fill_between(forecast_series.index, lower_band, upper_band, color="#ff7f0e", alpha=0.18, label="Estimated Uncertainty")

    ax.axvline(x=history_slice.index[-1], color="gray", linestyle=":", label="Prediction Horizon")
    ax.set_title(f"{ticker} ({isin}) — 6-Month AI Projection (Bullish Prob: {ai_prob*100:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    filename = os.path.join(output_dir, f"{ticker}_{isin}.png")
    fig.savefig(filename, dpi=180)
    
    # Explicitly close this specific figure to free memory
    plt.close(fig)
    

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


def calculate_metrics(df: pd.DataFrame, ticker: yf.Ticker) -> tuple[dict | None, pd.Series | None, float]:
    if df is None or df.empty or "Close" not in df:
        return None, None, 0.0

    close = df["Close"].dropna()
    if len(close) < 252:
        return None, None, 0.0

    current_price = close.iloc[-1]

    sma_200 = close.rolling(window=200).mean().iloc[-1]
    trend_pct = ((current_price / sma_200) - 1) * 100

    lookback_1y = min(252, len(close) - 1)
    return_1y = ((current_price / close.iloc[-lookback_1y]) - 1) * 100
    return_5y = ((current_price / close.iloc[0]) - 1) * 100

    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() * 100

    daily_returns = close.pct_change().dropna()
    rf_daily = 0.03 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (
        (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

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
    if debt_to_equity is not None and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100.0

    operating_cashflow = info.get("operatingCashflow")
    positive_cashflow = (operating_cashflow > 0) if operating_cashflow is not None else None

    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")

    ai_prob, forecast_series, expected_ret = train_ai_predictor(df)

    metrics = {
        "Price": round(current_price, 2),
        "Sector": sector,
        "Industry": industry,
        "Div_Yield_%": div_yield,
        "Payout_Ratio": round(payout_ratio, 2) if payout_ratio is not None else None,
        "Sustainable_Div": sustainable_div,
        "PE_Ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
        "Debt_To_Equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
        "Positive_CashFlow": positive_cashflow,
        "1Y_Return_%": round(return_1y, 2),
        "5Y_Return_%": round(return_5y, 2),
        "Max_DD_%": round(max_drawdown, 2),
        "Sharpe": round(sharpe, 2),
        "Trend_SMA200_%": round(trend_pct, 2),
        "AI_Prob": round(ai_prob, 3),
    }

    return metrics, forecast_series, expected_ret


def rank_portfolio(records: list[dict], max_per_sector: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(records)
    if df.empty:
        return df, df

    df["rank_ret_5y"] = df["5Y_Return_%"].rank(pct=True)
    df["rank_sharpe"] = df["Sharpe"].rank(pct=True)
    df["rank_mom"] = df["1Y_Return_%"].rank(pct=True)
    df["rank_risk"] = df["Max_DD_%"].rank(pct=True)
    df["rank_trend"] = df["Trend_SMA200_%"].rank(pct=True)

    clean_yield = df.apply(
        lambda r: r["Div_Yield_%"] if r["Sustainable_Div"] else 0.0, axis=1
    )
    df["rank_dividend"] = clean_yield.rank(pct=True)

    def pe_score(pe):
        if pe is None or pe <= 0:
            return 0.2
        if 5 <= pe <= 20:
            return 1.0
        if 20 < pe <= 35:
            return 0.6
        return 0.3

    df["val_score"] = df["PE_Ratio"].apply(pe_score)

    base_score = (
        df["rank_ret_5y"] * 0.20 +
        df["rank_sharpe"] * 0.20 +
        df["val_score"] * 0.15 +
        df["rank_dividend"] * 0.15 +
        df["rank_risk"] * 0.10 +
        df["rank_mom"] * 0.10 +
        df["rank_trend"] * 0.10
    ) * 100

    df["Score"] = (base_score * 0.65) + ((df["AI_Prob"] * 100) * 0.35)

    df.loc[df["Debt_To_Equity"] > 2.0, "Score"] *= 0.85
    df.loc[df["Positive_CashFlow"] == False, "Score"] *= 0.85
    df["Score"] = df["Score"].round(1)

    display_cols = [
        "ISIN", "Ticker", "Sector", "Score", "AI_Prob", "Price",
        "PE_Ratio", "Div_Yield_%", "Payout_Ratio", "5Y_Return_%", "Sharpe", "Max_DD_%"
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


def process_single_isin(isin: str) -> tuple[dict | None, pd.DataFrame | None, pd.Series | None, float]:
    try:
        ticker, ticker_obj, df_history = get_history_by_isin(isin)
        metrics, forecast_series, expected_ret = calculate_metrics(df_history, ticker_obj)

        if metrics:
            metrics["ISIN"] = isin
            metrics["Ticker"] = ticker
            save_instrument(
                isin=isin,
                ticker=ticker,
                sector=metrics["Sector"],
                industry=metrics["Industry"],
                metrics=metrics,
                status="processed",
            )
            # Generate and save chart for this stock
            plot_single_prediction(
                isin=isin,
                ticker=ticker,
                df_history=df_history,
                forecast_series=forecast_series,
                ai_prob=metrics["AI_Prob"],
                expected_ret=expected_ret,
            )
            print(f"✓ {isin} ({ticker}) — Scored & Plot Saved [AI Prob: {metrics['AI_Prob']}]")
            return metrics, df_history, forecast_series, expected_ret
        else:
            save_instrument(isin=isin, ticker=ticker, status="skipped")
            print(f"⚠ {isin} ({ticker}) — Skipped: < 1 year of data")
            return None, None, None, 0.0

    except Exception as e:
        save_instrument(isin=isin, status="failed", error=str(e))
        print(f"✗ {isin} — Failed: {e}")
        return None, None, None, 0.0


def plot_top_predictions_grid(top_picks: list[dict], output_file: str = "top_predictions_grid.png"):
    """Creates a side-by-side 2x3 grid comparison of top candidates' projected paths."""
    if not top_picks:
        return

    n_plots = min(6, len(top_picks))
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 9))
    axes = axes.flatten()

    for idx in range(n_plots):
        item = top_picks[idx]
        ax = axes[idx]
        hist = item["history"]["Close"].iloc[-250:]
        f_series = item["forecast"]
        ticker = item["metrics"]["Ticker"]
        isin = item["metrics"]["ISIN"]
        prob = item["metrics"]["AI_Prob"]

        ax.plot(hist.index, hist.values, color="#1f77b4", label="History")
        if f_series is not None:
            bridge_dates = [hist.index[-1], f_series.index[0]]
            bridge_vals = [hist.values[-1], f_series.values[0]]
            ax.plot(bridge_dates, bridge_vals, color="#ff7f0e", linestyle="--")
            ax.plot(f_series.index, f_series.values, color="#ff7f0e", label="6M AI Forecast")
            ax.axvline(x=hist.index[-1], color="gray", linestyle=":")

        ax.set_title(f"#{idx+1}: {ticker} ({isin}) | Score: {item['metrics']['Score']} | Prob: {prob*100:.0f}%", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    # Turn off any unused subplot frames
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Top Investment Candidates: 6-Month Forward AI Projections", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"✓ Summary comparison grid saved to '{output_file}'")


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"

    limit_count = None
    isin_list = get_isin_list(url=url, limit=limit_count)
    print(f"Loaded {len(isin_list)} ISINs. Starting multi-threaded analysis...\n")

    initialize_database()

    collected_data = []
    bundle_store = {}
    max_workers = 3

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_isin, isin_list)
        for res in results:
            metrics, df_history, forecast_series, expected_ret = res
            if metrics is not None:
                collected_data.append(metrics)
                bundle_store[metrics["ISIN"]] = {
                    "metrics": metrics,
                    "history": df_history,
                    "forecast": forecast_series,
                    "ret": expected_ret,
                }

    elapsed = round(time.time() - start_time, 1)
    print(f"\nProcessing finished in {elapsed}s.")

    full_ranking, diversified = rank_portfolio(collected_data, max_per_sector=2)

    print("\n" + "=" * 115)
    print("FINAL INVESTMENT RANKINGS")
    print("=" * 115)
    print(full_ranking.to_string())

    save_scores(full_ranking)

    # Add scores into the bundle store for top grid generation
    top_candidates = []
    for _, row in full_ranking.head(6).iterrows():
        isin_val = row["ISIN"]
        if isin_val in bundle_store:
            bundle_store[isin_val]["metrics"]["Score"] = row["Score"]
            top_candidates.append(bundle_store[isin_val])

    plot_top_predictions_grid(top_candidates)
    print(f"\nIndividual stock charts written to '{PLOTS_DIR}/'.")
    print(f"Database records and scores updated in '{DATABASE_PATH}'.")


if __name__ == "__main__":
    main()