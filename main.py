import csv
import itertools
import sqlite3
import time
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

DATABASE_PATH = "ranking.db"


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the table used to persist each instrument with metrics and safety flags."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            isin TEXT PRIMARY KEY,
            ticker TEXT,
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
            score REAL
        )
        """
    )
    connection.commit()


def save_instrument(
    connection: sqlite3.Connection,
    isin: str,
    ticker: str | None = None,
    metrics: dict | None = None,
    status: str = "processed",
    error: str | None = None,
) -> None:
    """Persist metrics and fundamental checks for an instrument immediately."""
    pos_cf = None
    if metrics and metrics.get("Positive_CashFlow") is not None:
        pos_cf = 1 if metrics.get("Positive_CashFlow") else 0

    connection.execute(
        """
        INSERT INTO portfolio (
            isin, ticker, status, error, price, div_yield, payout_ratio,
            pe_ratio, debt_to_equity, positive_cashflow, one_year_return,
            five_year_return, max_drawdown, sharpe, trend_sma200, score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(isin) DO UPDATE SET
            ticker = excluded.ticker,
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
            score = NULL
        """,
        (
            isin,
            ticker,
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
        ),
    )
    connection.commit()


def save_scores(connection: sqlite3.Connection, ranked_table: pd.DataFrame) -> None:
    """Persist final composite scores back to the database."""
    if ranked_table.empty:
        return

    connection.executemany(
        "UPDATE portfolio SET score = ? WHERE isin = ?",
        ranked_table[["Score", "ISIN"]].itertuples(index=False, name=None),
    )
    connection.commit()


def get_isin_list(url: str, limit: int | None = None) -> list[str]:
    """Download CSV from Deutsche Börse and return clean, unique 12-char ISINs."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        lines = [line.decode("utf-8", errors="ignore") for line in response.readlines()]

    reader = csv.reader(lines, delimiter=";")

    # Skip header rows
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
    isin: str, preferred_exchange: str | None = None
) -> tuple[str, yf.Ticker, pd.DataFrame]:
    """Resolve ISIN via Yahoo Search and fetch 5-year price history."""
    search = yf.Search(isin, max_results=5)
    if not search.quotes:
        raise ValueError(f"No ticker found for ISIN: {isin}")

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


def calculate_metrics(df: pd.DataFrame, ticker: yf.Ticker) -> dict | None:
    """Compute technical, momentum, downside, and fundamental safety metrics."""
    if df is None or df.empty or "Close" not in df:
        return None

    close = df["Close"].dropna()
    if len(close) < 252:
        return None

    current_price = close.iloc[-1]

    # --- 1. Technical & Momentum Metrics ---
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    trend_pct = ((current_price / sma_200) - 1) * 100

    lookback_1y = min(252, len(close) - 1)
    return_1y = ((current_price / close.iloc[-lookback_1y]) - 1) * 100
    return_5y = ((current_price / close.iloc[0]) - 1) * 100

    # --- 2. Downside & Volatility Metrics ---
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

    # --- 3. Fundamentals & Guardrails ---
    info = ticker.info or {}

    # Dividend Yield & Dividend Trap Flag
    raw_yield = ticker.fast_info.get("dividend_yield") or info.get("dividendYield") or 0.0
    div_yield = raw_yield * 100 if raw_yield < 1 else raw_yield

    payout_ratio = info.get("payoutRatio")
    sustainable_div = True
    if div_yield > 0 and payout_ratio is not None:
        # High payouts (>85% of net income) or negative earnings indicate danger
        if payout_ratio > 0.85 or payout_ratio < 0:
            sustainable_div = False

    # Valuation & Leverage
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    debt_to_equity = info.get("debtToEquity")
    if debt_to_equity is not None and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100.0  # Normalize percentage to ratio

    # Cash Flow Check
    operating_cashflow = info.get("operatingCashflow")
    positive_cashflow = (operating_cashflow > 0) if operating_cashflow is not None else None

    return {
        "Price": round(current_price, 2),
        "Div_Yield_%": round(div_yield, 2),
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
    }


def rank_portfolio(records: list[dict]) -> pd.DataFrame:
    """Rank instruments using weighted percentiles plus fundamental guardrail penalties."""
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Relative Percentile Rankings (0.0 to 1.0)
    df["rank_ret_5y"] = df["5Y_Return_%"].rank(pct=True)
    df["rank_sharpe"] = df["Sharpe"].rank(pct=True)
    df["rank_mom"] = df["1Y_Return_%"].rank(pct=True)
    df["rank_risk"] = df["Max_DD_%"].rank(pct=True)  # Less negative is better
    df["rank_trend"] = df["Trend_SMA200_%"].rank(pct=True)

    # Filter out yield for identified dividend traps
    clean_yield = df.apply(
        lambda r: r["Div_Yield_%"] if r["Sustainable_Div"] else 0.0, axis=1
    )
    df["rank_dividend"] = clean_yield.rank(pct=True)

    # Valuation Multiplier: Prefer reasonable multiples (5-20), penalize negative/bubble P/Es
    def pe_score(pe):
        if pe is None or pe <= 0:
            return 0.2
        if 5 <= pe <= 20:
            return 1.0
        if 20 < pe <= 35:
            return 0.6
        return 0.3

    df["val_score"] = df["PE_Ratio"].apply(pe_score)

    # 100-Point Composite Model
    df["Score"] = (
        df["rank_ret_5y"] * 0.20 +       # Long-term growth
        df["rank_sharpe"] * 0.20 +       # Risk-adjusted consistency
        df["val_score"] * 0.15 +         # Reasonable valuation
        df["rank_dividend"] * 0.15 +     # Sustainable yield
        df["rank_risk"] * 0.10 +         # Capital preservation
        df["rank_mom"] * 0.10 +          # Medium-term momentum
        df["rank_trend"] * 0.10          # SMA trend alignment
    ) * 100

    # Fundamental Penalties:
    # 1. Over-leveraged balance sheet (Debt/Equity > 2.0)
    df.loc[df["Debt_To_Equity"] > 2.0, "Score"] *= 0.85

    # 2. Negative operating cash flow (burning operational cash)
    df.loc[df["Positive_CashFlow"] == False, "Score"] *= 0.85

    df["Score"] = df["Score"].round(1)

    display_cols = [
        "ISIN", "Ticker", "Score", "Price", "PE_Ratio",
        "Div_Yield_%", "Payout_Ratio", "Debt_To_Equity", "Positive_CashFlow",
        "5Y_Return_%", "1Y_Return_%", "Sharpe", "Max_DD_%"
    ]
    return df[display_cols].sort_values(by="Score", ascending=False).reset_index(drop=True)


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"

    # Set limit=None to scan the complete exchange list, or an int (e.g., 20) for testing
    limit_count = 20
    isin_list = get_isin_list(url=url, limit=limit_count)
    print(f"Collected {len(isin_list)} ISINs from Deutsche Börse.")

    collected_data = []

    with sqlite3.connect(DATABASE_PATH) as connection:
        initialize_database(connection)

        for idx, isin in enumerate(isin_list, start=1):
            try:
                ticker, ticker_obj, df_history = get_history_by_isin(isin)
                metrics = calculate_metrics(df_history, ticker_obj)

                if metrics:
                    metrics["ISIN"] = isin
                    metrics["Ticker"] = ticker
                    collected_data.append(metrics)
                    save_instrument(connection, isin, ticker, metrics)
                    print(f"[{idx}/{len(isin_list)}] ✓ Processed {isin} ({ticker})")
                else:
                    save_instrument(connection, isin, ticker, status="skipped")
                    print(f"[{idx}/{len(isin_list)}] ⚠ Skipped {isin} ({ticker}): < 1 year of data")
            except Exception as e:
                save_instrument(connection, isin, status="failed", error=str(e))
                print(f"[{idx}/{len(isin_list)}] ✗ Failed {isin}: {e}")

            time.sleep(0.3)  # Anti-throttling rate limit buffer

        print("\n" + "=" * 110)
        print("FINAL INVESTMENT RANKINGS (FUNDAMENTALS + PERFORMANCE + RISK)")
        print("=" * 110)
        ranked_table = rank_portfolio(collected_data)
        print(ranked_table.to_string())

        save_scores(connection, ranked_table)
        print(f"\nProcessing results saved to {DATABASE_PATH}")


if __name__ == "__main__":
    main()
