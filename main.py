import csv
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf


def get_isin_list(url: str, limit: int = 10) -> list[str]:
    # Custom User-Agent avoids 403 blocks from exchange servers
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        lines = [line.decode("utf-8", errors="ignore") for line in response.readlines()]

    reader = csv.reader(lines, delimiter=";")

    # Skip metadata/header rows
    next(reader, None)
    next(reader, None)
    next(reader, None)

    isin_list = []
    for row in reader:
        if len(row) > 3 and row[3].strip():
            isin_list.append(row[3].strip())
        if len(isin_list) >= limit:
            break

    return isin_list


def get_history_by_isin(isin: str, preferred_exchange: str | None = None) -> tuple[str, pd.DataFrame]:
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

    return ticker_symbol, df


def calculate_metrics(df: pd.DataFrame) -> dict | None:
    """Calculates investment scoring metrics from a 5-year OHLCV DataFrame."""
    if df is None or df.empty or "Close" not in df:
        return None

    close = df["Close"].dropna()
    if len(close) < 252:
        return None

    current_price = close.iloc[-1]

    # 1. 200-day SMA Trend
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    trend_pct = ((current_price / sma_200) - 1) * 100

    # 2. 1-Year (252 trading days) Return
    lookback_1y = min(252, len(close) - 1)
    return_1y = ((current_price / close.iloc[-lookback_1y]) - 1) * 100

    # 3. 5-Year Maximum Drawdown
    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() * 100  # negative percentage

    # 4. Annualized Sharpe Ratio (3% risk-free rate)
    daily_returns = close.pct_change().dropna()
    rf_daily = 0.03 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

    return {
        "Price": round(current_price, 2),
        "1Y_Return_%": round(return_1y, 2),
        "Max_DD_%": round(max_drawdown, 2),
        "Sharpe": round(sharpe, 2),
        "Trend_SMA200_%": round(trend_pct, 2),
    }


def rank_portfolio(records: list[dict]) -> pd.DataFrame:
    """Ranks collected stock records from best to worst using percentile ranking."""
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Relative percentile scoring (0.0 to 1.0)
    # Higher Max_DD (closer to 0) is better
    df["rank_sharpe"] = df["Sharpe"].rank(pct=True)
    df["rank_mom"] = df["1Y_Return_%"].rank(pct=True)
    df["rank_risk"] = df["Max_DD_%"].rank(pct=True)
    df["rank_trend"] = df["Trend_SMA200_%"].rank(pct=True)

    # 0 - 100 Composite Score
    df["Score"] = (
        df["rank_sharpe"] * 0.35 +
        df["rank_mom"] * 0.25 +
        df["rank_risk"] * 0.25 +
        df["rank_trend"] * 0.15
    ) * 100

    df["Score"] = df["Score"].round(1)

    columns = ["ISIN", "Ticker", "Score", "Price", "Sharpe", "1Y_Return_%", "Max_DD_%", "Trend_SMA200_%"]
    return df[columns].sort_values(by="Score", ascending=False).reset_index(drop=True)


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    
    # Start with 15-20 to avoid rate limits while testing
    isin_list = get_isin_list(url=url, limit=15)
    print(f"Collected {len(isin_list)} ISINs from Deutsche Börse.")

    collected_data = []

    for isin in isin_list:
        try:
            ticker, df_history = get_history_by_isin(isin)
            metrics = calculate_metrics(df_history)

            if metrics:
                metrics["ISIN"] = isin
                metrics["Ticker"] = ticker
                collected_data.append(metrics)
                print(f"✓ Processed {isin} ({ticker})")
            else:
                print(f"⚠ Skipped {isin} ({ticker}): Less than 1 year of data.")
        except Exception as e:
            print(f"✗ Failed {isin}: {e}")

    # Rank and display
    print("\n" + "=" * 80)
    print("BEST TO WORST INVESTMENT RANKING")
    print("=" * 80)
    ranked_table = rank_portfolio(collected_data)
    print(ranked_table.to_string())


if __name__ == "__main__":
    main()