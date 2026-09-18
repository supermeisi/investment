import csv
import numpy as np
import urllib.request
import yfinance as yf


def get_isin_list(url, limit=10):
    response = urllib.request.urlopen(url)
    lines = [line.decode("utf-8") for line in response.readlines()]
    reader = csv.reader(lines, delimiter=";")

    next(reader)
    next(reader)
    next(reader)

    isin_list = []

    for row in list(reader)[:limit]:
        print(row[3])
        isin_list.append(row[3])

    return isin_list


def get_price_by_isin(ticker_symbol: str):
    try:
        ticker = yf.Ticker(ticker_symbol)
    except:
        return None, None
    
    price = ticker.fast_info.get("last_price")
    if price is None:
        price = ticker.fast_info.get("previous_close")
    
    currency = ticker.fast_info.get("currency")

    if price is None:
        hist = ticker.history(period="5d")  # 5d covers weekends and holidays
        if not hist.empty:
            price = hist["Close"].dropna().iloc[-1]
            
    if price is None:
        info = ticker.info
        price = (
            info.get("regularMarketPrice") 
            or info.get("currentPrice") 
            or info.get("previousClose")
        )
        if not currency:
            currency = info.get("currency")
            
    return price, currency


def get_history_by_isin(isin: str, preferred_exchange: str | None = None) -> tuple[str, pd.DataFrame]:
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
    exchange = selected_quote.get("exchange", "Unknown")
    print(f"ISIN {isin} -> {ticker_symbol} (Exchange: {exchange})")
    
    ticker = yf.Ticker(ticker_symbol)
    
    df = ticker.history(period="5y")
    
    return ticker_symbol, df


def evaluate_investment(df: pd.DataFrame) -> dict:
    """
    Evaluates a 5-year OHLCV DataFrame using trend, momentum, and risk metrics.
    """
    close = df["Close"].dropna()
    if len(close) < 252:
        return {"decision": "INSUFFICIENT DATA", "details": {}}

    # 1. 200-day SMA Trend
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    current_price = close.iloc[-1]
    above_sma = bool(current_price > sma_200)

    # 2. 1-Year (252 trading days) Return
    lookback_1y = min(252, len(close) - 1)
    return_1y = (current_price / close.iloc[-lookback_1y] - 1) * 100

    # 3. 5-Year Maximum Drawdown
    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() * 100  # negative percentage

    # 4. Annualized Sharpe Ratio (assuming 3% risk-free rate)
    daily_returns = close.pct_change().dropna()
    rf_daily = 0.03 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Decision Matrix
    passes = {
        "Above 200 SMA": above_sma,
        "Positive 1Y Return": return_1y > 0,
        "Acceptable Drawdown (>-35%)": max_drawdown > -35,
        "Healthy Sharpe (>0.5)": sharpe > 0.5,
    }

    score = sum(passes.values())
    if score == 4:
        verdict = "INVEST (Strong Trend & Healthy Risk Profile)"
    elif above_sma and return_1y > 0:
        verdict = "WATCHLIST (Uptrend intact, but elevated risk or volatility)"
    else:
        verdict = "AVOID (Weak trend or negative momentum)"

    return {
        "verdict": verdict,
        "current_price": round(current_price, 2),
        "sma_200": round(sma_200, 2),
        "return_1y_pct": round(return_1y, 2),
        "max_drawdown_5y_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "checks": passes,
    }
    

def evaluate_investment(df: pd.DataFrame) -> dict:
    """
    Evaluates a 5-year OHLCV DataFrame using trend, momentum, and risk metrics.
    """
    close = df["Close"].dropna()
    if len(close) < 252:
        return {"decision": "INSUFFICIENT DATA", "details": {}}

    # 1. 200-day SMA Trend
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    current_price = close.iloc[-1]
    above_sma = bool(current_price > sma_200)

    # 2. 1-Year (252 trading days) Return
    lookback_1y = min(252, len(close) - 1)
    return_1y = (current_price / close.iloc[-lookback_1y] - 1) * 100

    # 3. 5-Year Maximum Drawdown
    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() * 100  # negative percentage

    # 4. Annualized Sharpe Ratio (assuming 3% risk-free rate)
    daily_returns = close.pct_change().dropna()
    rf_daily = 0.03 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Decision Matrix
    passes = {
        "Above 200 SMA": above_sma,
        "Positive 1Y Return": return_1y > 0,
        "Acceptable Drawdown (>-35%)": max_drawdown > -35,
        "Healthy Sharpe (>0.5)": sharpe > 0.5,
    }

    score = sum(passes.values())
    if score == 4:
        verdict = "INVEST (Strong Trend & Healthy Risk Profile)"
    elif above_sma and return_1y > 0:
        verdict = "WATCHLIST (Uptrend intact, but elevated risk or volatility)"
    else:
        verdict = "AVOID (Weak trend or negative momentum)"

    return {
        "verdict": verdict,
        "current_price": round(current_price, 2),
        "sma_200": round(sma_200, 2),
        "return_1y_pct": round(return_1y, 2),
        "max_drawdown_5y_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "checks": passes,
    }


def get_prices(isin_list):
    for isin in isin_list:
        price, currency = get_price_by_isin(isin)
        print(price, currency)
        ticker_symbol, df_history = get_history_by_isin(isin)
        
        print("\n--- First 5 Trading Days ---")
        print(df_history.head())

        print("\n--- Most Recent 5 Trading Days ---")
        print(df_history.tail())
        
        result = evaluate_investment(df_history)
        
        print(f"Verdict: {result['verdict']}")
        
        for metric, passed in result["checks"].items():
            print(f" - {metric}: {'PASS' if passed else 'FAIL'}")
            print(f"Metrics: 1Y Return={result['return_1y_pct']}%, Max DD={result['max_drawdown_5y_pct']}%, Sharpe={result['sharpe_ratio']}")


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    isin_list = get_isin_list(url=url, limit=100)

    print(isin_list)

    get_prices(isin_list)


if __name__ == "__main__":
    main()