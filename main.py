import csv
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


def get_prices(isin_list):
    for isin in isin_list:
        price, currency = get_price_by_isin(isin)
        print(price, currency)
        ticker_symbol, df_history = get_history_by_isin(isin)
        
        print("\n--- First 5 Trading Days ---")
        print(df_history.head())

        print("\n--- Most Recent 5 Trading Days ---")
        print(df_history.tail())


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    isin_list = get_isin_list(url=url, limit=100)

    print(isin_list)

    get_prices(isin_list)


if __name__ == "__main__":
    main()