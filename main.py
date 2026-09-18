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


import yfinance as yf

def get_price_by_isin(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Try fast_info primary and secondary fields
    price = ticker.fast_info.get("last_price")
    if price is None:
        price = ticker.fast_info.get("previous_close")
    
    currency = ticker.fast_info.get("currency")
    
    # 2. Fall back to 1-day history if fast_info returned None
    if price is None:
        hist = ticker.history(period="5d")  # 5d covers weekends and holidays
        if not hist.empty:
            price = hist["Close"].dropna().iloc[-1]
            
    # 3. Fall back to regular .info dictionary as last resort
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


def get_prices(isin_list):
    for isin in isin_list:
        price, currency = get_price_by_isin(isin)
        print(price, currency)


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    isin_list = get_isin_list(url=url, limit=100)

    print(isin_list)

    get_prices(isin_list)


if __name__ == "__main__":
    main()