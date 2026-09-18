import csv
import urllib.request


def get_isin_list(url, limit=10):
    response = urllib.request.urlopen(url)
    lines = [line.decode("utf-8") for line in response.readlines()]
    reader = csv.reader(lines, delimiter=";")

    next(reader)
    next(reader)
    next(reader)

    isin_list = []

    for row in list(reader)[:10]:
        print(row[3])
        isin_list.append(row[3])

    return isin_list


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    isin_list = get_isin_list(url=url, limit=10)

    print(isin_list)


if __name__ == "__main__":
    main()