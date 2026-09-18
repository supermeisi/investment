import csv
import urllib.request


def get_isin_list(url):
    response = urllib.request.urlopen(url)
    lines = [line.decode("utf-8") for line in response.readlines()]
    reader = csv.reader(lines, delimiter=";")

    print('Printing first 5 ISIN:')

    for row in list(reader)[3:8]:
        print(row[3])


def main():
    url = "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
    get_isin_list(url)


if __name__ == "__main__":
    main()