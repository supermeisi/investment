"""Download/update local market caches only.

Examples:
    python update_data.py
    MAX_INSTRUMENTS=500 python update_data.py
    FORCE_REFRESH=1 python update_data.py
"""
import os

# This workflow is intentionally online.
os.environ["OFFLINE"] = "0"

from investment.core import (
    collect_records,
    get_isin_list,
    initialize_database,
    print_cache_summary,
)

XETRA_URL = (
    "https://www.cashmarket.deutsche-boerse.com/resource/blob/1528/"
    "5d846a1a320a80f824bcd1b9db5f3067/data/t7-xetr-allTradableInstruments.csv"
)


def main() -> None:
    env_limit = int(os.getenv("MAX_INSTRUMENTS", "100"))
    limit = None if env_limit <= 0 else env_limit
    initialize_database()
    isins = get_isin_list(XETRA_URL, limit=limit)
    collect_records(isins, max_workers=int(os.getenv("DOWNLOAD_WORKERS", "2")), phase_name="Cache-Update")
    print_cache_summary()


if __name__ == "__main__":
    main()
