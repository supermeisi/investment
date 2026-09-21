"""Run walk-forward analysis using local cache only. No downloads or updates occur.

Examples:
    python analyze.py
    MAX_INSTRUMENTS=500 python analyze.py
"""
import os

# Must be set before importing investment.core because cache mode is read at import time.
os.environ["OFFLINE"] = "1"
os.environ["FORCE_REFRESH"] = "0"

from investment.core import analyze_records, collect_records, get_cached_isin_list, print_cache_summary


def main() -> None:
    env_limit = int(os.getenv("MAX_INSTRUMENTS", "0"))
    limit = None if env_limit <= 0 else env_limit
    print_cache_summary()
    isins = get_cached_isin_list(limit=limit)
    if not isins:
        raise SystemExit("Kein verwendbarer lokaler Cache vorhanden. Zuerst 'python update_data.py' ausführen.")
    records = collect_records(isins, max_workers=int(os.getenv("ANALYSIS_WORKERS", "4")), phase_name="Lokale Analyse")
    if len(records) < 5:
        raise SystemExit("Zu wenige gültige gecachte Instrumente für die Analyse.")
    analyze_records(records)


if __name__ == "__main__":
    main()
