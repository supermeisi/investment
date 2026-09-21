"""Train and save production models using local cache only. No downloads occur.

Examples:
    python train.py
    MAX_INSTRUMENTS=1000 python train.py
"""
import os

os.environ["OFFLINE"] = "1"
os.environ["FORCE_REFRESH"] = "0"

from investment.core import collect_records, get_cached_isin_list, print_cache_summary, train_and_save_records


def main() -> None:
    env_limit = int(os.getenv("MAX_INSTRUMENTS", "0"))
    limit = None if env_limit <= 0 else env_limit
    print_cache_summary()
    isins = get_cached_isin_list(limit=limit)
    if not isins:
        raise SystemExit("Kein verwendbarer lokaler Cache vorhanden. Zuerst 'python update_data.py' ausführen.")
    records = collect_records(isins, max_workers=int(os.getenv("ANALYSIS_WORKERS", "4")), phase_name="Lokale Trainingsdaten")
    if len(records) < 5:
        raise SystemExit("Zu wenige gültige gecachte Instrumente für das Training.")
    train_and_save_records(records)


if __name__ == "__main__":
    main()
