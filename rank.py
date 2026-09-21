"""Generate a ranking from local cache and the saved model only. No downloads occur.

Examples:
    python rank.py
    MAX_INSTRUMENTS=500 NO_PLOTS=1 python rank.py
"""
import os

os.environ["OFFLINE"] = "1"
os.environ["FORCE_REFRESH"] = "0"

from investment.core import (
    collect_records,
    generate_ranking,
    get_cached_isin_list,
    load_production_models,
    print_cache_summary,
)


def main() -> None:
    env_limit = int(os.getenv("MAX_INSTRUMENTS", "0"))
    limit = None if env_limit <= 0 else env_limit
    create_plots = os.getenv("NO_PLOTS", "0").lower() not in {"1", "true", "yes", "on"}

    print_cache_summary()
    isins = get_cached_isin_list(limit=limit)
    if not isins:
        raise SystemExit("Kein verwendbarer lokaler Cache vorhanden. Zuerst 'python update_data.py' ausführen.")

    classifier, regressor, feature_cols, metadata = load_production_models()
    print(f"✓ Modell geladen (trainiert: {metadata.get('trained_at', 'unbekannt')}).")

    records = collect_records(isins, max_workers=int(os.getenv("ANALYSIS_WORKERS", "4")), phase_name="Lokale Ranking-Daten")
    if not records:
        raise SystemExit("Keine gültigen gecachten Instrumente für das Ranking.")
    generate_ranking(records, classifier, regressor, feature_cols, create_plots=create_plots)


if __name__ == "__main__":
    main()
