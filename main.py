"""Complete pipeline: update data -> analyze -> train -> rank.

For individual steps use:
    python update_data.py
    python analyze.py
    python train.py
    python rank.py
"""
import os

# main.py is the intentionally-online full workflow.
os.environ["OFFLINE"] = "0"

from investment.core import (
    collect_records,
    generate_ranking,
    get_isin_list,
    initialize_database,
    prepare_training_dataset,
    print_cache_summary,
    save_production_models,
    walk_forward_validate,
    train_production_models_from_dataset,
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
    records = collect_records(isins, max_workers=int(os.getenv("DOWNLOAD_WORKERS", "2")), phase_name="1/4 Cache-Update")
    if len(records) < 5:
        raise SystemExit("Zu wenige gültige Instrumente.")

    print("\n=== 2/4 Walk-Forward-Analyse ===")
    dataset, feature_cols = prepare_training_dataset(records, save_training_data=True)
    validation = walk_forward_validate(dataset, feature_cols)
    if not validation.empty:
        numeric_cols = ["roc_auc", "brier", "reg_mae", "spearman", "top20_excess"]
        print("\nDurchschnittliche Out-of-Sample-Metriken:")
        print(validation[numeric_cols].mean(numeric_only=True).round(3).to_string())

    print("\n=== 3/4 Produktionsmodell trainieren ===")
    classifier, regressor = train_production_models_from_dataset(dataset, feature_cols)
    save_production_models(classifier, regressor, feature_cols)

    print("\n=== 4/4 Ranking ===")
    create_plots = os.getenv("NO_PLOTS", "0").lower() not in {"1", "true", "yes", "on"}
    generate_ranking(records, classifier, regressor, feature_cols, create_plots=create_plots)
    print_cache_summary()


if __name__ == "__main__":
    main()
