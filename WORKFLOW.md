# Investment workflow

The project is split so that analysis, training and ranking can run **without any network access**.

## 1. Update/download cache

```bash
python update_data.py
```

Optional:

```bash
MAX_INSTRUMENTS=500 python update_data.py
FORCE_REFRESH=1 python update_data.py
DOWNLOAD_WORKERS=4 python update_data.py
```

This is the only dedicated step that contacts Xetra/Yahoo.

## 2. Analyze existing data only

```bash
python analyze.py
```

This uses only locally cached ticker mappings, prices, fundamentals and features. It writes
`walk_forward_metrics.csv` and does not train/save the production model.

To analyze a reproducible random subset of cached instruments:

```bash
MAX_INSTRUMENTS=500 python analyze.py
```

## 3. Train production model from existing cache

```bash
python train.py
```

The trained model is saved to:

```text
models/production_models.joblib
```

No network access occurs.

## 4. Generate ranking from existing cache + saved model

```bash
python rank.py
```

Skip plots for a faster run:

```bash
NO_PLOTS=1 python rank.py
```

No network access occurs.

## Complete workflow

```bash
python main.py
```

This intentionally performs all steps: update -> analyze -> train -> rank.

## Typical usage

Occasionally update market data:

```bash
python update_data.py
```

Experiment with the model as often as you want without downloading anything:

```bash
python analyze.py
```

When satisfied with the model:

```bash
python train.py
python rank.py
```


## Random Forest workers / Python 3.14 warning

The project defaults to serial Random Forest execution for maximum compatibility:

```bash
RF_N_JOBS=1 python analyze.py
```

`RF_N_JOBS=1` is already the default, so normally you can simply run:

```bash
python analyze.py
```

If your Python/scikit-learn environment works cleanly with parallel forests, you can opt in:

```bash
RF_N_JOBS=4 python analyze.py
RF_N_JOBS=4 python train.py
RF_N_JOBS=4 python rank.py
```

Use `RF_N_JOBS=-1` only if you explicitly want scikit-learn to use all available CPUs.
The code no longer uses `joblib.parallel_config(...)`; each Random Forest controls its own worker count.

## Training progress bar

Random Forest training now shows tree-level progress, elapsed time and an ETA. Example:

```text
Production classifier        [████████████░░░░░░░░░░░░░░░░░░] 100/250 ( 40.0%) | 2m 14s | ETA 3m 21s
```

The default update interval is 10 trees. You can make the bar update more or less often:

```bash
RF_PROGRESS_BATCH=5 python train.py
RF_PROGRESS_BATCH=25 python train.py
```

Disable the progress bar and use the original one-shot `.fit()` behavior:

```bash
RF_PROGRESS=0 python train.py
```

The same progress reporting is also used for Random Forest fits during `python analyze.py` walk-forward validation.
