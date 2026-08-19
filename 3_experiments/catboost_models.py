"""CatBoost drop-in replacements for classifier/classify.py and classifier/regression.py.

This is the entire B1 experiment: identical data, identical splits, identical
evaluation metrics as baseline_0 (XGBoost) — the only change is the model
family. Every function here is a line-for-line mirror of its XGBoost
counterpart with the estimator swapped and hyperparameters translated 1:1
(objective/loss, n_estimators/iterations, learning_rate, device, scale_pos_weight).
Hyperparameters that were never explicitly tuned in the original (e.g. the
regressor's tree depth) are left at each library's own default, same as the
original code did for XGBoost.

See 1_snakemake/classifier/classify.py::binary_classifier/predict_binary and
1_snakemake/classifier/regression.py::xgboost_regression/predict_axiom_assays
for the XGBoost originals this mirrors.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_snakemake"))
from classifier.classify import detect_gpu_count  # noqa: E402, F401  (re-exported for callers)


def cpu_worker_count() -> int:
    """Respect OMP_NUM_THREADS (this session is capped at 4 even though the host has 32 cores)."""
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit():
        return max(int(env), 1)
    return max(len(os.sched_getaffinity(0)), 1)


# Benchmarked on this machine (see notebook): a single CatBoost GPU fit costs ~1.3s of
# pure CUDA-context init overhead before training even starts, which dominates when
# training thousands of tiny per-label models. CPU (thread_count=1, parallelized across
# models instead of within one model) was measurably faster end-to-end. XGBoost's GPU
# path did not have this problem, so this is a CatBoost-specific, benchmarked choice,
# not a blanket "GPU is bad" claim.

# CatBoost auto-selects `boosting_type="Ordered"` for datasets under ~50k rows (a
# permutation-driven scheme intended to reduce target leakage/overfitting on small
# data) -- benchmarked at ~6.5x slower per fit than "Plain" on this pipeline's
# per-label subsets (52.4s vs 8.1s for one fit), which would put the full classifier
# run at ~6h instead of ~1h. Explicitly set to "Plain" here: a deliberate choice for a
# closer "same gradient-boosting recipe, different library" comparison against
# XGBoost (which has no Ordered-boosting equivalent), not a hidden default.
BOOSTING_TYPE = "Plain"

# --------------------------------------------------------------------------
# Binary classifier (mirrors classifier/classify.py)
# --------------------------------------------------------------------------


def binary_classifier_catboost(
    dat: pd.DataFrame,
    meta: pd.DataFrame,
    n_splits: int,
    gpu_id: int | None,
    *,
    shuffle: bool = False,
    cc: bool = False,
) -> pl.DataFrame:
    """Same StratifiedKFold(n_splits) CV scheme as classify.binary_classifier, CatBoost instead of XGBoost."""
    dat["Label"] = dat["Label"].astype(int)
    x = dat.drop(columns=["Label"])
    y = dat["Label"]

    if cc:
        x = x[["Cell_Count"]]
    else:
        x = x.drop(columns=["Cell_Count"])

    if shuffle:
        y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=n_splits)

    pred_df = []
    fold = 1
    for train_index, val_index in kf.split(x, y):
        x_fold_train, x_fold_val = x.iloc[train_index].to_numpy(), x.iloc[val_index].to_numpy()
        y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]

        le = LabelEncoder()
        y_fold_train = le.fit_transform(y_fold_train)
        y_fold_val = le.fit_transform(y_fold_val)

        meta_fold_val = meta.iloc[val_index]

        model_kwargs = dict(
            loss_function="Logloss",
            iterations=150,
            learning_rate=0.05,
            boosting_type=BOOSTING_TYPE,
            scale_pos_weight=(y_fold_train == 0).sum() / (y_fold_train == 1).sum(),
            verbose=False,
        )
        if gpu_id is not None:
            model_kwargs.update(task_type="GPU", devices=str(gpu_id))
        else:
            model_kwargs.update(thread_count=1)  # parallelize across models (thread_map), not within one
        model = CatBoostClassifier(**model_kwargs)

        model.fit(x_fold_train, y_fold_train)

        y_fold_prob = model.predict_proba(x_fold_val)[:, 1]
        y_fold_pred = model.predict(x_fold_val)

        pred_df.append(
            pl.DataFrame({
                "Metadata_OASIS_ID": list(meta_fold_val["Metadata_OASIS_ID"]),
                "y_prob": list(y_fold_prob),
                "y_pred": list(y_fold_pred),
                "y_actual": list(y_fold_val),
                "k_fold": fold,
            }),
        )
        fold += 1

    return pl.concat(pred_df, how="vertical")


def process_label_and_agg_catboost(dat, label_column, agg_type, n_splits, labels, gpu_id, *, shuffle: bool = False, cc: bool = False):
    """Mirrors classify.process_label_and_agg, calling the CatBoost classifier."""
    try:
        prof = dat.filter(
            (pl.col("Metadata_AggType") == agg_type) & (pl.col(label_column).is_not_null())
        ).rename({label_column: "Label"})

        num_0 = prof.filter(pl.col("Label") == 0).height
        num_1 = prof.filter(pl.col("Label") == 1).height

        if (num_0 >= n_splits) & (num_1 >= n_splits):
            meta_cols = [i for i in prof.columns if "Metadata_" in i]
            all_meta_cols = [i for i in prof.columns if i in labels] + meta_cols

            prof_meta = prof.select(meta_cols)
            prof = prof.drop(all_meta_cols)

            class_res = binary_classifier_catboost(
                prof.to_pandas(),
                prof_meta.to_pandas(),
                n_splits=n_splits,
                gpu_id=gpu_id,
                shuffle=shuffle,
                cc=cc,
            )

            class_res = class_res.with_columns(
                pl.lit(agg_type).alias("Metadata_AggType"),
                pl.lit(label_column).alias("Metadata_Label"),
                pl.lit(num_0).alias("Metadata_Count_0"),
                pl.lit(num_1).alias("Metadata_Count_1"),
            )

            return class_res

    except Exception:
        print(f"An error occurred for label '{label_column}' and aggregation type '{agg_type}':")
        print(traceback.format_exc())
        return None


def predict_binary_catboost(input_path: str, label_path: str, output_path: str, *, use_gpu: bool = False, num_workers: int | None = None) -> None:
    """Mirrors classify.predict_binary exactly: same 3 Model_types (Actual/Random_baseline/Cellcount_baseline).

    Defaults to CPU (see the benchmark note above the module-level constants) with
    thread_count=1 per model, parallelized across models via `num_workers` threads.
    """
    n_splits = 5
    gpu_count = detect_gpu_count() if use_gpu else 0
    if num_workers is None:
        num_workers = max(gpu_count, 1) if use_gpu else cpu_worker_count()

    dat = pl.read_parquet(input_path)
    meta = pl.read_parquet(label_path).rename({"OASIS_ID": "Metadata_OASIS_ID"})
    labels = [i for i in meta.columns if "Metadata_" not in i]

    dat = dat.join(meta, on="Metadata_OASIS_ID", how="left")

    agg_types = dat.select("Metadata_AggType").to_series().unique().to_list()
    tasks = [
        (dat, label_column, agg_type, n_splits, labels, (i % gpu_count) if gpu_count else None)
        for i, (label_column, agg_type) in enumerate(
            [(label_column, agg_type) for label_column in labels for agg_type in agg_types]
        )
    ]

    pred_results = thread_map(
        lambda args: process_label_and_agg_catboost(*args, shuffle=False),
        tasks, max_workers=num_workers, desc="Actual",
    )
    pred_results = [res for res in pred_results if res is not None]
    pred_df = pl.concat(pred_results, how="vertical").with_columns(pl.lit("Actual").alias("Model_type")) if pred_results else pl.DataFrame()

    null_results = thread_map(
        lambda args: process_label_and_agg_catboost(*args, shuffle=True),
        tasks, max_workers=num_workers, desc="Random_baseline",
    )
    null_results = [res for res in null_results if res is not None]
    null_df = pl.concat(null_results, how="vertical").with_columns(pl.lit("Random_baseline").alias("Model_type")) if null_results else pl.DataFrame()

    cc_results = thread_map(
        lambda args: process_label_and_agg_catboost(*args, cc=True),
        tasks, max_workers=num_workers, desc="Cellcount_baseline",
    )
    cc_results = [res for res in cc_results if res is not None]
    cc_df = pl.concat(cc_results, how="vertical").with_columns(pl.lit("Cellcount_baseline").alias("Model_type")) if cc_results else pl.DataFrame()

    if not pred_df.is_empty() and not null_df.is_empty() and not cc_df.is_empty():
        pl.concat([pred_df, null_df, cc_df], how="vertical").write_parquet(output_path)


# --------------------------------------------------------------------------
# Continuous regression (mirrors classifier/regression.py)
# --------------------------------------------------------------------------


def catboost_regression(dat: pd.DataFrame, target: str, feat_cols: list, split_group: str, *, mean_pred: bool = False, gpu_id: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same GroupShuffleSplit(n_splits=10, test_size=0.2) scheme as regression.xgboost_regression, CatBoost instead of XGBoost."""
    groups = dat[split_group]
    gss = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    i = 1
    results = []
    pred_obs = []

    for split_idx, (train_idx, test_idx) in tqdm(enumerate(gss.split(dat, groups=groups))):
        train_data = dat.iloc[train_idx]
        test_data = dat.iloc[test_idx]

        train_data = train_data.dropna(subset=[target]).reset_index(drop=True)
        test_data = test_data.dropna(subset=[target]).reset_index(drop=True)

        X_train = train_data[feat_cols]
        y_train = train_data[target]
        X_test = test_data[feat_cols]
        y_test = test_data[target]

        if mean_pred:
            mean_value = np.mean(y_train)
            predictions = np.full(len(y_test), mean_value)
        else:
            model_kwargs = dict(loss_function="RMSE", boosting_type=BOOSTING_TYPE, verbose=False)
            if gpu_id is not None:
                model_kwargs.update(task_type="GPU", devices=str(gpu_id))
            else:
                # Regression isn't parallelized across an outer thread pool (unlike the
                # classifier), so each model gets the full CPU budget to itself here.
                model_kwargs.update(thread_count=cpu_worker_count())
            model = CatBoostRegressor(**model_kwargs)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)

        results.append((target, split_idx, r2, rmse, mae))
        pred_obs.append(pl.DataFrame({
            "Predicted": predictions,
            "Observed": y_test.values,
            "Metadata_Plate": test_data["Metadata_Plate"].values,
            "Metadata_Well": test_data["Metadata_Well"].values,
            "Metadata_Compound": test_data["Metadata_Compound"].values,
            "Metadata_OASIS_ID": test_data["Metadata_OASIS_ID"].values,
            "Metadata_Log10Conc": test_data["Metadata_Log10Conc"].values,
            "Variable": target,
            "Split": i,
        }))

    results_df = pd.DataFrame(results, columns=["Variable", "Split", "R²", "RMSE", "MAE"])
    pred_obs_df = pl.concat(pred_obs, how="vertical_relaxed").to_pandas()

    return results_df, pred_obs_df


def predict_axiom_assays_catboost(prof_path: str, prediction_path: str, results_path: str, *, use_gpu: bool = False) -> None:
    """Mirrors regression.predict_axiom_assays exactly: same 2 targets (MTT/LDH ridge-norm),
    same 3 Model_types (Morphology/Baseline/Mean_predictor), same baseline feature columns.

    Defaults to CPU: benchmarked at 22.8s/fit (thread_count=4) vs 27.1s/fit on GPU for this
    profile size (~15k rows x 667 features) -- GPU wins only once transfer overhead is
    amortized over much larger data than this pipeline ever fits in one regression call.
    """
    gpu_id = 0 if (use_gpu and detect_gpu_count()) else None

    profiles = pd.read_parquet(prof_path)
    profiles = profiles.dropna(subset=["Metadata_OASIS_ID"]).reset_index(drop=True)
    profiles = profiles[profiles["Metadata_Perturbation"] != "DMSO_0.0"].reset_index(drop=True)
    profiles["Metadata_Log10Conc"] = np.round(profiles["Metadata_Log10Conc"], 2)

    profiles["Metadata_Plate_cat"] = profiles["Metadata_Plate"].astype("category").cat.codes
    profiles["Metadata_source_cat"] = profiles["Metadata_source"].astype("category").cat.codes
    profiles["Metadata_Well_cat"] = profiles["Metadata_Well"].astype("category").cat.codes
    profiles["Metadata_Compound_cat"] = profiles["Metadata_Compound"].astype("category").cat.codes

    feat_cols = [i for i in profiles.columns if "Metadata" not in i]
    baseline_cols = ["Metadata_Plate_cat", "Metadata_Well_cat", "Metadata_source_cat", "Metadata_Count_Cells"]

    targets = [
        ("Metadata_ldh_ridge_norm", "ldh"),
        ("Metadata_mtt_ridge_norm", "mtt"),
    ]

    all_pred, all_res = [], []
    for target, _short in targets:
        res, pred = catboost_regression(profiles, target, feat_cols, "Metadata_Compound", gpu_id=gpu_id)
        pred["Variable_Name"], pred["Model_type"] = target, "Morphology"
        res["Variable_Name"], res["Model_type"] = target, "Morphology"
        all_pred.append(pred)
        all_res.append(res)

        bl_res, bl_pred = catboost_regression(profiles, target, baseline_cols, "Metadata_Compound", gpu_id=gpu_id)
        bl_pred["Variable_Name"], bl_pred["Model_type"] = target, "Baseline"
        bl_res["Variable_Name"], bl_res["Model_type"] = target, "Baseline"
        all_pred.append(bl_pred)
        all_res.append(bl_res)

        mean_res, mean_pred = catboost_regression(profiles, target, [], "Metadata_Compound", mean_pred=True)
        mean_pred["Variable_Name"], mean_pred["Model_type"] = target, "Mean_predictor"
        mean_res["Variable_Name"], mean_res["Model_type"] = target, "Mean_predictor"
        all_pred.append(mean_pred)
        all_res.append(mean_res)

    pd.concat(all_pred, ignore_index=True).to_parquet(prediction_path)
    pd.concat(all_res, ignore_index=True).to_parquet(results_path)
