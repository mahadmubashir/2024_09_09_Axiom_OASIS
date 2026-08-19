"""Shared metric-computation and plotting utilities for Axiom OASIS pipeline experiments.

Every experiment notebook (baseline_0, b1_catboost, ...) is expected to write its
compiled metrics to:

    3_experiments/outputs/<experiment>/metrics/<experiment>_classifier_metrics.parquet
    3_experiments/outputs/<experiment>/metrics/<experiment>_regression_metrics.parquet

using `compile_classifier_metrics` / `summarize_regression_metrics` below -- not a
copy-pasted reimplementation -- so "same evaluation metrics" between experiments is
guaranteed by shared code, not just by two notebooks looking similar. Loading
functions then read from that shared convention, so any new experiment automatically
becomes comparable against baseline_0 (or any other experiment) just by naming it in
`load_classifier_metrics(...)` / `load_regression_metrics(...)`.

All figures are saved as both PNG (300dpi, for quick viewing / slides) and PDF
(vector, for LaTeX \\includegraphics) next to each other at the given path stem.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import Image, display
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

EXP_ROOT = Path(__file__).resolve().parent / "outputs"

# Consistent color coding across every figure in every experiment notebook:
# "real model" variants are green, "technical/naive baseline" variants are
# orange, "sanity-check baseline" variants are gray.
MODEL_TYPE_COLORS = {
    "Actual": "#1b7837",
    "Morphology": "#1b7837",
    "Cellcount_baseline": "#d95f02",
    "Baseline": "#d95f02",
    "Random_baseline": "#999999",
    "Mean_predictor": "#999999",
}
MODEL_TYPE_ORDER_CLASSIFIER = ["Actual", "Cellcount_baseline", "Random_baseline"]
MODEL_TYPE_ORDER_REGRESSION = ["Morphology", "Baseline", "Mean_predictor"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)


def savefig(fig, path) -> Path:
    """Save a figure as both PNG and PDF next to `path` (extension is replaced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def show(path) -> None:
    """Display a saved PNG inline in the notebook, regardless of matplotlib backend."""
    display(Image(filename=str(Path(path).with_suffix(".png"))))


def _short_name(variable_name: str) -> str:
    return variable_name.replace("Metadata_", "").replace("_ridge_norm", "").upper()


# --------------------------------------------------------------------------
# Loading metrics across one or more experiments
# --------------------------------------------------------------------------

def load_classifier_metrics(*experiments: str) -> pl.DataFrame:
    dfs = []
    for exp in experiments:
        p = EXP_ROOT / exp / "metrics" / f"{exp}_classifier_metrics.parquet"
        if not p.exists():
            print(f"[warn] no classifier metrics for '{exp}' at {p}")
            continue
        dfs.append(pl.read_parquet(p).with_columns(pl.lit(exp).alias("Experiment")))
    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()


def load_regression_metrics(*experiments: str) -> pl.DataFrame:
    dfs = []
    for exp in experiments:
        p = EXP_ROOT / exp / "metrics" / f"{exp}_regression_metrics.parquet"
        if not p.exists():
            print(f"[warn] no regression metrics for '{exp}' at {p}")
            continue
        dfs.append(pl.read_parquet(p).with_columns(pl.lit(exp).alias("Experiment")))
    return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()


# --------------------------------------------------------------------------
# Shared evaluation-metric computation (used by every experiment notebook)
# --------------------------------------------------------------------------

def _binary_metrics(y_pred, y_actual, y_prob):
    try:
        auroc = roc_auc_score(y_actual, y_prob)
    except ValueError:
        auroc = None
    try:
        precision, recall, _ = precision_recall_curve(y_actual, y_prob)
        prauc = auc(recall, precision)
    except ValueError:
        prauc = None
    return auroc, prauc


def compute_classifier_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    """AUROC/PRAUC per (Metadata_AggType, Metadata_Label, Model_type) from pooled out-of-fold predictions."""
    class_balance = predictions.select(
        ["Metadata_AggType", "Metadata_Label", "Metadata_Count_0", "Metadata_Count_1", "Model_type"],
    ).unique()

    grouped = predictions.group_by(["Metadata_AggType", "Metadata_Label", "Model_type"]).agg([
        pl.col("y_pred").alias("y_pred_list"),
        pl.col("y_actual").alias("y_actual_list"),
        pl.col("y_prob").alias("y_prob_list"),
    ])

    result = grouped.with_columns(
        pl.struct(["y_pred_list", "y_actual_list", "y_prob_list"])
        .map_elements(
            lambda s: _binary_metrics(s["y_pred_list"], s["y_actual_list"], s["y_prob_list"]),
            return_dtype=pl.List(pl.Float64),
        )
        .alias("metrics"),
    )
    result = result.with_columns([
        pl.col("metrics").list.get(0).alias("AUROC"),
        pl.col("metrics").list.get(1).alias("PRAUC"),
    ]).drop(["y_pred_list", "y_actual_list", "y_prob_list", "metrics"])

    return result.join(class_balance, on=["Metadata_AggType", "Metadata_Label", "Model_type"])


def compile_classifier_metrics(binary_pred_paths: dict, save_path, feat_type: str = "cpcnn") -> pl.DataFrame:
    """Compile AUROC/PRAUC across every outcome type in `binary_pred_paths` ({name: path}) and save."""
    out = pl.concat(
        [
            compute_classifier_metrics(pl.read_parquet(path)).with_columns(
                pl.lit(name).alias("Outcome_type"), pl.lit(feat_type).alias("Feat_type"),
            )
            for name, path in binary_pred_paths.items()
        ],
        how="vertical",
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(save_path)
    return out


def summarize_regression_metrics(raw_metrics: pl.DataFrame, save_path) -> pl.DataFrame:
    """Mean/SD of R2/RMSE/MAE per (Variable_Name, Model_type) across GroupShuffleSplit splits, and save."""
    summary = (
        raw_metrics.group_by(["Variable_Name", "Model_type"])
        .agg([
            pl.col("R²").mean().alias("R2_mean"), pl.col("R²").std().alias("R2_std"),
            pl.col("RMSE").mean().alias("RMSE_mean"), pl.col("RMSE").std().alias("RMSE_std"),
            pl.col("MAE").mean().alias("MAE_mean"), pl.col("MAE").std().alias("MAE_std"),
        ])
        .sort(["Variable_Name", "Model_type"])
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(save_path)
    return summary


# --------------------------------------------------------------------------
# Single-experiment figures
# --------------------------------------------------------------------------

def plot_classifier_auroc_summary(classifier_metrics: pl.DataFrame, save_path, title: str) -> Path:
    """Mean AUROC (+/- SD across labels) per outcome type, Actual vs baselines."""
    df = classifier_metrics.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df, x="Outcome_type", y="AUROC", hue="Model_type",
        hue_order=[m for m in MODEL_TYPE_ORDER_CLASSIFIER if m in df["Model_type"].unique()],
        palette=MODEL_TYPE_COLORS, errorbar="sd", ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (mean ± SD across labels)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_classifier_auroc_distribution(classifier_metrics: pl.DataFrame, save_path, title: str) -> Path:
    """Per-label AUROC spread (boxplot) per outcome type, Actual vs baselines."""
    df = classifier_metrics.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="Outcome_type", y="AUROC", hue="Model_type",
        hue_order=[m for m in MODEL_TYPE_ORDER_CLASSIFIER if m in df["Model_type"].unique()],
        palette=MODEL_TYPE_COLORS, showfliers=False, ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (per label)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_roc_curve(binary_predictions: pl.DataFrame, label: str, agg_type: str, save_path, title: str) -> Path:
    """ROC curve for one label, comparing Actual vs baselines (pooled out-of-fold predictions)."""
    df = binary_predictions.filter(
        (pl.col("Metadata_Label") == label) & (pl.col("Metadata_AggType") == agg_type),
    ).to_pandas()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for model_type in MODEL_TYPE_ORDER_CLASSIFIER:
        sub = df[df["Model_type"] == model_type]
        if sub.empty or sub["y_actual"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["y_actual"], sub["y_prob"])
        auc_val = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{model_type} (AUROC={auc_val:.2f})", color=MODEL_TYPE_COLORS[model_type])
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_regression_metric(regression_metrics: pl.DataFrame, metric: str, save_path, title: str) -> Path:
    """Grouped bar chart of a regression metric (R2/RMSE/MAE), Morphology vs baselines."""
    df = regression_metrics.to_pandas()
    variables = sorted(df["Variable_Name"].unique())
    model_types = [m for m in MODEL_TYPE_ORDER_REGRESSION if m in df["Model_type"].unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / len(model_types)
    x = np.arange(len(variables))
    for i, mt in enumerate(model_types):
        sub = df[df["Model_type"] == mt].set_index("Variable_Name")
        means = [sub.loc[v, f"{metric}_mean"] if v in sub.index else np.nan for v in variables]
        stds = [sub.loc[v, f"{metric}_std"] if v in sub.index else np.nan for v in variables]
        ax.bar(x + (i - (len(model_types) - 1) / 2) * width, means, width, yerr=stds,
               label=mt, color=MODEL_TYPE_COLORS.get(mt), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(v) for v in variables])
    ax.set_ylabel(f"{metric} (mean ± SD across splits)")
    if metric == "R2":
        ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(title="Model")
    ax.set_title(title)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_pred_vs_obs(predictions: pl.DataFrame, variable_name: str, model_type: str, save_path, title: str) -> Path:
    """Predicted vs observed scatter for one regression target/model."""
    df = predictions.filter(
        (pl.col("Variable_Name") == variable_name) & (pl.col("Model_type") == model_type),
    ).to_pandas()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["Observed"], df["Predicted"], s=8, alpha=0.25, color=MODEL_TYPE_COLORS.get(model_type, "#1b7837"))
    lo = min(df["Observed"].min(), df["Predicted"].min())
    hi = max(df["Observed"].max(), df["Predicted"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    r2 = np.corrcoef(df["Observed"], df["Predicted"])[0, 1] ** 2
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}  (R²≈{r2:.2f}, n={len(df)})")
    ax.legend()
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_pod_distribution(pods: pl.DataFrame, save_path, title: str) -> Path:
    df = pods.to_pandas()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.histplot(df["bmd"], bins=30, color=MODEL_TYPE_COLORS["Actual"], ax=ax)
    ax.set_xlabel("Morphological POD (log10 concentration)")
    ax.set_ylabel("Number of compounds")
    ax.set_title(f"{title}  (n={len(df)})")
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_pod_vs_cytotox(pods: pl.DataFrame, save_path, title: str) -> Path:
    """Morphological POD vs cytotoxicity POD. Points below y=x are bioactive
    before they become cytotoxic (a genuine phenotypic signal, not just cell death).
    Compounds with no detected cytotoxicity (cc_POD sentinel = 9999) are excluded."""
    df = pods.to_pandas()
    df = df[df["cc_POD"] < 9999]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = df["PAC_below_cc_POD"].map({True: MODEL_TYPE_COLORS["Actual"], False: MODEL_TYPE_COLORS["Cellcount_baseline"]})
    ax.scatter(df["cc_POD"], df["bmd"], s=14, alpha=0.6, c=colors)
    lo, hi = 0, max(df["cc_POD"].max(), df["bmd"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Cytotoxicity POD (log10 concentration)")
    ax.set_ylabel("Morphological POD (log10 concentration)")
    ax.set_title(f"{title}  (n={len(df)})")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MODEL_TYPE_COLORS["Actual"], markeredgecolor="none", label="Bioactive before cytotoxic"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MODEL_TYPE_COLORS["Cellcount_baseline"], markeredgecolor="none", label="Cytotoxic at/before bioactive"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Cross-experiment comparison figures (the point of this module)
# --------------------------------------------------------------------------

def _grouped_bar(ax, df, x_col: str, group_col: str, mean_col: str, std_col: str, palette: dict | None = None):
    xs = sorted(df[x_col].unique())
    groups = sorted(df[group_col].unique())
    width = 0.8 / max(len(groups), 1)
    x = np.arange(len(xs))
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g].set_index(x_col)
        means = [sub.loc[v, mean_col] if v in sub.index else np.nan for v in xs]
        stds = [sub.loc[v, std_col] if v in sub.index else np.nan for v in xs]
        color = palette.get(g) if palette else None
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, means, width, yerr=stds, label=g, color=color, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(xs, rotation=15, ha="right")
    return ax


def plot_experiment_comparison_auroc(classifier_metrics_multi: pl.DataFrame, save_path, title: str, model_type: str = "Actual") -> Path:
    """Compare mean AUROC (a single model_type, usually 'Actual') across experiments, per outcome type."""
    df = (
        classifier_metrics_multi.filter(pl.col("Model_type") == model_type)
        .group_by(["Outcome_type", "Experiment"])
        .agg(pl.col("AUROC").mean().alias("AUROC_mean"), pl.col("AUROC").std().alias("AUROC_std"))
        .to_pandas()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, df, x_col="Outcome_type", group_col="Experiment", mean_col="AUROC_mean", std_col="AUROC_std")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"AUROC ({model_type}, mean ± SD across labels)")
    ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_title(title)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


def plot_experiment_comparison_r2(regression_metrics_multi: pl.DataFrame, save_path, title: str, model_type: str = "Morphology") -> Path:
    """Compare R2 (a single model_type, usually 'Morphology') across experiments, per target variable."""
    df = regression_metrics_multi.filter(pl.col("Model_type") == model_type).to_pandas()
    df["Variable_Name"] = df["Variable_Name"].map(_short_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    _grouped_bar(ax, df, x_col="Variable_Name", group_col="Experiment", mean_col="R2_mean", std_col="R2_std")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"R² ({model_type}, mean ± SD across splits)")
    ax.legend(title="Experiment")
    ax.set_title(title)
    fig.tight_layout()
    path = savefig(fig, save_path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Dissertation-ready summary tables (CSV + LaTeX), built across experiments
# --------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path, float_format: str = "%.3f") -> dict:
    """Save a table as both CSV (data/Excel) and a booktabs-style .tex snippet
    ready for \\input{} into a LaTeX dissertation. Returns the two paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    tex_path = path.with_suffix(".tex")
    df.to_csv(csv_path, index=False, float_format=float_format)
    with open(tex_path, "w") as f:
        f.write(
            df.to_latex(index=False, float_format=lambda x: float_format % x, na_rep="--", escape=True)
        )
    return {"csv": csv_path, "tex": tex_path}


def build_classifier_summary_table(classifier_metrics_multi: pl.DataFrame, agg_type: str = "all") -> pd.DataFrame:
    """Outcome-type level AUROC/PRAUC, every Model_type, every experiment present in the input."""
    df = (
        classifier_metrics_multi.filter(pl.col("Metadata_AggType") == agg_type)
        .group_by(["Outcome_type", "Model_type", "Experiment"])
        .agg(pl.col("AUROC").mean().alias("AUROC"), pl.col("PRAUC").mean().alias("PRAUC"))
        .sort(["Outcome_type", "Model_type", "Experiment"])
        .rename({"Outcome_type": "Outcome Type", "Model_type": "Model"})
    )
    return df.to_pandas()


def build_axiom_label_table(classifier_metrics_multi: pl.DataFrame, agg_type: str = "all") -> pd.DataFrame:
    """Per-label (MTT/LDH/cell_count) AUROC/PRAUC, every Model_type, every experiment present in the input."""
    df = (
        classifier_metrics_multi.filter(
            (pl.col("Outcome_type") == "axiom") & (pl.col("Metadata_AggType") == agg_type),
        )
        .select(["Metadata_Label", "Model_type", "Experiment", "AUROC", "PRAUC"])
        .sort(["Metadata_Label", "Model_type", "Experiment"])
        .rename({"Metadata_Label": "Label", "Model_type": "Model"})
    )
    return df.to_pandas()


def build_regression_summary_table(regression_metrics_multi: pl.DataFrame) -> pd.DataFrame:
    """R2/RMSE/MAE, every Model_type, every experiment present in the input."""
    df = (
        regression_metrics_multi.with_columns(pl.col("Variable_Name").map_elements(_short_name, return_dtype=pl.Utf8))
        .select(["Variable_Name", "Model_type", "Experiment", "R2_mean", "R2_std", "RMSE_mean", "MAE_mean"])
        .sort(["Variable_Name", "Model_type", "Experiment"])
        .rename({
            "Variable_Name": "Target", "Model_type": "Model",
            "R2_mean": "R2", "R2_std": "R2 SD", "RMSE_mean": "RMSE", "MAE_mean": "MAE",
        })
    )
    return df.to_pandas()
