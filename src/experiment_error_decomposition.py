from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from final_reproduce import (
    ROOT,
    clean_train,
    load_raw_data,
    make_daily_panel,
    make_recent_blend_prediction,
    make_sku_activity,
    postprocess_prediction,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "error_audit"
PRED_TARGET = "y_net_clip"
METRIC_TARGET = "y_net"
EPS = 1e-8

DEFAULT_CURRENT_BEST_MODEL = "recent14_18_21_w20_45_35_sunday0"

FOLDS = [
    {
        "fold": "recent_2025",
        "train_end": "2025-07-11",
        "valid_start": "2025-07-12",
        "horizon": 56,
    },
    {
        "fold": "seasonal_2024",
        "train_end": "2024-09-05",
        "valid_start": "2024-09-06",
        "horizon": 56,
    },
]

MODEL_CONFIGS = [
    {"model": "zero", "type": "zero"},
    {"model": "recent14_sunday0", "windows": (14,), "weights": (1.0,)},
    {"model": "recent18_sunday0", "windows": (18,), "weights": (1.0,)},
    {"model": "recent21_sunday0", "windows": (21,), "weights": (1.0,)},
    {"model": "recent28_sunday0", "windows": (28,), "weights": (1.0,)},
    {"model": "recent56_sunday0", "windows": (56,), "weights": (1.0,)},
    {"model": "recent84_sunday0", "windows": (84,), "weights": (1.0,)},
    {
        "model": "recent14_18_21_w20_45_35_sunday0",
        "windows": (14, 18, 21),
        "weights": (0.20, 0.45, 0.35),
    },
    {
        "model": "recent18_21_28_w35_50_15_sunday0",
        "windows": (18, 21, 28),
        "weights": (0.35, 0.50, 0.15),
    },
    {
        "model": "recent21_28_w50_50_sunday0",
        "windows": (21, 28),
        "weights": (0.50, 0.50),
    },
    {
        "model": "recent21_28_56_w25_50_25_sunday0",
        "windows": (21, 28, 56),
        "weights": (0.25, 0.50, 0.25),
    },
    {
        "model": "recent21_28_56_w35_45_20_sunday0",
        "windows": (21, 28, 56),
        "weights": (0.35, 0.45, 0.20),
    },
]


@dataclass(frozen=True)
class FoldConfig:
    fold: str
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    horizon: int


def log_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline error decomposition for HBAAC SKU demand forecasting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit CSV outputs.",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        default=["recent_2025", "seasonal_2024"],
        choices=[cfg["fold"] for cfg in FOLDS],
        help="Fold names to evaluate.",
    )
    parser.add_argument(
        "--current-best-model",
        type=str,
        default=DEFAULT_CURRENT_BEST_MODEL,
        help="Model name used as the comparison baseline in the segment winner table.",
    )
    return parser.parse_args()


def get_fold_configs(selected_folds: list[str]) -> list[FoldConfig]:
    fold_map = {cfg["fold"]: cfg for cfg in FOLDS}
    return [
        FoldConfig(
            fold=name,
            train_end=pd.Timestamp(fold_map[name]["train_end"]),
            valid_start=pd.Timestamp(fold_map[name]["valid_start"]),
            horizon=int(fold_map[name]["horizon"]),
        )
        for name in selected_folds
    ]


def make_actual_matrix(
    panel: pd.DataFrame,
    start_date,
    horizon: int,
    target_col: str = METRIC_TARGET,
) -> pd.DataFrame:
    start_date = pd.Timestamp(start_date)
    forecast_dates = pd.date_range(start_date, periods=horizon, freq="D")
    itemcodes = sorted(panel["ItemCode"].unique())

    actual = panel.loc[
        (panel["Date"] >= start_date) & (panel["Date"] <= forecast_dates[-1]),
        ["ItemCode", "Date", target_col],
    ].copy()

    actual_wide = (
        actual.pivot(index="ItemCode", columns="Date", values=target_col)
        .reindex(index=itemcodes, columns=forecast_dates)
        .fillna(0.0)
    )
    actual_wide.index.name = "ItemCode"
    actual_wide.columns.name = None

    if actual_wide.isna().any().any():
        raise ValueError("Actual matrix contains NaN values.")
    if actual_wide.shape != (len(itemcodes), horizon):
        raise ValueError(
            f"Unexpected actual matrix shape: {actual_wide.shape}, expected {(len(itemcodes), horizon)}"
        )

    return actual_wide.astype(float)


def make_zero_prediction(
    itemcodes: list[str],
    forecast_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    pred = pd.DataFrame(0.0, index=itemcodes, columns=forecast_dates)
    pred.index.name = "ItemCode"
    pred.columns.name = None
    return pred


def compute_sku_metric_info(
    panel: pd.DataFrame,
    train_end,
    target_col: str = METRIC_TARGET,
    profit_col: str = "profit",
    eps: float = EPS,
) -> pd.DataFrame:
    """
    Fold-specific metric metadata.

    weight_i = positive total profit by SKU / total positive profit
    scale_i = mean squared first difference of y_net up to train_end
    """

    train_end = pd.Timestamp(train_end)
    hist = panel.loc[panel["Date"] <= train_end].copy()

    profit_info = (
        hist.groupby("ItemCode", as_index=False)
        .agg(
            total_profit=(profit_col, "sum"),
            active_days=("y_gross", lambda x: int((x > 0).sum())),
        )
    )
    profit_info["positive_profit"] = profit_info["total_profit"].clip(lower=0)

    total_positive_profit = float(profit_info["positive_profit"].sum())
    if total_positive_profit <= 0:
        raise ValueError("Total positive profit is zero; WRMSSE weight construction failed.")
    profit_info["weight"] = profit_info["positive_profit"] / total_positive_profit

    sale_dates = (
        hist.loc[hist["y_gross"] > 0]
        .groupby("ItemCode", as_index=False)
        .agg(last_sale_date=("Date", "max"))
    )

    scale_series = (
        hist.sort_values(["ItemCode", "Date"])
        .groupby("ItemCode")[target_col]
        .apply(
            lambda s: float(np.mean(np.square(np.diff(s.to_numpy(dtype=float)))))
            if len(s) > 1
            else 0.0
        )
    )
    scale_info = scale_series.reset_index(name="scale")

    metric_info = (
        profit_info.merge(scale_info, on="ItemCode", how="left")
        .merge(sale_dates, on="ItemCode", how="left")
    )
    metric_info["scale"] = metric_info["scale"].fillna(0.0)
    metric_info["scale_safe"] = metric_info["scale"].clip(lower=eps)
    metric_info["zero_scale_flag"] = (metric_info["scale"] < eps).astype(int)
    metric_info["profit_rank"] = (
        metric_info["positive_profit"].rank(method="min", ascending=False).astype(int)
    )
    metric_info["days_since_last_sale"] = (
        train_end - metric_info["last_sale_date"]
    ).dt.days.fillna(9999).astype(int)

    metric_info = metric_info[
        [
            "ItemCode",
            "total_profit",
            "positive_profit",
            "weight",
            "scale",
            "scale_safe",
            "zero_scale_flag",
            "profit_rank",
            "active_days",
            "days_since_last_sale",
        ]
    ].copy()

    metric_info = metric_info.sort_values("profit_rank").reset_index(drop=True)

    if metric_info.isna().any().any():
        raise ValueError("Metric info contains NaN values.")

    return metric_info


def wrmsse_score(
    actual_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    metric_info: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    actual_wide = actual_wide.sort_index()
    pred_wide = pred_wide.sort_index()

    actual = actual_wide.to_numpy(dtype=float)
    pred = pred_wide.to_numpy(dtype=float)

    if actual.shape != pred.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs pred {pred.shape}")

    metric = metric_info.set_index("ItemCode").reindex(actual_wide.index)
    required_cols = ["weight", "scale_safe", "profit_rank", "total_profit", "positive_profit"]
    missing_cols = [col for col in required_cols if col not in metric.columns]
    if missing_cols:
        raise ValueError(f"Missing metric columns: {missing_cols}")
    if metric.isna().any().any():
        raise ValueError("Metric info reindexing produced NaN values.")

    diff = pred - actual
    mse = np.mean(np.square(diff), axis=1)
    rmsse = np.sqrt(mse / metric["scale_safe"].to_numpy(dtype=float))
    weighted_rmsse = metric["weight"].to_numpy(dtype=float) * rmsse

    detail = metric.reset_index().copy()
    detail["actual_total"] = actual.sum(axis=1)
    detail["pred_total"] = pred.sum(axis=1)
    detail["signed_bias"] = detail["pred_total"] - detail["actual_total"]
    detail["mse"] = mse
    detail["rmsse"] = rmsse
    detail["weighted_rmsse"] = weighted_rmsse

    numeric_cols = ["actual_total", "pred_total", "signed_bias", "mse", "rmsse", "weighted_rmsse"]
    if detail[numeric_cols].isna().any().any():
        raise ValueError("WRMSSE detail contains NaN values.")
    if np.any(pred < -1e-12):
        raise ValueError("Prediction array contains negative values.")

    score = float(detail["weighted_rmsse"].sum())
    return score, detail


def score_column_subset(
    actual_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    metric_info: pd.DataFrame,
    columns: list[pd.Timestamp],
) -> tuple[float, pd.DataFrame]:
    return wrmsse_score(actual_wide.loc[:, columns], pred_wide.loc[:, columns], metric_info)


def build_row_bucket_summary(
    detail: pd.DataFrame,
    bucket_col: str,
    bucket_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for bucket_label, subset in detail.groupby(bucket_col, dropna=False):
        if subset.empty:
            continue
        rows.append(
            {
                "bucket_name": bucket_name,
                "bucket_label": bucket_label,
                "sku_count": int(len(subset)),
                "weight_sum": float(subset["weight"].sum()),
                "actual_total": float(subset["actual_total"].sum()),
                "pred_total": float(subset["pred_total"].sum()),
                "signed_bias": float(subset["signed_bias"].sum()),
                "bias_ratio": float(
                    subset["pred_total"].sum() / subset["actual_total"].sum()
                    if abs(subset["actual_total"].sum()) > EPS
                    else np.nan
                ),
                "segment_wrmsse": float(subset["weighted_rmsse"].sum()),
                "mean_rmsse": float(subset["rmsse"].mean()),
                "mean_weighted_rmsse": float(subset["weighted_rmsse"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["bucket_name", "bucket_label"]).reset_index(drop=True)
    return out


def build_column_group_summary(
    actual_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    metric_info: pd.DataFrame,
    groups: dict[str, list[pd.Timestamp]],
    family_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name, columns in groups.items():
        score, detail = score_column_subset(actual_wide, pred_wide, metric_info, columns)
        actual_total = float(actual_wide.loc[:, columns].to_numpy(dtype=float).sum())
        pred_total = float(pred_wide.loc[:, columns].to_numpy(dtype=float).sum())
        rows.append(
            {
                "family_name": family_name,
                "group_name": group_name,
                "n_days": len(columns),
                "actual_total": actual_total,
                "pred_total": pred_total,
                "signed_bias": pred_total - actual_total,
                "bias_ratio": float(pred_total / actual_total if abs(actual_total) > EPS else np.nan),
                "segment_wrmsse": float(score),
                "weight_sum": float(detail["weight"].sum()),
                "weighted_contribution": float(detail["weighted_rmsse"].sum()),
            }
        )

    return pd.DataFrame(rows)


def build_weekday_summary(
    actual_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    metric_info: pd.DataFrame,
) -> pd.DataFrame:
    weekday_name_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    groups: dict[str, list[pd.Timestamp]] = {}
    for dt in actual_wide.columns:
        groups.setdefault(weekday_name_map[pd.Timestamp(dt).dayofweek], []).append(dt)
    return build_column_group_summary(
        actual_wide=actual_wide,
        pred_wide=pred_wide,
        metric_info=metric_info,
        groups=groups,
        family_name="weekday",
    )


def build_horizon_summary(
    actual_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    metric_info: pd.DataFrame,
) -> pd.DataFrame:
    cols = list(actual_wide.columns)
    groups = {
        "first28": cols[:28],
        "second28": cols[28:56],
        "H1-7": cols[0:7],
        "H8-14": cols[7:14],
        "H15-28": cols[14:28],
        "H29-42": cols[28:42],
        "H43-56": cols[42:56],
    }
    return build_column_group_summary(
        actual_wide=actual_wide,
        pred_wide=pred_wide,
        metric_info=metric_info,
        groups=groups,
        family_name="horizon",
    )


def assign_profit_bucket(rank_value: float) -> str:
    if rank_value <= 100:
        return "rank_001_100"
    if rank_value <= 500:
        return "rank_101_500"
    if rank_value <= 1000:
        return "rank_501_1000"
    if rank_value <= 2000:
        return "rank_1001_2000"
    return "rank_2001_plus"


def assign_activity_bucket(active_days: float) -> str:
    if active_days <= 3:
        return "active_days <= 3"
    if active_days <= 10:
        return "active_days 4-10"
    if active_days <= 50:
        return "active_days 11-50"
    return "active_days > 50"


def assign_recency_bucket(days_since_last_sale: float) -> str:
    if days_since_last_sale <= 14:
        return "<=14"
    if days_since_last_sale <= 56:
        return "15-56"
    if days_since_last_sale <= 180:
        return "57-180"
    return ">180"


def build_prediction(
    model_cfg: dict,
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    train_end,
    forecast_start,
    horizon: int,
) -> pd.DataFrame:
    forecast_dates = pd.date_range(pd.Timestamp(forecast_start), periods=horizon, freq="D")
    itemcodes = sorted(panel["ItemCode"].unique())

    if model_cfg.get("type") == "zero":
        pred = make_zero_prediction(itemcodes=itemcodes, forecast_dates=forecast_dates)
    else:
        if len(model_cfg["windows"]) != len(model_cfg["weights"]):
            raise ValueError(f"Invalid windows/weights for {model_cfg['model']}")

        pred = make_recent_blend_prediction(
            panel=panel,
            train_end=train_end,
            forecast_start=forecast_start,
            horizon=horizon,
            target_col=PRED_TARGET,
            windows=tuple(model_cfg["windows"]),
            weights=tuple(model_cfg["weights"]),
        )
        pred = postprocess_prediction(
            pred=pred,
            panel=panel,
            sku_activity=sku_activity,
            train_end=train_end,
            target_col=PRED_TARGET,
            sunday_factor=0.0,
            apply_cap=True,
        )
        pred = pred.reindex(index=itemcodes, columns=forecast_dates).fillna(0.0)

    pred = pred.sort_index().sort_index(axis=1).astype(float).fillna(0.0).clip(lower=0.0)

    if pred.shape != (len(itemcodes), horizon):
        raise ValueError(f"Unexpected prediction shape for {model_cfg['model']}: {pred.shape}")
    if pred.isna().any().any():
        raise ValueError(f"Prediction matrix contains NaN for {model_cfg['model']}")
    if (pred.to_numpy(dtype=float) < -1e-12).any():
        raise ValueError(f"Prediction matrix contains negative values for {model_cfg['model']}")

    return pred


def evaluate_fold_model(
    fold: FoldConfig,
    model_cfg: dict,
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    output_dir: Path,
) -> dict:
    metric_info = compute_sku_metric_info(panel=panel, train_end=fold.train_end, target_col=METRIC_TARGET)
    actual_wide = make_actual_matrix(
        panel=panel,
        start_date=fold.valid_start,
        horizon=fold.horizon,
        target_col=METRIC_TARGET,
    )
    pred_wide = build_prediction(
        model_cfg=model_cfg,
        panel=panel,
        sku_activity=sku_activity,
        train_end=fold.train_end,
        forecast_start=fold.valid_start,
        horizon=fold.horizon,
    )

    pred_wide = pred_wide.reindex(index=actual_wide.index, columns=actual_wide.columns).fillna(0.0)

    if actual_wide.shape != pred_wide.shape:
        raise ValueError(
            f"Shape mismatch on fold {fold.fold}, model {model_cfg['model']}: {actual_wide.shape} vs {pred_wide.shape}"
        )

    overall_score, detail = wrmsse_score(actual_wide, pred_wide, metric_info)

    detail["active_days"] = detail["active_days"].astype(int)
    detail["profit_bucket"] = detail["profit_rank"].apply(assign_profit_bucket)
    detail["activity_bucket"] = detail["active_days"].apply(assign_activity_bucket)
    detail["recency_bucket"] = detail["days_since_last_sale"].apply(assign_recency_bucket)

    if detail.isna().any().any():
        raise ValueError(f"Detail table contains NaN for fold {fold.fold}, model {model_cfg['model']}")

    first28_score, first28_detail = wrmsse_score(actual_wide.iloc[:, :28], pred_wide.iloc[:, :28], metric_info)
    second28_score, second28_detail = wrmsse_score(actual_wide.iloc[:, 28:56], pred_wide.iloc[:, 28:56], metric_info)

    horizon_summary = build_horizon_summary(actual_wide, pred_wide, metric_info)
    weekday_summary = build_weekday_summary(actual_wide, pred_wide, metric_info)

    profit_summary = build_row_bucket_summary(detail, "profit_bucket", "profit_rank")
    activity_summary = build_row_bucket_summary(detail, "activity_bucket", "activity")
    recency_summary = build_row_bucket_summary(detail, "recency_bucket", "recency")

    for df, family_name in [
        (profit_summary, "profit_rank"),
        (activity_summary, "activity"),
        (recency_summary, "recency"),
    ]:
        if not df.empty:
            df["family_name"] = family_name
            df["model"] = model_cfg["model"]
            df["fold"] = fold.fold

    for df in [horizon_summary, weekday_summary]:
        if not df.empty:
            df["model"] = model_cfg["model"]
            df["fold"] = fold.fold

    overall_record = {
        "fold": fold.fold,
        "model": model_cfg["model"],
        "wrmsse": float(overall_score),
        "first28_wrmsse": float(first28_score),
        "second28_wrmsse": float(second28_score),
    }

    sku_detail = detail.copy()
    sku_detail["fold"] = fold.fold
    sku_detail["model"] = model_cfg["model"]
    sku_detail = sku_detail[
        [
            "fold",
            "model",
            "ItemCode",
            "profit_rank",
            "weight",
            "scale",
            "actual_total",
            "pred_total",
            "signed_bias",
            "rmsse",
            "weighted_rmsse",
            "active_days",
            "days_since_last_sale",
            "total_profit",
            "positive_profit",
            "zero_scale_flag",
        ]
    ].sort_values("weighted_rmsse", ascending=False)

    sku_path = output_dir / f"sku_contribution_{fold.fold}_{model_cfg['model']}.csv"
    sku_detail.to_csv(sku_path, index=False)

    return {
        "overall": overall_record,
        "horizon": horizon_summary,
        "weekday": weekday_summary,
        "profit": profit_summary,
        "activity": activity_summary,
        "recency": recency_summary,
        "sku_detail": sku_detail,
        "first28_detail": first28_detail,
        "second28_detail": second28_detail,
    }


def build_segment_winner_table(
    all_segment_tables: list[pd.DataFrame],
    current_best_model: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    combined = pd.concat(all_segment_tables, ignore_index=True)

    for (fold, family_name, group_name), group in combined.groupby(["fold", "family_name", "group_name"], dropna=False):
        best_row = group.sort_values(["segment_wrmsse", "model"], ascending=[True, True]).iloc[0]
        current_row = group.loc[group["model"] == current_best_model]
        current_score = float(current_row.iloc[0]["segment_wrmsse"]) if not current_row.empty else np.nan
        best_score = float(best_row["segment_wrmsse"])
        rows.append(
            {
                "fold": fold,
                "family_name": family_name,
                "group_name": group_name,
                "best_model": best_row["model"],
                "best_score": best_score,
                "current_best_model": current_best_model,
                "current_best_model_score": current_score,
                "improvement_abs": float(current_score - best_score) if np.isfinite(current_score) else np.nan,
                "improvement_pct": float((current_score - best_score) / current_score)
                if np.isfinite(current_score) and abs(current_score) > EPS
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["fold", "family_name", "group_name"]).reset_index(drop=True)


def top_n_models_for_fold(overall_df: pd.DataFrame, fold_name: str, n: int = 10) -> pd.DataFrame:
    subset = overall_df.loc[overall_df["fold"] == fold_name].sort_values("wrmsse")
    return subset.head(n).reset_index(drop=True)


def build_recommendations(
    segment_winner_table: pd.DataFrame,
    overall_df: pd.DataFrame,
    fold_name: str,
    current_best_model: str,
) -> list[str]:
    fold_table = segment_winner_table.loc[segment_winner_table["fold"] == fold_name].copy()
    fold_overall = overall_df.loc[overall_df["fold"] == fold_name].sort_values("wrmsse")
    top_model = fold_overall.iloc[0]["model"] if not fold_overall.empty else current_best_model

    recommendations: list[str] = []

    row_focus = fold_table.loc[fold_table["family_name"].isin(["profit_rank", "activity", "recency"])]
    if not row_focus.empty:
        best_gap = row_focus.sort_values("improvement_abs", ascending=False).iloc[0]
        recommendations.append(
            f"Segment gap: {best_gap['family_name']} / {best_gap['group_name']} prefers {best_gap['best_model']} over {current_best_model} by {best_gap['improvement_abs']:.6f}. Build a selector that routes those SKUs to the better specialist model."
        )

    horizon_focus = fold_table.loc[fold_table["family_name"] == "horizon"]
    if not horizon_focus.empty:
        half_rows = horizon_focus.loc[horizon_focus["group_name"].isin(["first28", "second28"])]
        if not half_rows.empty:
            best_horizon = half_rows.sort_values("improvement_abs", ascending=False).iloc[0]
            recommendations.append(
                f"Horizon split signal: {best_horizon['group_name']} is better served by {best_horizon['best_model']} than {current_best_model}. Test a first-half / second-half blend instead of a single static window."
            )

    top_rank_row = fold_table.loc[(fold_table["family_name"] == "profit_rank") & (fold_table["group_name"] == "rank_001_100")]
    if not top_rank_row.empty:
        winner = top_rank_row.iloc[0]
        recommendations.append(
            f"High-profit routing: rank_001_100 currently favors {winner['best_model']}. Give the top-profit tail a different window mix than the long tail and keep Sunday-zero plus cap only where it helps."
        )

    if not recommendations:
        recommendations = [
            f"Use {top_model} as the main anchor and test a small segment-aware selector around the largest bucket gaps.",
            "Check whether the inactivity rule is too aggressive for rank_101_500 and rank_501_1000 SKUs.",
            "Try a first28 / second28 blend before tuning any single window weight.",
        ]

    return recommendations[:3]


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_folds = args.folds or ["recent_2025"]
    fold_configs = get_fold_configs(selected_folds)

    log_section("Load and rebuild daily panel")
    train, sample = load_raw_data()
    train_clean = clean_train(train)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    if sample["id"].duplicated().any():
        raise ValueError("sample_submission contains duplicate ids.")

    overall_records: list[dict] = []
    horizon_records: list[pd.DataFrame] = []
    weekday_records: list[pd.DataFrame] = []
    profit_records: list[pd.DataFrame] = []
    activity_records: list[pd.DataFrame] = []
    recency_records: list[pd.DataFrame] = []
    sku_tables: list[pd.DataFrame] = []
    segment_tables: list[pd.DataFrame] = []

    for fold in fold_configs:
        log_section(
            f"Fold: {fold.fold} | train_end={fold.train_end.date()} | valid_start={fold.valid_start.date()}"
        )

        for model_cfg in MODEL_CONFIGS:
            print(f"Evaluating {fold.fold} :: {model_cfg['model']}")
            result = evaluate_fold_model(
                fold=fold,
                model_cfg=model_cfg,
                panel=daily_panel,
                sku_activity=sku_activity,
                output_dir=output_dir,
            )

            overall_records.append(result["overall"])
            horizon_records.append(result["horizon"].copy())
            weekday_records.append(result["weekday"].copy())
            profit_records.append(result["profit"].copy())
            activity_records.append(result["activity"].copy())
            recency_records.append(result["recency"].copy())
            sku_tables.append(result["sku_detail"].copy())

            segment_tables.extend([result["profit"], result["activity"], result["recency"], result["horizon"], result["weekday"]])

    overall_df = pd.DataFrame(overall_records).sort_values(["fold", "wrmsse", "model"]).reset_index(drop=True)
    horizon_df = pd.concat(horizon_records, ignore_index=True).sort_values(
        ["fold", "family_name", "group_name", "segment_wrmsse"]
    )
    weekday_df = pd.concat(weekday_records, ignore_index=True).sort_values(
        ["fold", "group_name", "segment_wrmsse"]
    )
    profit_df = pd.concat(profit_records, ignore_index=True).sort_values(
        ["fold", "bucket_label", "segment_wrmsse"]
    )
    activity_df = pd.concat(activity_records, ignore_index=True).sort_values(
        ["fold", "bucket_label", "segment_wrmsse"]
    )
    recency_df = pd.concat(recency_records, ignore_index=True).sort_values(
        ["fold", "bucket_label", "segment_wrmsse"]
    )
    sku_df = pd.concat(sku_tables, ignore_index=True).sort_values(
        ["fold", "model", "weighted_rmsse"], ascending=[True, True, False]
    )

    overall_df.to_csv(output_dir / "overall_model_scores.csv", index=False)
    horizon_df.to_csv(output_dir / "horizon_split_scores.csv", index=False)
    profit_df.to_csv(output_dir / "profit_rank_bucket_scores.csv", index=False)
    activity_df.to_csv(output_dir / "activity_bucket_scores.csv", index=False)
    recency_df.to_csv(output_dir / "recency_bucket_scores.csv", index=False)
    weekday_df.to_csv(output_dir / "weekday_bias_summary.csv", index=False)
    sku_df.to_csv(output_dir / "sku_contribution_top.csv", index=False)

    segment_winner_df = build_segment_winner_table(segment_tables, current_best_model=args.current_best_model)
    segment_winner_df.to_csv(output_dir / "segment_winner_table.csv", index=False)

    log_section("Final console summary")
    recent_top10 = top_n_models_for_fold(overall_df, "recent_2025", n=10)
    print("Top 10 models overall on recent_2025:")
    print(recent_top10[["model", "wrmsse", "first28_wrmsse", "second28_wrmsse"]].to_string(index=False))

    recent_horizon = horizon_df.loc[horizon_df["fold"] == "recent_2025"]
    first28_winner = recent_horizon.loc[recent_horizon["group_name"] == "first28"].sort_values("segment_wrmsse").iloc[0]
    second28_winner = recent_horizon.loc[recent_horizon["group_name"] == "second28"].sort_values("segment_wrmsse").iloc[0]
    print(f"Best model for first28: {first28_winner['model']} | score={first28_winner['segment_wrmsse']:.6f}")
    print(f"Best model for second28: {second28_winner['model']} | score={second28_winner['segment_wrmsse']:.6f}")

    recent_profit = profit_df.loc[profit_df["fold"] == "recent_2025"]
    best_rank_001_100 = recent_profit.loc[recent_profit["bucket_label"] == "rank_001_100"].sort_values("segment_wrmsse").iloc[0]
    best_rank_101_500 = recent_profit.loc[recent_profit["bucket_label"] == "rank_101_500"].sort_values("segment_wrmsse").iloc[0]
    print(f"Best model for rank_001_100: {best_rank_001_100['model']} | score={best_rank_001_100['segment_wrmsse']:.6f}")
    print(f"Best model for rank_101_500: {best_rank_101_500['model']} | score={best_rank_101_500['segment_wrmsse']:.6f}")

    current_best_rows = sku_df.loc[
        (sku_df["fold"] == "recent_2025") & (sku_df["model"] == args.current_best_model)
    ].copy()
    print(f"Top 20 SKU contribution under current best ({args.current_best_model}):")
    print(
        current_best_rows.head(20)[
            [
                "ItemCode",
                "profit_rank",
                "weight",
                "scale",
                "actual_total",
                "pred_total",
                "signed_bias",
                "rmsse",
                "weighted_rmsse",
                "active_days",
                "days_since_last_sale",
                "total_profit",
            ]
        ].to_string(index=False)
    )

    recommendations = build_recommendations(
        segment_winner_table=segment_winner_df,
        overall_df=overall_df,
        fold_name="recent_2025",
        current_best_model=args.current_best_model,
    )
    print("Recommended candidate ideas:")
    for idx, rec in enumerate(recommendations, start=1):
        print(f"{idx}. {rec}")

    print("\nSaved outputs to:", output_dir)
    print("- overall_model_scores.csv")
    print("- horizon_split_scores.csv")
    print("- profit_rank_bucket_scores.csv")
    print("- activity_bucket_scores.csv")
    print("- recency_bucket_scores.csv")
    print("- weekday_bias_summary.csv")
    print("- sku_contribution_top.csv")
    print("- segment_winner_table.csv")


if __name__ == "__main__":
    main()