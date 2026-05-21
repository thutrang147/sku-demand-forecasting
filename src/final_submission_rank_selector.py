from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_error_decomposition import (
    FoldConfig,
    assign_profit_bucket,
    build_prediction,
    compute_sku_metric_info,
    make_actual_matrix,
    wrmsse_score,
)
from final_reproduce import (
    ROOT,
    clean_train,
    load_raw_data,
    make_daily_panel,
    make_kaggle_submission,
    make_recent_blend_prediction,
    make_sku_activity,
    postprocess_prediction,
)


FINAL_TRAIN_END = pd.Timestamp("2025-09-05")
FINAL_FORECAST_START = pd.Timestamp("2025-09-06")
HORIZON = 56
TARGET_FOR_PRED = "y_net_clip"

DEFAULT_BASE_SUBMISSION = (
    ROOT
    / "outputs"
    / "submissions"
    / "submission_PUBLIC_ANCHOR_recent14_18_21_w20_45_35_sunday0__eval_recent18_21_28_56_private_safe.csv"
)
DEFAULT_OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "submissions"
    / "submission_FINAL_public_anchor_eval_balanced_rank_selector.csv"
)

CURRENT_BEST_MODEL = "recent14_18_21_w20_45_35_sunday0"
CURRENT_BEST_WINDOWS = (14, 18, 21)
CURRENT_BEST_WEIGHTS = (0.20, 0.45, 0.35)

MODEL_28 = {
    "model": "recent28_sunday0",
    "windows": (28,),
    "weights": (1.0,),
}
MODEL_352 = {
    "model": "recent21_28_56_w35_45_20_sunday0",
    "windows": (21, 28, 56),
    "weights": (0.35, 0.45, 0.20),
}
MODEL_255 = {
    "model": "recent21_28_56_w25_50_25_sunday0",
    "windows": (21, 28, 56),
    "weights": (0.25, 0.50, 0.25),
}

BACKTEST_FOLDS = [
    FoldConfig(
        fold="recent_2025",
        train_end=pd.Timestamp("2025-07-11"),
        valid_start=pd.Timestamp("2025-07-12"),
        horizon=56,
    ),
    FoldConfig(
        fold="seasonal_2024",
        train_end=pd.Timestamp("2024-09-05"),
        valid_start=pd.Timestamp("2024-09-06"),
        horizon=56,
    ),
]


def log_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final rank-selector submission for HBAAC.")
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=DEFAULT_BASE_SUBMISSION,
        help="Path to the current best base submission whose validation rows will be preserved.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the final submission CSV.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the optional offline backtest diagnostics.",
    )
    return parser.parse_args()


def load_base_submission(base_path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base submission not found at {base_path}. Pass a valid path with --base-submission."
        )

    base = pd.read_csv(base_path)
    expected_cols = ["id"] + [f"F{i}" for i in range(1, 29)]
    if list(base.columns) != expected_cols:
        raise ValueError(f"Unexpected base submission columns: {base.columns.tolist()}")

    if set(base["id"]) != set(sample["id"]):
        raise ValueError("Base submission ids do not match sample_submission ids.")

    base = sample[["id"]].merge(base, on="id", how="left", validate="one_to_one")
    if base.isna().any().any():
        raise ValueError("Base submission reindexing introduced NaN values.")

    return base


def make_model_submission(
    model_name: str,
    windows: tuple[int, ...],
    weights: tuple[float, ...],
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    pred_56 = make_recent_blend_prediction(
        panel=panel,
        train_end=FINAL_TRAIN_END,
        forecast_start=FINAL_FORECAST_START,
        horizon=HORIZON,
        target_col=TARGET_FOR_PRED,
        windows=windows,
        weights=weights,
    )
    pred_56 = postprocess_prediction(
        pred=pred_56,
        panel=panel,
        sku_activity=sku_activity,
        train_end=FINAL_TRAIN_END,
        target_col=TARGET_FOR_PRED,
        sunday_factor=0.0,
        apply_cap=True,
    )
    return make_kaggle_submission(pred_56=pred_56, sample=sample, output_path=None)


def make_zero_submission(sample: pd.DataFrame, itemcodes: list[str]) -> pd.DataFrame:
    zero_pred = pd.DataFrame(0.0, index=itemcodes, columns=pd.date_range(FINAL_FORECAST_START, periods=HORIZON, freq="D"))
    zero_pred.index.name = None
    return make_kaggle_submission(pred_56=zero_pred, sample=sample, output_path=None)


def submission_to_eval_matrix(sub: pd.DataFrame) -> pd.DataFrame:
    f_cols = [f"F{i}" for i in range(1, 29)]
    eval_rows = sub[sub["id"].str.endswith("_evaluation")].copy()
    eval_rows["ItemCode"] = eval_rows["id"].str.replace("_evaluation", "", regex=False)
    eval_rows = eval_rows.set_index("ItemCode")[f_cols].astype(float)
    eval_rows.columns = pd.RangeIndex(start=0, stop=28, step=1)
    return eval_rows


def set_eval_rows_from_model(
    base: pd.DataFrame,
    model_sub: pd.DataFrame,
    itemcode_to_bucket: pd.DataFrame,
) -> pd.DataFrame:
    f_cols = [f"F{i}" for i in range(1, 29)]
    out = base.copy()

    base_eval_mask = out["id"].str.endswith("_evaluation")
    model_eval = model_sub[model_sub["id"].str.endswith("_evaluation")].copy().set_index("id")

    if not model_eval.index.is_unique:
        raise ValueError("Model evaluation rows are not unique.")

    base_eval_ids = out.loc[base_eval_mask, "id"]
    if not base_eval_ids.isin(model_eval.index).all():
        missing = base_eval_ids[~base_eval_ids.isin(model_eval.index)].tolist()
        raise ValueError(f"Model submission is missing evaluation ids: {missing[:5]}")

    out.loc[base_eval_mask, f_cols] = model_eval.loc[base_eval_ids, f_cols].to_numpy(dtype=float)
    out.loc[base_eval_mask, ["F2", "F9", "F16", "F23"]] = 0.0
    out[f_cols] = out[f_cols].fillna(0.0).clip(lower=0.0)
    return out


def build_selector_submission(
    base_submission: pd.DataFrame,
    sample: pd.DataFrame,
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    item_meta = sku_activity[["ItemCode", "profit_rank"]].copy()
    item_meta["rank_bucket"] = item_meta["profit_rank"].apply(assign_profit_bucket)

    model_28_sub = make_model_submission(
        model_name=MODEL_28["model"],
        windows=MODEL_28["windows"],
        weights=MODEL_28["weights"],
        panel=panel,
        sku_activity=sku_activity,
        sample=sample,
    )
    model_352_sub = make_model_submission(
        model_name=MODEL_352["model"],
        windows=MODEL_352["windows"],
        weights=MODEL_352["weights"],
        panel=panel,
        sku_activity=sku_activity,
        sample=sample,
    )
    model_255_sub = make_model_submission(
        model_name=MODEL_255["model"],
        windows=MODEL_255["windows"],
        weights=MODEL_255["weights"],
        panel=panel,
        sku_activity=sku_activity,
        sample=sample,
    )
    zero_sub = make_zero_submission(sample=sample, itemcodes=sorted(panel["ItemCode"].unique()))

    selection = base_submission.copy()
    f_cols = [f"F{i}" for i in range(1, 29)]

    eval_rows = selection["id"].str.endswith("_evaluation")
    eval_ids = selection.loc[eval_rows, "id"].tolist()
    eval_itemcodes = pd.Index([x.replace("_evaluation", "") for x in eval_ids])

    bucket_lookup = item_meta.set_index("ItemCode").reindex(eval_itemcodes)
    if bucket_lookup.isna().any().any():
        raise ValueError("Missing profit_rank information for some evaluation items.")

    model_eval_map = {
        MODEL_28["model"]: model_28_sub,
        MODEL_352["model"]: model_352_sub,
        MODEL_255["model"]: model_255_sub,
        "zero": zero_sub,
    }

    for bucket_label in ["rank_001_100", "rank_101_500", "rank_501_1000", "rank_1001_2000", "rank_2001_plus"]:
        pass

    # Route evaluation rows by profit rank.
    bucket_masks = {
        "rank_001_100": bucket_lookup["profit_rank"] <= 100,
        "rank_101_500": (bucket_lookup["profit_rank"] >= 101) & (bucket_lookup["profit_rank"] <= 500),
        "rank_501_1000": (bucket_lookup["profit_rank"] >= 501) & (bucket_lookup["profit_rank"] <= 1000),
        "rank_1001_2000": (bucket_lookup["profit_rank"] >= 1001) & (bucket_lookup["profit_rank"] <= 2000),
        "rank_2001_plus": bucket_lookup["profit_rank"] > 2000,
    }

    selection_eval = selection.loc[eval_rows, ["id"] + f_cols].copy().set_index("id")

    eval_model_source = pd.Series(index=eval_ids, dtype=object)
    eval_model_source.loc[[x + "_evaluation" for x in bucket_masks["rank_001_100"].index[bucket_masks["rank_001_100"]]]] = MODEL_28["model"]
    eval_model_source.loc[[x + "_evaluation" for x in bucket_masks["rank_101_500"].index[bucket_masks["rank_101_500"]]]] = MODEL_352["model"]
    eval_model_source.loc[[x + "_evaluation" for x in bucket_masks["rank_501_1000"].index[bucket_masks["rank_501_1000"]]]] = MODEL_255["model"]
    eval_model_source.loc[[x + "_evaluation" for x in bucket_masks["rank_1001_2000"].index[bucket_masks["rank_1001_2000"]]]] = "zero"
    eval_model_source.loc[[x + "_evaluation" for x in bucket_masks["rank_2001_plus"].index[bucket_masks["rank_2001_plus"]]]] = "zero"

    for model_name, model_sub in model_eval_map.items():
        eval_part = model_sub[model_sub["id"].str.endswith("_evaluation")].set_index("id")
        if eval_part.isna().any().any():
            raise ValueError(f"Model submission {model_name} contains NaN values.")
        if (eval_part[f_cols].to_numpy(dtype=float) < -1e-12).any():
            raise ValueError(f"Model submission {model_name} contains negative values.")

    for bucket_label, mask in bucket_masks.items():
        ids = bucket_lookup.index[mask].tolist()
        if not ids:
            continue
        chosen_model = eval_model_source.loc[[x + "_evaluation" for x in ids]].iloc[0]
        chosen_sub = model_eval_map[chosen_model].set_index("id")
        selection_eval.loc[[f"{item}_evaluation" for item in ids], f_cols] = chosen_sub.loc[
            [f"{item}_evaluation" for item in ids], f_cols
        ].to_numpy(dtype=float)

    selection.loc[eval_rows, f_cols] = selection_eval.reset_index(drop=True).to_numpy(dtype=float)
    selection.loc[eval_rows, ["F2", "F9", "F16", "F23"]] = 0.0
    selection[f_cols] = selection[f_cols].fillna(0.0).clip(lower=0.0)

    outputs = {
        MODEL_28["model"]: model_28_sub,
        MODEL_352["model"]: model_352_sub,
        MODEL_255["model"]: model_255_sub,
        "zero": zero_sub,
    }
    return selection, outputs


def qa_submission(sub: pd.DataFrame, sample: pd.DataFrame, base_submission: pd.DataFrame) -> None:
    f_cols = [f"F{i}" for i in range(1, 29)]
    validation_rows = sub["id"].str.endswith("_validation")
    evaluation_rows = sub["id"].str.endswith("_evaluation")
    sunday_cols = ["F2", "F9", "F16", "F23"]

    summary = {
        "rows": len(sub),
        "columns": sub.shape[1],
        "unique_ids": sub["id"].nunique(),
        "missing": int(sub[f_cols].isna().sum().sum()),
        "negative": int((sub[f_cols] < 0).sum().sum()),
        "total": float(sub[f_cols].sum().sum()),
        "validation_total": float(sub.loc[validation_rows, f_cols].sum().sum()),
        "evaluation_total": float(sub.loc[evaluation_rows, f_cols].sum().sum()),
        "validation_sunday_total": float(sub.loc[validation_rows, sunday_cols].sum().sum()),
        "evaluation_sunday_total": float(sub.loc[evaluation_rows, sunday_cols].sum().sum()),
    }

    print("\nQA SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    base_validation = base_submission.loc[validation_rows, f_cols].to_numpy(dtype=float)
    sub_validation = sub.loc[validation_rows, f_cols].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(base_validation - sub_validation)))
    print(f"validation max_abs_diff vs base_submission: {max_abs_diff}")

    assert summary["rows"] == 31944
    assert summary["columns"] == 29
    assert summary["unique_ids"] == 31944
    assert summary["missing"] == 0
    assert summary["negative"] == 0
    assert max_abs_diff == 0.0
    assert abs(summary["evaluation_sunday_total"]) < 1e-9


def print_evaluation_bucket_totals(sub: pd.DataFrame, sku_activity: pd.DataFrame) -> None:
    f_cols = [f"F{i}" for i in range(1, 29)]
    eval_rows = sub[sub["id"].str.endswith("_evaluation")].copy()
    eval_rows["ItemCode"] = eval_rows["id"].str.replace("_evaluation", "", regex=False)

    meta = sku_activity[["ItemCode", "profit_rank"]].copy()
    meta["rank_bucket"] = meta["profit_rank"].apply(assign_profit_bucket)
    eval_rows = eval_rows.merge(meta[["ItemCode", "rank_bucket"]], on="ItemCode", how="left")

    print("\nEvaluation total by rank bucket")
    bucket_summary = (
        eval_rows.groupby("rank_bucket", as_index=False)[f_cols]
        .sum()
        .assign(total=lambda df: df[f_cols].sum(axis=1))[["rank_bucket", "total"]]
        .sort_values("rank_bucket")
    )
    print(bucket_summary.to_string(index=False))


def run_backtest(panel: pd.DataFrame, sku_activity: pd.DataFrame) -> None:
    log_section("Offline backtest")

    current_score_cfg = {
        "model": CURRENT_BEST_MODEL,
        "windows": CURRENT_BEST_WINDOWS,
        "weights": CURRENT_BEST_WEIGHTS,
    }
    selector_cfgs = [MODEL_28, MODEL_352, MODEL_255]

    for fold in BACKTEST_FOLDS:
        print(f"\nFold: {fold.fold} | train_end={fold.train_end.date()} | valid_start={fold.valid_start.date()}")
        metric_info = compute_sku_metric_info(panel=panel, train_end=fold.train_end, target_col="y_net")
        actual_wide = make_actual_matrix(panel=panel, start_date=fold.valid_start, horizon=fold.horizon, target_col="y_net")

        current_pred = build_prediction(
            model_cfg=current_score_cfg,
            panel=panel,
            sku_activity=sku_activity,
            train_end=fold.train_end,
            forecast_start=fold.valid_start,
            horizon=fold.horizon,
        )
        current_score, current_detail = wrmsse_score(actual_wide, current_pred, metric_info)

        selector_parts = []
        for model_cfg in selector_cfgs:
            pred = build_prediction(
                model_cfg=model_cfg,
                panel=panel,
                sku_activity=sku_activity,
                train_end=fold.train_end,
                forecast_start=fold.valid_start,
                horizon=fold.horizon,
            )
            selector_parts.append((model_cfg["model"], pred))
        zero_pred = pd.DataFrame(0.0, index=actual_wide.index, columns=actual_wide.columns)

        item_meta = sku_activity[["ItemCode", "profit_rank"]].copy()
        item_meta["rank_bucket"] = item_meta["profit_rank"].apply(assign_profit_bucket)
        item_meta = item_meta.set_index("ItemCode")

        selector_pred = pd.DataFrame(index=actual_wide.index, columns=actual_wide.columns, dtype=float)
        for item_code in selector_pred.index:
            rank = int(item_meta.loc[item_code, "profit_rank"])
            if rank <= 100:
                source = selector_parts[0][1]
            elif rank <= 500:
                source = selector_parts[1][1]
            elif rank <= 1000:
                source = selector_parts[2][1]
            else:
                source = zero_pred
            selector_pred.loc[item_code] = source.loc[item_code].to_numpy(dtype=float)

        selector_score, selector_detail = wrmsse_score(actual_wide, selector_pred, metric_info)

        print(f"current_score: {current_score:.6f}")
        print(f"selector_score: {selector_score:.6f}")
        print(f"improvement: {current_score - selector_score:.6f}")

        selector_detail["rank_bucket"] = selector_detail["profit_rank"].apply(assign_profit_bucket)
        bucket_contrib = (
            selector_detail.groupby("rank_bucket", as_index=False)
            .agg(
                weight_sum=("weight", "sum"),
                weighted_rmsse=("weighted_rmsse", "sum"),
                rmsse=("rmsse", "mean"),
            )
            .sort_values("rank_bucket")
        )
        print("profit bucket contributions:")
        print(bucket_contrib.to_string(index=False))


def main() -> None:
    args = parse_args()

    log_section("Load raw data")
    train, sample = load_raw_data()
    train_clean = clean_train(train)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    base_submission = load_base_submission(args.base_submission, sample)

    log_section("Build final selector submission")
    selection_sub, model_subs = build_selector_submission(
        base_submission=base_submission,
        sample=sample,
        panel=daily_panel,
        sku_activity=sku_activity,
    )

    # Keep a record of the model submissions in memory only; no extra files are written.
    _ = model_subs

    print_evaluation_bucket_totals(selection_sub, sku_activity)
    qa_submission(selection_sub, sample, base_submission)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    selection_sub.to_csv(args.output_path, index=False)
    print(f"\nSaved final submission: {args.output_path}")

    if not args.skip_backtest:
        run_backtest(panel=daily_panel, sku_activity=sku_activity)


if __name__ == "__main__":
    main()