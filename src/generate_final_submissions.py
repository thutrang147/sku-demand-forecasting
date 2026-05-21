"""Generate the final reproducible Kaggle submissions.

Run from project root:
    python src/generate_final_submissions.py

This script reproduces the two final selected submissions:
- submission_anchor_longtail_v16_lottery.csv
- submission_anchor_longtail_v14_controlled_allin.csv

It does not auto-submit and it does not depend on notebook state.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_reproduce import (  # noqa: E402
    load_raw_data,
    clean_train,
    make_daily_panel,
    make_sku_activity,
    make_recent_blend_prediction,
    postprocess_prediction,
    make_kaggle_submission,
    FINAL_TRAIN_END,
    FINAL_FORECAST_START,
    HORIZON,
    TARGET_FOR_PRED,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_SUB_DIR = PROJECT_ROOT / "outputs" / "final_submissions"
DIAG_DIR = PROJECT_ROOT / "outputs" / "diagnostics"

V16_NAME = "anchor_longtail_v16_lottery"
V14_NAME = "anchor_longtail_v14_controlled_allin"
V16_FILE = "submission_anchor_longtail_v16_lottery.csv"
V14_FILE = "submission_anchor_longtail_v14_controlled_allin.csv"

PUBLIC_SCORE_REFERENCE = {
    V16_NAME: 0.48778,
    V14_NAME: 0.48883,
}


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SUB_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_panel_path = PROCESSED_DIR / "daily_panel.parquet"
    sku_activity_path = PROCESSED_DIR / "sku_activity.parquet"

    if daily_panel_path.exists() and sku_activity_path.exists():
        print("Loading cached processed data...")
        daily_panel = pd.read_parquet(daily_panel_path)
        sku_activity = pd.read_parquet(sku_activity_path)
        return daily_panel, sku_activity

    print("Building processed data from raw train...")
    train, _sample = load_raw_data()
    train_clean = clean_train(train)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    daily_panel.to_parquet(daily_panel_path)
    sku_activity.to_parquet(sku_activity_path)
    return daily_panel, sku_activity


def build_recent_prediction(
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    windows: tuple[int, ...],
    weights: tuple[float, ...],
) -> pd.DataFrame:
    pred = make_recent_blend_prediction(
        panel=panel,
        train_end=FINAL_TRAIN_END,
        forecast_start=FINAL_FORECAST_START,
        horizon=HORIZON,
        target_col=TARGET_FOR_PRED,
        windows=windows,
        weights=weights,
    )
    pred = postprocess_prediction(
        pred=pred,
        panel=panel,
        sku_activity=sku_activity,
        train_end=FINAL_TRAIN_END,
        target_col=TARGET_FOR_PRED,
        sunday_factor=0.0,
        apply_cap=True,
    )
    return pred


def set_sundays_zero(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    sunday_cols = [col for col in pred.columns if pd.Timestamp(col).dayofweek == 6]
    pred.loc[:, sunday_cols] = 0.0
    return pred


def assign_rank_bucket(rank: float) -> str:
    if rank <= 100:
        return "rank_001_100"
    if rank <= 500:
        return "rank_101_500"
    if rank <= 1000:
        return "rank_501_1000"
    if rank <= 2000:
        return "rank_1001_2000"
    return "rank_2001_plus"


def assign_recency_bucket(days_since: float) -> str:
    if days_since == 9999:
        return "9999_no_sale"
    if days_since <= 14:
        return "<=14"
    if days_since <= 56:
        return "15-56"
    if days_since <= 180:
        return "57-180"
    return ">180"


def write_csv(path: Path, df: pd.DataFrame, overwrite: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    df.to_csv(path, index=False)
    return path


def to_rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def apply_candidate_rules(candidate_name: str, anchor_pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    profit_rank = meta["profit_rank"].fillna(999999)
    days_since = meta["days_since_last_sale"].fillna(9999)
    active_days = meta["active_days"].fillna(0)

    candidate = anchor_pred.copy()

    mask_top500 = profit_rank <= 500
    mask_501_1000 = (profit_rank > 500) & (profit_rank <= 1000)
    mask_gt1000 = profit_rank > 1000

    if candidate_name == V16_NAME:
        m = mask_501_1000
        m_le14 = m & (days_since <= 14)
        m_le56 = m & (days_since > 14) & (days_since <= 56)
        m_far_or_inactive = m & ((active_days <= 3) | (days_since > 180))
        m_rest = m & (~m_le14) & (~m_le56) & (~m_far_or_inactive)

        candidate.loc[m_le56.values, :] = anchor_pred.loc[m_le56.values, :].values * 0.50
        candidate.loc[m_far_or_inactive.values, :] = anchor_pred.loc[m_far_or_inactive.values, :].values * 0.10
        candidate.loc[m_rest.values, :] = anchor_pred.loc[m_rest.values, :].values * 0.25

        m = mask_gt1000
        m_le14 = m & (days_since <= 14)
        m_le56_active = m & (days_since <= 56) & (active_days > 10) & (~m_le14)
        m_le56 = m & (days_since <= 56) & (~m_le14) & (~m_le56_active)
        m_zero = m & (~m_le14) & (~m_le56_active) & (~m_le56)

        candidate.loc[m_le14.values, :] = anchor_pred.loc[m_le14.values, :].values * 0.35
        candidate.loc[m_le56_active.values, :] = anchor_pred.loc[m_le56_active.values, :].values * 0.08
        candidate.loc[m_le56.values, :] = 0.0
        candidate.loc[m_zero.values, :] = 0.0

    elif candidate_name == V14_NAME:
        m = mask_501_1000
        m_le14 = m & (days_since <= 14)
        m_le56 = m & (days_since > 14) & (days_since <= 56)
        m_far_or_inactive = m & ((active_days <= 3) | (days_since > 180))
        m_rest = m & (~m_le14) & (~m_le56) & (~m_far_or_inactive)

        candidate.loc[m_le56.values, :] = anchor_pred.loc[m_le56.values, :].values * 0.75
        candidate.loc[m_far_or_inactive.values, :] = anchor_pred.loc[m_far_or_inactive.values, :].values * 0.40
        candidate.loc[m_rest.values, :] = anchor_pred.loc[m_rest.values, :].values * 0.60

        m = mask_gt1000
        m_le14 = m & (days_since <= 14)
        m_le56_active = m & (days_since <= 56) & (active_days > 10) & (~m_le14)
        m_le56 = m & (days_since <= 56) & (~m_le14) & (~m_le56_active)
        m_zero = m & (~m_le14) & (~m_le56_active) & (~m_le56)

        candidate.loc[m_le14.values, :] = anchor_pred.loc[m_le14.values, :].values * 0.60
        candidate.loc[m_le56_active.values, :] = anchor_pred.loc[m_le56_active.values, :].values * 0.30
        candidate.loc[m_le56.values, :] = anchor_pred.loc[m_le56.values, :].values * 0.15
        candidate.loc[m_zero.values, :] = 0.0

    else:
        raise ValueError(f"Unknown candidate name: {candidate_name}")

    candidate.loc[mask_top500.values, :] = anchor_pred.loc[mask_top500.values, :].values
    candidate = candidate.clip(lower=0)
    candidate = set_sundays_zero(candidate)
    return candidate


def summarize_rank_delta(candidate_name: str, candidate_pred: pd.DataFrame, anchor_pred: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    profit_rank = meta["profit_rank"].fillna(999999)
    df = pd.DataFrame(
        {
            "ItemCode": anchor_pred.index,
            "profit_rank": profit_rank.values,
            "rank_bucket": profit_rank.apply(assign_rank_bucket).values,
            "anchor_total": anchor_pred.sum(axis=1).values,
            "candidate_total": candidate_pred.sum(axis=1).values,
        }
    )
    df["delta"] = df["candidate_total"] - df["anchor_total"]
    df["abs_delta"] = df["delta"].abs()

    summary = (
        df.groupby("rank_bucket", as_index=False)
        .agg(
            sku_count=("ItemCode", "count"),
            changed_sku_count=("abs_delta", lambda x: int((x > 1e-9).sum())),
            anchor_total=("anchor_total", "sum"),
            candidate_total=("candidate_total", "sum"),
            delta=("delta", "sum"),
            max_abs_delta=("abs_delta", "max"),
        )
    )
    summary["delta_pct"] = summary["delta"] / (summary["anchor_total"] + 1e-12)
    summary.insert(0, "candidate_name", candidate_name)

    out_path = DIAG_DIR / f"diagnostics_{candidate_name}_delta_by_rank_bucket.csv"
    out_path = write_csv(out_path, summary, overwrite=True)

    def get_val(bucket: str, metric: str) -> float:
        row = summary.loc[summary["rank_bucket"] == bucket]
        return float(row[metric].iloc[0]) if not row.empty else 0.0

    if abs(get_val("rank_001_100", "delta")) > 1e-8 or abs(get_val("rank_101_500", "delta")) > 1e-8:
        raise ValueError(f"Hard gate failed for {candidate_name}: rank <= 500 delta is not zero")
    if get_val("rank_001_100", "max_abs_delta") > 1e-8 or get_val("rank_101_500", "max_abs_delta") > 1e-8:
        raise ValueError(f"Hard gate failed for {candidate_name}: rank <= 500 max_abs_delta is not zero")
    if "rank_501_1000" in summary["rank_bucket"].values:
        r501 = float(summary.loc[summary["rank_bucket"] == "rank_501_1000", "delta_pct"].iloc[0])
        if r501 < -0.60:
            raise ValueError(f"Hard gate failed for {candidate_name}: rank_501_1000 delta_pct < -60%")

    return summary, out_path


def summarize_recency_delta(candidate_name: str, candidate_pred: pd.DataFrame, anchor_pred: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, Path, list[str], float]:
    days_since = meta["days_since_last_sale"].fillna(9999)
    df = pd.DataFrame(
        {
            "ItemCode": anchor_pred.index,
            "days_since_last_sale": days_since.values,
            "anchor_total": anchor_pred.sum(axis=1).values,
            "candidate_total": candidate_pred.sum(axis=1).values,
        }
    )
    df["delta"] = df["candidate_total"] - df["anchor_total"]
    df["abs_delta"] = df["delta"].abs()
    df["recency_bucket"] = df["days_since_last_sale"].apply(assign_recency_bucket)

    summary = (
        df.groupby("recency_bucket", as_index=False)
        .agg(
            sku_count=("ItemCode", "count"),
            changed_sku_count=("abs_delta", lambda x: int((x > 1e-9).sum())),
            anchor_total=("anchor_total", "sum"),
            candidate_total=("candidate_total", "sum"),
            delta=("delta", "sum"),
        )
    )
    summary["delta_pct"] = summary["delta"] / (summary["anchor_total"] + 1e-12)
    summary.insert(0, "candidate_name", candidate_name)

    out_path = DIAG_DIR / f"diagnostics_{candidate_name}_delta_by_recency_bucket.csv"
    out_path = write_csv(out_path, summary, overwrite=True)

    warnings_list: list[str] = []
    total_delta_pct = float(df["delta"].sum() / (df["anchor_total"].sum() + 1e-12))
    if total_delta_pct < -0.25:
        warnings_list.append("WARNING: candidate likely too aggressive.")

    row14 = summary.loc[summary["recency_bucket"] == "<=14"]
    if not row14.empty:
        rec14 = float(row14["delta_pct"].iloc[0])
        if rec14 < -0.22:
            raise ValueError(f"Hard gate failed for {candidate_name}: recency <=14 delta_pct < -0.22")
        if rec14 < -0.18:
            warnings_list.append("WARNING: recency <=14 delta_pct < -0.18.")

    row_15_56 = summary.loc[summary["recency_bucket"] == "15-56"]
    if not row_15_56.empty:
        rec_15_56 = float(row_15_56["delta_pct"].iloc[0])
        if rec_15_56 < -0.90:
            warnings_list.append("WARNING: recency 15-56 delta_pct < -0.90.")

    return summary, out_path, warnings_list, total_delta_pct


def check_recent_longtail_not_zeroed(candidate_name: str, candidate_pred: pd.DataFrame, anchor_pred: pd.DataFrame, meta: pd.DataFrame) -> tuple[int, int, int]:
    profit_rank = meta["profit_rank"].fillna(999999)
    days_since = meta["days_since_last_sale"].fillna(9999)
    mask = (profit_rank > 1000) & (days_since <= 14)

    anchor_total = anchor_pred.sum(axis=1)
    candidate_total = candidate_pred.sum(axis=1)
    group = pd.DataFrame({"anchor_total": anchor_total, "candidate_total": candidate_total}).loc[mask.index]
    group = group.loc[mask.values]

    n_group = len(group)
    n_anchor_pos = int((group["anchor_total"] > 0).sum())
    n_candidate_zero = int(((group["candidate_total"] == 0) & (group["anchor_total"] > 0)).sum())

    print(f" - recent rank>1000 & days_since<=14 SKU count ({candidate_name}):", n_group)
    print(f" - anchor_total>0 in that group ({candidate_name}):", n_anchor_pos)
    print(f" - candidate_total==0 while anchor_total>0 ({candidate_name}):", n_candidate_zero)

    if n_candidate_zero > 0:
        raise ValueError(f"Hard gate failed for {candidate_name}: recent long-tail SKUs were zeroed")

    return n_group, n_anchor_pos, n_candidate_zero


def write_top_changed_sku(candidate_name: str, candidate_pred: pd.DataFrame, anchor_pred: pd.DataFrame, meta: pd.DataFrame) -> Path:
    delta = candidate_pred - anchor_pred
    abs_delta_by_sku = delta.abs().sum(axis=1)
    df = pd.DataFrame(
        {
            "ItemCode": anchor_pred.index,
            "anchor_total": anchor_pred.sum(axis=1).values,
            "candidate_total": candidate_pred.sum(axis=1).values,
            "delta": (candidate_pred.sum(axis=1) - anchor_pred.sum(axis=1)).values,
            "abs_delta": abs_delta_by_sku.values,
        }
    )

    meta_df = meta.reset_index()
    if "ItemCode" not in meta_df.columns:
        meta_df = meta_df.rename(columns={meta_df.columns[0]: "ItemCode"})
    df = df.merge(
        meta_df[["ItemCode", "profit_rank", "active_days", "days_since_last_sale"]],
        on="ItemCode",
        how="left",
    )
    out = df.sort_values("abs_delta", ascending=False).head(100)
    out.insert(0, "candidate_name", candidate_name)

    out_path = DIAG_DIR / f"diagnostics_{candidate_name}_top_changed_sku.csv"
    return write_csv(out_path, out, overwrite=True)


def write_submission_and_diagnostics(candidate_name: str, public_score_reference: float, candidate_pred: pd.DataFrame, anchor_pred: pd.DataFrame, meta: pd.DataFrame, sample: pd.DataFrame) -> dict:
    f_cols = [f"F{i}" for i in range(1, 29)]

    submission_filename = V16_FILE if candidate_name == V16_NAME else V14_FILE
    submission_path = FINAL_SUB_DIR / submission_filename
    submission_df = make_kaggle_submission(pred_56=candidate_pred, sample=sample, output_path=None)
    submission_path = write_csv(submission_path, submission_df, overwrite=True)

    print(f"\nCandidate: {candidate_name}")
    print(" - submission_path:", to_rel(submission_path))
    print(" - shape:", submission_df.shape)
    print(" - unique ids:", submission_df["id"].nunique())
    print(" - missing:", int(submission_df[f_cols].isna().sum().sum()))
    print(" - inf:", int(np.isinf(submission_df[f_cols].to_numpy()).sum()))
    print(" - negative:", int((submission_df[f_cols] < 0).sum().sum()))
    print(" - total_forecast:", float(submission_df[f_cols].sum().sum()))
    print(" - max_forecast:", float(submission_df[f_cols].max().max()))

    validation_rows = submission_df["id"].str.endswith("_validation")
    evaluation_rows = submission_df["id"].str.endswith("_evaluation")
    sunday_f = ["F2", "F9", "F16", "F23"]
    validation_sunday_total = float(submission_df.loc[validation_rows, sunday_f].sum().sum())
    evaluation_sunday_total = float(submission_df.loc[evaluation_rows, sunday_f].sum().sum())
    print(" - validation Sunday total:", validation_sunday_total)
    print(" - evaluation Sunday total:", evaluation_sunday_total)

    if submission_df.shape != sample.shape:
        raise ValueError(f"{candidate_name}: submission shape mismatch: {submission_df.shape} != {sample.shape}")
    if submission_df["id"].tolist() != sample["id"].tolist():
        raise ValueError(f"{candidate_name}: id order mismatch with sample")
    if int(submission_df[f_cols].isna().sum().sum()) != 0:
        raise ValueError(f"{candidate_name}: submission contains NaN")
    if not np.isfinite(submission_df[f_cols].to_numpy()).all():
        raise ValueError(f"{candidate_name}: submission contains inf")
    if int((submission_df[f_cols] < 0).sum().sum()) != 0:
        raise ValueError(f"{candidate_name}: submission contains negative values")
    if abs(validation_sunday_total) >= 1e-8 or abs(evaluation_sunday_total) >= 1e-8:
        raise ValueError(f"{candidate_name}: Sunday totals are not zero")

    delta = candidate_pred - anchor_pred
    abs_delta_by_sku = delta.abs().sum(axis=1)
    changed_skus = abs_delta_by_sku[abs_delta_by_sku > 1e-9].index
    anchor_total = float(anchor_pred.values.sum())
    candidate_total = float(candidate_pred.values.sum())
    total_delta = candidate_total - anchor_total
    total_delta_pct = total_delta / (anchor_total + 1e-12)
    max_abs_delta = float(delta.abs().max().max())

    print(" - changed SKU count:", len(changed_skus))
    print(" - anchor total:", anchor_total)
    print(" - candidate total:", candidate_total)
    print(" - total delta:", total_delta)
    print(" - total delta pct:", total_delta_pct)
    print(" - max abs delta:", max_abs_delta)

    rank_summary, rank_path = summarize_rank_delta(candidate_name, candidate_pred, anchor_pred, meta)
    recency_summary, recency_path, warnings_list, _ = summarize_recency_delta(candidate_name, candidate_pred, anchor_pred, meta)
    n_group, n_anchor_pos, n_candidate_zero = check_recent_longtail_not_zeroed(candidate_name, candidate_pred, anchor_pred, meta)
    top_changed_path = write_top_changed_sku(candidate_name, candidate_pred, anchor_pred, meta)

    def get_metric(df: pd.DataFrame, bucket_col: str, bucket_value: str, metric: str) -> float:
        row = df.loc[df[bucket_col] == bucket_value]
        return float(row[metric].iloc[0]) if not row.empty else 0.0

    rank_001_100_delta = get_metric(rank_summary, "rank_bucket", "rank_001_100", "delta")
    rank_101_500_delta = get_metric(rank_summary, "rank_bucket", "rank_101_500", "delta")
    rank_001_100_max_abs = get_metric(rank_summary, "rank_bucket", "rank_001_100", "max_abs_delta")
    rank_101_500_max_abs = get_metric(rank_summary, "rank_bucket", "rank_101_500", "max_abs_delta")
    rank_le500_max_abs = max(rank_001_100_max_abs, rank_101_500_max_abs)

    print(" - rank <=500 delta:", rank_001_100_delta + rank_101_500_delta)
    print(" - rank <=500 max_abs_delta:", rank_le500_max_abs)
    print(" - recent_rank_gt1000_zeroed_count:", n_candidate_zero)

    if abs(rank_001_100_delta) > 1e-8 or abs(rank_101_500_delta) > 1e-8:
        raise ValueError(f"Hard gate failed for {candidate_name}: rank <= 500 delta is not zero")
    if rank_001_100_max_abs > 1e-8 or rank_101_500_max_abs > 1e-8:
        raise ValueError(f"Hard gate failed for {candidate_name}: rank <= 500 max_abs_delta is not zero")

    summary_row = {
        "candidate_name": candidate_name,
        "submission_path": to_rel(submission_path),
        "public_score_reference": public_score_reference,
        "total_forecast": candidate_total,
        "total_delta": total_delta,
        "total_delta_pct": total_delta_pct,
        "changed_sku_count": int(len(changed_skus)),
        "max_abs_delta": max_abs_delta,
        "rank_001_100_delta": rank_001_100_delta,
        "rank_101_500_delta": rank_101_500_delta,
        "rank_501_1000_delta_pct": get_metric(rank_summary, "rank_bucket", "rank_501_1000", "delta_pct"),
        "rank_1001_2000_delta_pct": get_metric(rank_summary, "rank_bucket", "rank_1001_2000", "delta_pct"),
        "rank_2001_plus_delta_pct": get_metric(rank_summary, "rank_bucket", "rank_2001_plus", "delta_pct"),
        "recency_le_14_delta_pct": get_metric(recency_summary, "recency_bucket", "<=14", "delta_pct"),
        "recency_15_56_delta_pct": get_metric(recency_summary, "recency_bucket", "15-56", "delta_pct"),
        "recent_rank_gt1000_zeroed_count": int(n_candidate_zero),
        "pass_hard_gates": True,
        "rank_diag_path": to_rel(rank_path),
        "recency_diag_path": to_rel(recency_path),
        "top_changed_path": to_rel(top_changed_path),
    }

    if summary_row["recent_rank_gt1000_zeroed_count"] != 0:
        raise ValueError(f"Hard gate failed for {candidate_name}: recent_rank_gt1000_zeroed_count != 0")
    if summary_row["recency_le_14_delta_pct"] < -0.22:
        raise ValueError(f"Hard gate failed for {candidate_name}: recency <=14 delta_pct < -0.22")
    if summary_row["rank_501_1000_delta_pct"] < -0.60:
        raise ValueError(f"Hard gate failed for {candidate_name}: rank_501_1000 delta_pct < -60%")

    if total_delta_pct < -0.25:
        print(f"WARNING: {candidate_name} may be too aggressive. total_delta_pct={total_delta_pct:.6f}")
    if summary_row["recency_le_14_delta_pct"] < -0.18:
        print(f"WARNING: {candidate_name} recency <=14 delta_pct < -0.18 -> {summary_row['recency_le_14_delta_pct']:.6f}")
    if summary_row["recency_15_56_delta_pct"] < -0.90:
        print(f"WARNING: {candidate_name} recency 15-56 delta_pct < -0.90 -> {summary_row['recency_15_56_delta_pct']:.6f}")

    print(" - recent rank>1000 & days_since<=14 SKU count:", n_group)
    print(" - anchor_total>0 in that group:", n_anchor_pos)
    print(" - candidate_total==0 while anchor_total>0:", n_candidate_zero)

    return summary_row


def main() -> None:
    ensure_dirs()
    daily_panel, sku_activity = load_or_build_processed_data()
    _train, sample = load_raw_data()

    anchor_pred = build_recent_prediction(daily_panel, sku_activity, (14, 18, 21), (0.20, 0.45, 0.35))
    anchor_pred = anchor_pred.reindex(sorted(anchor_pred.index))
    meta = sku_activity.set_index("ItemCode").reindex(anchor_pred.index)

    summary_rows = []
    summary_rows.append(
        write_submission_and_diagnostics(
            candidate_name=V16_NAME,
            public_score_reference=PUBLIC_SCORE_REFERENCE[V16_NAME],
            candidate_pred=apply_candidate_rules(V16_NAME, anchor_pred, meta),
            anchor_pred=anchor_pred,
            meta=meta,
            sample=sample,
        )
    )
    summary_rows.append(
        write_submission_and_diagnostics(
            candidate_name=V14_NAME,
            public_score_reference=PUBLIC_SCORE_REFERENCE[V14_NAME],
            candidate_pred=apply_candidate_rules(V14_NAME, anchor_pred, meta),
            anchor_pred=anchor_pred,
            meta=meta,
            sample=sample,
        )
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[[
        "candidate_name",
        "submission_path",
        "public_score_reference",
        "total_forecast",
        "total_delta",
        "total_delta_pct",
        "changed_sku_count",
        "max_abs_delta",
        "rank_001_100_delta",
        "rank_101_500_delta",
        "rank_501_1000_delta_pct",
        "rank_1001_2000_delta_pct",
        "rank_2001_plus_delta_pct",
        "recency_le_14_delta_pct",
        "recency_15_56_delta_pct",
        "recent_rank_gt1000_zeroed_count",
        "pass_hard_gates",
        "rank_diag_path",
        "recency_diag_path",
        "top_changed_path",
    ]]

    summary_path = DIAG_DIR / "final_submission_summary.csv"
    summary_path = write_csv(summary_path, summary_df, overwrite=True)

    print("\nREADY TO REVIEW, NOT AUTO-SUBMITTED")
    print("Submission path:")
    for path in summary_df["submission_path"].tolist():
        print(" -", path)
    print("Diagnostics paths:")
    for col in ["rank_diag_path", "recency_diag_path", "top_changed_path"]:
        for path in summary_df[col].tolist():
            print(" -", path)
    print("Summary row:")
    print(summary_df.to_string(index=False))
    print("Summary path:", to_rel(summary_path))
    print("SUCCESS: final reproducible submissions generated.")
    print("FINAL REPRODUCTION COMPLETE")
    print("Generated:")
    print(f"1. {to_rel(FINAL_SUB_DIR / V16_FILE)}")
    print(f"2. {to_rel(FINAL_SUB_DIR / V14_FILE)}")


if __name__ == "__main__":
    main()
