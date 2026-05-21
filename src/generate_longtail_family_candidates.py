"""Generate multiple long-tail family candidates (v12, v13, blends).

Run from project root:
    python src/generate_longtail_family_candidates.py

Produces CSVs in `outputs/submissions_candidates/` and diagnostics in
`outputs/diagnostics/`. Does not auto-submit or overwrite existing files
with the same name (adds timestamp suffix).
"""
from pathlib import Path
import datetime
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_reproduce import (
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


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_SUB_DIR = PROJECT_ROOT / "outputs" / "submissions_candidates"
DIAG_DIR = PROJECT_ROOT / "outputs" / "diagnostics"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUB_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_processed_data():
    daily_panel_path = PROCESSED_DIR / "daily_panel.parquet"
    sku_activity_path = PROCESSED_DIR / "sku_activity.parquet"

    if daily_panel_path.exists() and sku_activity_path.exists():
        daily_panel = pd.read_parquet(daily_panel_path)
        sku_activity = pd.read_parquet(sku_activity_path)
        return daily_panel, sku_activity

    train, sample = load_raw_data()
    train_clean = clean_train(train)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    daily_panel.to_parquet(daily_panel_path)
    sku_activity.to_parquet(sku_activity_path)
    return daily_panel, sku_activity


def build_recent_prediction(panel, sku_activity, windows, weights):
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
    if sunday_cols:
        pred.loc[:, sunday_cols] = 0.0
    return pred


def assign_rank_bucket(r):
    if r <= 100:
        return "rank_001_100"
    if r <= 500:
        return "rank_101_500"
    if r <= 1000:
        return "rank_501_1000"
    if r <= 2000:
        return "rank_1001_2000"
    return "rank_2001_plus"


def assign_recency_bucket(ds):
    if ds == 9999:
        return "9999_no_sale"
    if ds <= 14:
        return "<=14"
    if ds <= 56:
        return "15-56"
    if ds <= 180:
        return "57-180"
    return ">180"


def apply_longtail_rules(version: str, anchor_pred: pd.DataFrame, recent56_pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # meta indexed by ItemCode
    profit_rank = meta["profit_rank"].fillna(999999)
    days_since = meta["days_since_last_sale"].fillna(9999)
    active_days = meta["active_days"].fillna(0)

    candidate = anchor_pred.copy()

    mask_top500 = profit_rank <= 500
    mask_501_1000 = (profit_rank > 500) & (profit_rank <= 1000)
    mask_gt1000 = profit_rank > 1000

    if version == "v12":
        # 501-1000
        m = mask_501_1000
        m_ds_le14 = m & (days_since <= 14)
        m_ds_15_56 = m & (days_since > 14) & (days_since <= 56)
        candidate.loc[m_ds_15_56.values, :] = anchor_pred.loc[m_ds_15_56.values, :].values * 0.90
        m_inactive_or_far = m & ((active_days <= 3) | (days_since > 180))
        candidate.loc[m_inactive_or_far.values, :] = anchor_pred.loc[m_inactive_or_far.values, :].values * 0.65
        m_remaining = m & (~m_ds_le14) & (~m_ds_15_56) & (~m_inactive_or_far)
        candidate.loc[m_remaining.values, :] = anchor_pred.loc[m_remaining.values, :].values * 0.80

        # >1000
        m = mask_gt1000
        m_ds_le14 = m & (days_since <= 14)
        candidate.loc[m_ds_le14.values, :] = anchor_pred.loc[m_ds_le14.values, :].values * 0.75
        m_ds_le56_active = m & (days_since <= 56) & (active_days > 10) & (~m_ds_le14)
        candidate.loc[m_ds_le56_active.values, :] = anchor_pred.loc[m_ds_le56_active.values, :].values * 0.50
        m_ds_le56 = m & (days_since <= 56) & (~m_ds_le14) & (~m_ds_le56_active)
        candidate.loc[m_ds_le56.values, :] = anchor_pred.loc[m_ds_le56.values, :].values * 0.35
        m_zero = m & (~m_ds_le14) & (~m_ds_le56_active) & (~m_ds_le56)
        candidate.loc[m_zero.values, :] = 0.0

    elif version == "v13":
        # 501-1000
        m = mask_501_1000
        m_ds_le14 = m & (days_since <= 14)
        m_ds_15_56 = m & (days_since > 14) & (days_since <= 56)
        candidate.loc[m_ds_15_56.values, :] = anchor_pred.loc[m_ds_15_56.values, :].values * 0.85
        m_inactive_or_far = m & ((active_days <= 3) | (days_since > 180))
        candidate.loc[m_inactive_or_far.values, :] = anchor_pred.loc[m_inactive_or_far.values, :].values * 0.55
        m_remaining = m & (~m_ds_le14) & (~m_ds_15_56) & (~m_inactive_or_far)
        candidate.loc[m_remaining.values, :] = anchor_pred.loc[m_remaining.values, :].values * 0.70

        # >1000
        m = mask_gt1000
        m_ds_le14 = m & (days_since <= 14)
        candidate.loc[m_ds_le14.values, :] = anchor_pred.loc[m_ds_le14.values, :].values * 0.70
        m_ds_le56_active = m & (days_since <= 56) & (active_days > 10) & (~m_ds_le14)
        candidate.loc[m_ds_le56_active.values, :] = anchor_pred.loc[m_ds_le56_active.values, :].values * 0.45
        m_ds_le56 = m & (days_since <= 56) & (~m_ds_le14) & (~m_ds_le56_active)
        candidate.loc[m_ds_le56.values, :] = anchor_pred.loc[m_ds_le56.values, :].values * 0.25
        m_zero = m & (~m_ds_le14) & (~m_ds_le56_active) & (~m_ds_le56)
        candidate.loc[m_zero.values, :] = 0.0

    else:
        raise ValueError(f"Unknown version: {version}")

    # Ensure top500 unchanged
    candidate.loc[mask_top500.values, :] = anchor_pred.loc[mask_top500.values, :].values

    # clip lower bound and set Sundays to zero (via helper)
    candidate = candidate.clip(lower=0)
    candidate = set_sundays_zero(candidate)
    return candidate


def safe_write(path: Path, df: pd.DataFrame):
    if path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path.with_name(f"{path.stem}_{ts}{path.suffix}")
    df.to_csv(path, index=False)
    return path


def summarize_rank_delta(candidate_name, anchor_pred, candidate_pred, profit_rank, out_dir):
    df = pd.DataFrame({
        "ItemCode": anchor_pred.index,
        "profit_rank": profit_rank.values,
        "rank_bucket": profit_rank.apply(assign_rank_bucket).values,
        "anchor_total": anchor_pred.sum(axis=1).values,
        "candidate_total": candidate_pred.sum(axis=1).values,
    })
    df["delta"] = df["candidate_total"] - df["anchor_total"]
    df["abs_delta"] = df["delta"].abs()

    bucket_summary = (
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
    bucket_summary["delta_pct"] = bucket_summary["delta"] / (bucket_summary["anchor_total"] + 1e-12)

    out_path = out_dir / f"diagnostics_{candidate_name}_delta_by_rank_bucket.csv"
    safe_write(out_path, bucket_summary)

    # Hard gates
    def gval(b, col):
        r = bucket_summary.loc[bucket_summary["rank_bucket"] == b]
        return float(r[col].iloc[0]) if not r.empty else 0.0

    if abs(gval("rank_001_100", "delta")) > 1e-8 or abs(gval("rank_101_500", "delta")) > 1e-8:
        raise ValueError("Hard gate failed: changes detected in rank <= 500 group")
    if gval("rank_001_100", "max_abs_delta") > 1e-8 or gval("rank_101_500", "max_abs_delta") > 1e-8:
        raise ValueError("Hard gate failed: max_abs_delta in rank <=500 > 0")
    if "rank_501_1000" in bucket_summary["rank_bucket"].values:
        if float(bucket_summary.loc[bucket_summary["rank_bucket"] == "rank_501_1000", "delta_pct"].iloc[0]) < -0.25:
            raise ValueError("Hard gate failed: rank_501_1000 delta_pct < -25%")

    return bucket_summary, out_path


def summarize_recency_delta(candidate_name, anchor_pred, candidate_pred, days_since, out_dir):
    df = pd.DataFrame({
        "ItemCode": anchor_pred.index,
        "days_since": days_since.values,
        "anchor_total": anchor_pred.sum(axis=1).values,
        "candidate_total": candidate_pred.sum(axis=1).values,
    })
    df["delta"] = df["candidate_total"] - df["anchor_total"]
    df["abs_delta"] = df["delta"].abs()
    df["recency_bucket"] = df["days_since"].apply(assign_recency_bucket)

    recency_summary = (
        df.groupby("recency_bucket", as_index=False)
        .agg(
            sku_count=("ItemCode", "count"),
            changed_sku_count=("abs_delta", lambda x: int((x > 1e-9).sum())),
            anchor_total=("anchor_total", "sum"),
            candidate_total=("candidate_total", "sum"),
            delta=("delta", "sum"),
        )
    )
    recency_summary["delta_pct"] = recency_summary["delta"] / (recency_summary["anchor_total"] + 1e-12)

    out_path = out_dir / f"diagnostics_{candidate_name}_delta_by_recency_bucket.csv"
    safe_write(out_path, recency_summary)

    total_delta_pct = recency_summary["delta"].sum() / (recency_summary["anchor_total"].sum() + 1e-12)
    warnings_list = []
    if total_delta_pct < -0.13:
        warnings_list.append("total_delta_pct < -13%")
    rec_le14 = recency_summary.loc[recency_summary["recency_bucket"] == "<=14"]
    if not rec_le14.empty:
        rec_le14_pct = float(rec_le14["delta_pct"].iloc[0])
        if rec_le14_pct < -0.15:
            raise ValueError("Hard gate failed: recency <=14 delta_pct < -15%")
        if rec_le14_pct < -0.10:
            warnings_list.append("recency <=14 delta_pct < -10%")

    return recency_summary, out_path, warnings_list


def check_recent_longtail_not_zeroed(anchor_pred, candidate_pred, profit_rank, days_since):
    mask = (profit_rank > 1000) & (days_since <= 14)
    group = pd.DataFrame({
        "anchor_total": anchor_pred.sum(axis=1),
        "candidate_total": candidate_pred.sum(axis=1),
    }).loc[mask.index]
    group = group.loc[mask.values]
    n_group = len(group)
    n_anchor_pos = int((group["anchor_total"] > 0).sum())
    n_candidate_zero = int(((group["candidate_total"] == 0) & (group["anchor_total"] > 0)).sum())
    if n_candidate_zero > 0:
        raise ValueError("Some recently-active long-tail SKUs were zeroed while anchor>0; abort")
    return n_group, n_anchor_pos, n_candidate_zero


def write_top_changed(candidate_name, anchor_pred, candidate_pred, meta, out_dir):
    delta = candidate_pred - anchor_pred
    abs_delta = delta.abs().sum(axis=1)
    df = pd.DataFrame({
        "ItemCode": anchor_pred.index,
        "anchor_total": anchor_pred.sum(axis=1).values,
        "candidate_total": candidate_pred.sum(axis=1).values,
        "delta": (candidate_pred.sum(axis=1) - anchor_pred.sum(axis=1)).values,
        "abs_delta": abs_delta.values,
    })
    meta_df = meta.reset_index()
    if "ItemCode" not in meta_df.columns:
        meta_df = meta_df.rename(columns={meta_df.columns[0]: "ItemCode"})
    df = df.merge(meta_df[["ItemCode", "profit_rank", "active_days", "days_since_last_sale"]], on="ItemCode", how="left")
    out = df.sort_values("abs_delta", ascending=False).head(100)
    out_path = out_dir / f"diagnostics_{candidate_name}_top_changed_sku.csv"
    safe_write(out_path, out)
    return out_path


def write_submission_and_diagnostics(candidate_name, candidate_pred, anchor_pred, meta, sample, out_dirs):
    # sample: dataframe
    profit_rank = meta["profit_rank"].fillna(999999)
    days_since = meta["days_since_last_sale"].fillna(9999)

    submission_path = OUTPUT_SUB_DIR / f"submission_{candidate_name}.csv"
    submission_df = make_kaggle_submission(pred_56=candidate_pred, sample=sample, output_path=None)
    # save without overwriting
    saved_path = safe_write(submission_path, submission_df)

    # QA prints
    f_cols = [f"F{i}" for i in range(1, 29)]
    print(f"\nCandidate: {candidate_name}")
    print(" - path:", saved_path)
    print(" - shape:", submission_df.shape)
    print(" - unique ids:", submission_df["id"].nunique())
    print(" - missing:", int(submission_df[f_cols].isna().sum().sum()))
    print(" - inf:", int(np.isinf(submission_df[f_cols].to_numpy()).sum()))
    print(" - negative:", int((submission_df[f_cols] < 0).sum().sum()))
    print(" - total:", float(submission_df[f_cols].sum().sum()))
    print(" - max:", float(submission_df[f_cols].max().max()))

    validation_rows = submission_df["id"].str.endswith("_validation")
    evaluation_rows = submission_df["id"].str.endswith("_evaluation")
    sunday_f = ["F2", "F9", "F16", "F23"]
    val_sunday = float(submission_df.loc[validation_rows, sunday_f].sum().sum())
    eval_sunday = float(submission_df.loc[evaluation_rows, sunday_f].sum().sum())
    print(" - validation Sunday total:", val_sunday)
    print(" - evaluation Sunday total:", eval_sunday)

    # Assertions
    assert submission_df.shape == sample.shape
    assert submission_df["id"].tolist() == sample["id"].tolist()
    assert submission_df[f_cols].isna().sum().sum() == 0
    assert np.isfinite(submission_df[f_cols].to_numpy()).all()
    assert (submission_df[f_cols] < 0).sum().sum() == 0
    assert abs(val_sunday) < 1e-8 and abs(eval_sunday) < 1e-8

    # Delta QA
    delta = candidate_pred.reindex(anchor_pred.index).fillna(0) - anchor_pred.reindex(anchor_pred.index).fillna(0)
    abs_delta_by_sku = delta.abs().sum(axis=1)
    changed_skus = abs_delta_by_sku[abs_delta_by_sku > 1e-9].index
    anchor_total = anchor_pred.values.sum()
    candidate_total = candidate_pred.values.sum()
    total_delta = candidate_total - anchor_total
    total_delta_pct = total_delta / (anchor_total + 1e-12)
    max_abs_delta = delta.abs().max().max()
    print(" - changed SKU count:", len(changed_skus))
    print(" - anchor total:", anchor_total)
    print(" - candidate total:", candidate_total)
    print(" - total delta:", total_delta)
    print(" - total delta pct:", total_delta_pct)
    print(" - max abs delta:", max_abs_delta)

    # Rank diagnostics
    bucket_summary, rank_path = summarize_rank_delta(candidate_name, anchor_pred, candidate_pred, profit_rank, DIAG_DIR)

    # Recency diagnostics
    recency_summary, recency_path, warnings_list = summarize_recency_delta(candidate_name, anchor_pred, candidate_pred, days_since, DIAG_DIR)

    # recent long-tail safety
    n_group, n_anchor_pos, n_candidate_zero = check_recent_longtail_not_zeroed(anchor_pred, candidate_pred, profit_rank, days_since)
    print(" - recent rank>1000 group size:", n_group)
    print(" - anchor>0 in group:", n_anchor_pos)
    print(" - zeroed while anchor>0:", n_candidate_zero)

    # Top changed
    top_path = write_top_changed(candidate_name, anchor_pred, candidate_pred, meta, DIAG_DIR)

    return {
        "candidate_name": candidate_name,
        "submission_path": str(saved_path),
        "total_forecast": float(candidate_total),
        "total_delta": float(total_delta),
        "total_delta_pct": float(total_delta_pct),
        "changed_sku_count": int(len(changed_skus)),
        "max_abs_delta": float(max_abs_delta),
        "rank_diag_path": str(rank_path),
        "recency_diag_path": str(recency_path),
        "top_changed_path": str(top_path),
        "warnings": "; ".join(warnings_list),
        "pass_hard_gates": True,
    }


def main():
    ensure_dirs()
    daily_panel, sku_activity = load_or_build_processed_data()
    # sample for submission
    train, sample = load_raw_data()

    anchor_pred = build_recent_prediction(daily_panel, sku_activity, (14, 18, 21), (0.20, 0.45, 0.35))
    recent56_pred = build_recent_prediction(daily_panel, sku_activity, (56,), (1.0,))

    anchor_pred = anchor_pred.reindex(sorted(anchor_pred.index))
    recent56_pred = recent56_pred.reindex(anchor_pred.index).fillna(0)

    meta = sku_activity.set_index("ItemCode").reindex(anchor_pred.index)

    candidates = []

    # Build v12 and v13
    v12 = apply_longtail_rules("v12", anchor_pred, recent56_pred, meta)
    v13 = apply_longtail_rules("v13", anchor_pred, recent56_pred, meta)

    # blends
    v12_blend85 = (v12 * 0.85) + (anchor_pred * 0.15)
    v12_blend85 = set_sundays_zero(v12_blend85.clip(lower=0))

    v13_blend85 = (v13 * 0.85) + (anchor_pred * 0.15)
    v13_blend85 = set_sundays_zero(v13_blend85.clip(lower=0))

    # Map candidates to filenames
    candidate_map = {
        "anchor_longtail_v12": ("submission_PUBLIC_anchor_longtail_soft_recency_v12_all_rows.csv", v12),
        "anchor_longtail_v13": ("submission_PUBLIC_anchor_longtail_soft_recency_v13_all_rows.csv", v13),
        "anchor_longtail_v12_blend85": ("submission_PUBLIC_anchor_longtail_soft_recency_v12_blend85_all_rows.csv", v12_blend85),
        "anchor_longtail_v13_blend85": ("submission_PUBLIC_anchor_longtail_soft_recency_v13_blend85_all_rows.csv", v13_blend85),
    }

    summary_records = []

    for cname, (fname, pred) in candidate_map.items():
        try:
            # ensure 56 cols
            if pred.shape[1] != HORIZON:
                raise ValueError(f"Candidate {cname} prediction must have {HORIZON} columns")

            # save submission and diagnostics
            rec = write_submission_and_diagnostics(cname, pred, anchor_pred, meta, sample, (OUTPUT_SUB_DIR, DIAG_DIR))
            summary_records.append(rec)
        except Exception as e:
            print(f"ERROR for {cname}: {e}")
            summary_records.append({"candidate_name": cname, "error": str(e)})

    # Comparison summary
    comp_rows = []
    for r in summary_records:
        if "error" in r:
            comp_rows.append({
                "candidate_name": r["candidate_name"],
                "pass_hard_gates": False,
                "error": r["error"],
            })
            continue
        comp_rows.append({
            "candidate_name": r["candidate_name"],
            "submission_path": r["submission_path"],
            "total_forecast": r["total_forecast"],
            "total_delta": r["total_delta"],
            "total_delta_pct": r["total_delta_pct"],
            "changed_sku_count": r["changed_sku_count"],
            "max_abs_delta": r["max_abs_delta"],
            "rank_diag_path": r["rank_diag_path"],
            "recency_diag_path": r["recency_diag_path"],
            "top_changed_path": r["top_changed_path"],
            "warnings": r["warnings"],
            "pass_hard_gates": r.get("pass_hard_gates", False),
        })

    comp_df = pd.DataFrame(comp_rows)
    # sort safest (least negative delta_pct) to most aggressive
    if "total_delta_pct" in comp_df.columns:
        comp_df = comp_df.sort_values("total_delta_pct", ascending=False)

    comp_path = DIAG_DIR / "longtail_candidate_comparison_summary.csv"
    safe_write(comp_path, comp_df)

    print("\nREADY TO REVIEW, NOT AUTO-SUBMITTED")
    print("Submissions:")
    for r in summary_records:
        if "submission_path" in r:
            print(" -", r["submission_path"])
    print("Diagnostics summary:", comp_path)


if __name__ == "__main__":
    main()
