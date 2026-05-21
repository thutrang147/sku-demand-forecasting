"""Generate candidate submission: PUBLIC_anchor_longtail_soft_recency_v15_all_rows

Usage:
    python src/generate_longtail_soft_recency_candidate.py

This script follows the user's specification and re-uses helper functions from
`src/final_reproduce.py` when available.
"""
from pathlib import Path
import sys
import datetime
import warnings

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.final_reproduce import (
        load_raw_data,
        clean_train,
        make_daily_panel,
        make_sku_activity,
        make_recent_blend_prediction,
        postprocess_prediction,
        make_kaggle_submission,
    )
except Exception as e:
    raise ImportError("Cannot import required helpers from src.final_reproduce.py: " + str(e))


# Constants / paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_SUB_DIR = PROJECT_ROOT / "outputs" / "submissions_candidates"
DIAG_DIR = PROJECT_ROOT / "outputs" / "diagnostics"

SUBMISSION_BASENAME = "submission_PUBLIC_anchor_longtail_soft_recency_v10_all_rows.csv"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUB_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_processed_data():
    daily_panel_path = PROCESSED_DIR / "daily_panel.parquet"
    sku_activity_path = PROCESSED_DIR / "sku_activity.parquet"

    if daily_panel_path.exists() and sku_activity_path.exists():
        print("Loading cached processed data...")
        daily_panel = pd.read_parquet(daily_panel_path)
        sku_activity = pd.read_parquet(sku_activity_path)
        return daily_panel, sku_activity

    print("Building processed data from raw files (this may take a while)...")
    train_path = RAW_DIR / "train.csv"
    sample_path = RAW_DIR / "sample_submission.csv"
    if not train_path.exists() or not sample_path.exists():
        raise FileNotFoundError("Expected raw files under data/raw/ (train.csv, sample_submission.csv)")

    # Use helpers from final_reproduce (no args)
    raw, _ = load_raw_data()
    train_clean = clean_train(raw)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    daily_panel.to_parquet(daily_panel_path)
    sku_activity.to_parquet(sku_activity_path)

    return daily_panel, sku_activity


def find_sunday_cols(cols):
    # cols: iterable of column names (strings)
    # If columns look like F1..F28, use F2,F9,F16,F23 as Sundays
    fcols = [c for c in cols if isinstance(c, str) and c.upper().startswith("F")]
    if len(fcols) >= 28:
        candidates = ["F2", "F9", "F16", "F23"]
        return [c for c in candidates if c in cols]

    # Otherwise try to parse columns as dates
    sunday_cols = []
    for c in cols:
        try:
            dt = pd.to_datetime(c)
            if dt.dayofweek == 6:
                sunday_cols.append(c)
        except Exception:
            continue
    return sunday_cols


def maybe_timestamp_columns(pred):
    # Ensure columns are strings for saving and comparisons
    pred = pred.copy()
    pred.columns = [str(c) for c in pred.columns]
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


def main():
    ensure_dirs()

    # Load sample submission to validate shape and ids
    sample_path = RAW_DIR / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError("Missing sample_submission.csv in data/raw")
    sample = pd.read_csv(sample_path)

    # Load or build processed data
    daily_panel, sku_activity = load_or_build_processed_data()

    # We need configuration variables. Try to read from a config module or final_reproduce
    try:
        from src.final_reproduce import SUB_DIR, FINAL_TRAIN_END, FINAL_FORECAST_START, HORIZON, TARGET_FOR_PRED
    except Exception:
        # Some fallback defaults (may break). Prefer user-provided notebook variables.
        raise ImportError("Required constants (SUB_DIR, FINAL_TRAIN_END, FINAL_FORECAST_START, HORIZON, TARGET_FOR_PRED) not found in src.final_reproduce.\nPlease ensure those are defined or adapt the script.")

    # Build anchor prediction
    print("Building anchor prediction (14,18,21 blend)...")
    anchor_pred = make_recent_blend_prediction(
        daily_panel,
        train_end=FINAL_TRAIN_END,
        forecast_start=FINAL_FORECAST_START,
        horizon=HORIZON,
        target_col=TARGET_FOR_PRED,
        windows=(14, 18, 21),
        weights=(0.20, 0.45, 0.35),
    )
    anchor_pred = postprocess_prediction(
        anchor_pred,
        panel=daily_panel,
        sku_activity=sku_activity,
        train_end=FINAL_TRAIN_END,
        sunday_factor=0.0,
        apply_cap=True,
    )

    # Build recent56 prediction
    print("Building recent56 prediction (56)...")
    recent56_pred = make_recent_blend_prediction(
        daily_panel,
        train_end=FINAL_TRAIN_END,
        forecast_start=FINAL_FORECAST_START,
        horizon=HORIZON,
        target_col=TARGET_FOR_PRED,
        windows=(56,),
        weights=(1.0,),
    )
    recent56_pred = postprocess_prediction(
        recent56_pred,
        panel=daily_panel,
        sku_activity=sku_activity,
        train_end=FINAL_TRAIN_END,
        sunday_factor=0.0,
        apply_cap=True,
    )

    # Normalize columns to strings
    anchor_pred = maybe_timestamp_columns(anchor_pred)
    recent56_pred = maybe_timestamp_columns(recent56_pred)

    # Candidate
    candidate_pred = anchor_pred.copy()

    # Metadata
    meta = sku_activity.set_index("ItemCode").reindex(anchor_pred.index)
    profit_rank = meta["profit_rank"].fillna(999999)
    days_since = meta["days_since_last_sale"].fillna(9999)
    active_days = meta["active_days"].fillna(0)

    # Masks
    mask_top500 = profit_rank <= 500
    mask_501_1000 = (profit_rank > 500) & (profit_rank <= 1000)
    mask_gt1000 = profit_rank > 1000

    # Apply updated candidate rules
    print("Applying new safe rules for rank 501-1000 and >1000...")

    # rank 501-1000
    m = mask_501_1000
    m_ds_le14 = m & (days_since <= 14)
    # keep anchor for ds<=14 (do nothing)
    m_ds_15_56 = m & (days_since > 14) & (days_since <= 56)
    candidate_pred.loc[m_ds_15_56.values, :] = anchor_pred.loc[m_ds_15_56.values, :].values * 0.95
    m_inactive_or_far = m & ((active_days <= 3) | (days_since > 180))
    candidate_pred.loc[m_inactive_or_far.values, :] = anchor_pred.loc[m_inactive_or_far.values, :].values * 0.75
    m_remaining = m & (~m_ds_le14) & (~m_ds_15_56) & (~m_inactive_or_far)
    candidate_pred.loc[m_remaining.values, :] = anchor_pred.loc[m_remaining.values, :].values * 0.85

    # rank >1000
    m = mask_gt1000
    m_ds_le14 = m & (days_since <= 14)
    candidate_pred.loc[m_ds_le14.values, :] = anchor_pred.loc[m_ds_le14.values, :].values * 0.85
    m_ds_le56_active = m & (days_since <= 56) & (active_days > 10) & (~m_ds_le14)
    candidate_pred.loc[m_ds_le56_active.values, :] = anchor_pred.loc[m_ds_le56_active.values, :].values * 0.65
    m_ds_le56 = m & (days_since <= 56) & (~m_ds_le14) & (~m_ds_le56_active)
    candidate_pred.loc[m_ds_le56.values, :] = anchor_pred.loc[m_ds_le56.values, :].values * 0.50
    m_zero = m & (~m_ds_le14) & (~m_ds_le56_active) & (~m_ds_le56)
    candidate_pred.loc[m_zero.values, :] = 0.0

    # Ensure top500 unchanged
    candidate_pred.loc[mask_top500.values, :] = anchor_pred.loc[mask_top500.values, :].values

    # Clip lower bound and set Sundays to zero
    candidate_pred = candidate_pred.clip(lower=0)
    sunday_cols = find_sunday_cols(candidate_pred.columns)
    if sunday_cols:
        candidate_pred.loc[:, sunday_cols] = 0.0

    # Prepare submission file path (avoid overwrite)
    submission_path = OUTPUT_SUB_DIR / SUBMISSION_BASENAME
    if submission_path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = OUTPUT_SUB_DIR / f"submission_PUBLIC_anchor_longtail_soft_recency_v10_all_rows_{ts}.csv"

    # Build submission CSV via helper
    print("Writing submission CSV (not submitting to Kaggle)...")
    submission_df = make_kaggle_submission(pred_56=candidate_pred, sample=sample, output_path=submission_path)

    # QA 1: Submission QA prints
    f_cols = [f"F{i}" for i in range(1, 29)]
    print("--- Submission QA ---")
    print("Shape:", submission_df.shape)
    print("Unique IDs:", submission_df["id"].nunique())
    print("Missing:", submission_df[f_cols].isna().sum().sum())
    print("Negative:", (submission_df[f_cols] < 0).sum().sum())
    print("Total forecast:", submission_df[f_cols].sum().sum())
    print("Max forecast:", submission_df[f_cols].max().max())

    # validation/evaluation split by id suffix
    validation_rows = submission_df["id"].str.endswith("_validation")
    evaluation_rows = submission_df["id"].str.endswith("_evaluation")
    validation_sunday_total = submission_df.loc[validation_rows, [c for c in f_cols if c in submission_df.columns and c in sunday_cols]].sum().sum() if sunday_cols else 0.0
    evaluation_sunday_total = submission_df.loc[evaluation_rows, [c for c in f_cols if c in submission_df.columns and c in sunday_cols]].sum().sum() if sunday_cols else 0.0
    print("Validation Sunday total:", validation_sunday_total)
    print("Evaluation Sunday total:", evaluation_sunday_total)

    # Hard gate: Sundays must be zero
    assert abs(validation_sunday_total) < 1e-8 and abs(evaluation_sunday_total) < 1e-8, "Hard gate failed: Sunday totals are not zero"

    # Assertions
    assert submission_df.shape == sample.shape, "Submission shape mismatch with sample"
    assert submission_df["id"].tolist() == sample["id"].tolist(), "Submission id order mismatch"
    assert submission_df[f_cols].isna().sum().sum() == 0, "Submission contains NaN"
    assert np.isfinite(submission_df[f_cols].values).all(), "Submission contains non-finite values"
    assert (submission_df[f_cols] < 0).sum().sum() == 0, "Submission contains negative forecasts"

    # Delta QA vs anchor
    # Ensure anchor_pred and candidate_pred aligned
    anchor_pred = anchor_pred.reindex(candidate_pred.index).fillna(0)
    delta = candidate_pred - anchor_pred
    abs_delta_by_sku = delta.abs().sum(axis=1)
    changed_skus = abs_delta_by_sku[abs_delta_by_sku > 1e-9].index

    anchor_total = anchor_pred.values.sum()
    candidate_total = candidate_pred.values.sum()
    total_delta = candidate_total - anchor_total
    total_delta_pct = total_delta / (anchor_total + 1e-12)
    max_abs_delta = delta.abs().max().max()

    print("--- Delta vs Anchor ---")
    print("Changed SKU count:", len(changed_skus))
    print("Anchor total:", anchor_total)
    print("Candidate total:", candidate_total)
    print("Total delta:", total_delta)
    print("Total delta pct:", total_delta_pct)
    print("Max abs delta (any cell):", max_abs_delta)

    # Extra warnings (non-fatal)
    if total_delta_pct < -0.10:
        print("WARNING: total_delta_pct < -10% ->", total_delta_pct)

    # Rank bucket diagnostics
    df_rank = pd.DataFrame({
        'ItemCode': anchor_pred.index,
        'profit_rank': profit_rank.values,
        'rank_bucket': profit_rank.apply(assign_rank_bucket).values,
        'anchor_total': anchor_pred.sum(axis=1).values,
        'candidate_total': candidate_pred.sum(axis=1).values,
        'abs_delta': abs_delta_by_sku.values,
    })
    df_rank['delta'] = df_rank['candidate_total'] - df_rank['anchor_total']

    bucket_summary = (
        df_rank.groupby('rank_bucket', as_index=False)
        .agg(
            sku_count=('ItemCode','count'),
            changed_sku_count=('abs_delta', lambda x: int((x>1e-9).sum())),
            anchor_total=('anchor_total','sum'),
            candidate_total=('candidate_total','sum'),
            delta=('delta','sum'),
            max_abs_delta=('abs_delta','max')
        )
    )
    bucket_summary['delta_pct'] = bucket_summary['delta'] / (bucket_summary['anchor_total'] + 1e-12)

    print("--- Delta by Rank Bucket ---")

    # Hard QA gates
    def get_bucket_val(df, bucket, col):
        if bucket in df['rank_bucket'].values:
            return float(df.loc[df['rank_bucket']==bucket, col].iloc[0])
        return 0.0

    b001100 = get_bucket_val(bucket_summary, 'rank_001_100', 'delta')
    b101500 = get_bucket_val(bucket_summary, 'rank_101_500', 'delta')
    max_001_500 = 0.0
    for b in ['rank_001_100', 'rank_101_500']:
        if b in bucket_summary['rank_bucket'].values:
            max_001_500 = max(max_001_500, float(bucket_summary.loc[bucket_summary['rank_bucket']==b, 'max_abs_delta'].iloc[0]))

    if abs(b001100) > 1e-8 or abs(b101500) > 1e-8 or (not np.isnan(max_001_500) and max_001_500 > 1e-8):
        raise ValueError('Hard gate failed: changes detected in rank <= 500 group')

    # Additional hard gate: rank_501_1000 delta_pct must not be lower than -25%
    if 'rank_501_1000' in bucket_summary['rank_bucket'].values:
        pct = float(bucket_summary.loc[bucket_summary['rank_bucket']=='rank_501_1000','delta_pct'].iloc[0])
        if pct < -0.25:
            raise ValueError('Hard gate failed: rank_501_1000 delta_pct < -25%')

    # Save rank diagnostics
    rank_diag_path = DIAG_DIR / 'diagnostics_PUBLIC_anchor_longtail_soft_recency_v10_delta_by_rank_bucket.csv'
    bucket_summary.to_csv(rank_diag_path, index=False)

    # Recency bucket diagnostics
    df_rank['recency_bucket'] = df_rank['profit_rank'].copy()  # placeholder
    # We need days_since aligned for df_rank rows
    ds_series = days_since.reindex(anchor_pred.index)
    df_rank['days_since_last_sale'] = ds_series.values
    df_rank['recency_bucket'] = df_rank['days_since_last_sale'].apply(assign_recency_bucket)

    recency_summary = (
        df_rank.groupby('recency_bucket', as_index=False)
        .agg(
            sku_count=('ItemCode','count'),
            changed_sku_count=('abs_delta', lambda x: int((x>1e-9).sum())),
            anchor_total=('anchor_total','sum'),
            candidate_total=('candidate_total','sum'),
            delta=('delta','sum')
        )
    )
    recency_summary['delta_pct'] = recency_summary['delta'] / (recency_summary['anchor_total'] + 1e-12)
    recency_diag_path = DIAG_DIR / 'diagnostics_PUBLIC_anchor_longtail_soft_recency_v10_delta_by_recency_bucket.csv'
    recency_summary.to_csv(recency_diag_path, index=False)

    print("--- Delta by Recency Bucket ---")
    print(recency_summary)

    # Check recently active long-tail SKUs: profit_rank >1000 & ds <=14
    mask_check = (profit_rank > 1000) & (days_since <= 14)
    group_check = df_rank.loc[mask_check.values, :]
    n_group = len(group_check)
    n_anchor_pos = int((group_check['anchor_total'] > 0).sum())
    n_candidate_zero = int(((group_check['candidate_total'] == 0) & (group_check['anchor_total'] > 0)).sum())
    print("--- Rank>1000 recent sale check ---")
    print("Total SKU in group (profit_rank>1000 & ds<=14):", n_group)
    print("SKUs with anchor_total>0:", n_anchor_pos)
    print("SKUs candidate_total==0 while anchor>0:", n_candidate_zero)
    if n_candidate_zero > 0:
        raise ValueError('Some recently-active long-tail SKUs were zeroed while anchor>0; abort')

    # Top changed SKU table
    top_changed = df_rank.copy()
    meta_df = meta.reset_index()
    if 'ItemCode' not in meta_df.columns:
        meta_df = meta_df.rename(columns={meta_df.columns[0]: 'ItemCode'})
    top_changed = top_changed.merge(meta_df[['ItemCode', 'profit_rank', 'active_days', 'days_since_last_sale']], on='ItemCode', how='left')
    top_changed['abs_delta'] = top_changed['abs_delta']
    top_changed['delta'] = top_changed['candidate_total'] - top_changed['anchor_total']
    top50 = top_changed.sort_values('abs_delta', ascending=False).head(50)
    top_changed_path = DIAG_DIR / 'diagnostics_PUBLIC_anchor_longtail_soft_recency_v10_top_changed_sku.csv'
    top50.to_csv(top_changed_path, index=False)

    # Extra warning: recency <=14 bucket delta_pct too negative
    recency_row = recency_summary.loc[recency_summary['recency_bucket'] == '<=14']
    if not recency_row.empty:
        recency_pct = float(recency_row['delta_pct'].iloc[0])
        if recency_pct < -0.08:
            print('WARNING: recency <=14 delta_pct < -8% ->', recency_pct)

    # Final prints
    print('\nREADY TO REVIEW, NOT AUTO-SUBMITTED')
    print('Submission path:', submission_path)
    print('Diagnostics:')
    print(' -', rank_diag_path)
    print(' -', recency_diag_path)
    print(' -', top_changed_path)


if __name__ == '__main__':
    main()
