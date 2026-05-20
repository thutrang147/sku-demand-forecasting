"""
Final reproducibility script for SKU demand forecasting.

Run from the project root:
    python src/final_reproduce.py

Expected input files:
    data/raw/train.csv
    data/raw/sample_submission.csv

Generated output files:
    outputs/submissions1/submission_recent21_28blend_sunday0.csv
    outputs/submissions1/submission_recent21_sunday0.csv
    outputs/submissions1/submission_recent28_sunday0.csv

Note:
    The output folder is intentionally set to outputs/submissions1 for checking.
    After verifying outputs, change OUTPUT_SUBDIR_NAME from "submissions1" to "submissions"
    if you want the script to write into the final submissions folder.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

# This file is intended to live in: <repo_root>/src/final_reproduce.py
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

# Temporary output folder for checking. Change to "submissions" after verification.
OUTPUT_SUBDIR_NAME = "submissions1"
SUB_DIR = OUTPUTS / OUTPUT_SUBDIR_NAME

TRAIN_PATH = DATA_RAW / "train.csv"
SAMPLE_PATH = DATA_RAW / "sample_submission.csv"

FINAL_TRAIN_END = pd.Timestamp("2025-09-05")
FINAL_FORECAST_START = pd.Timestamp("2025-09-06")
HORIZON = 56
TARGET_FOR_PRED = "y_net_clip"


# ============================================================
# Utility
# ============================================================

def log_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_number_series(s: pd.Series) -> pd.Series:
    """
    Parse numeric columns that may contain comma decimal separators.
    Example: '1277885,86' -> 1277885.86
    """
    if pd.api.types.is_numeric_dtype(s):
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan})
        .astype(float)
    )


# ============================================================
# Data loading and cleaning
# ============================================================

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find train.csv at {TRAIN_PATH}. "
            "Please place train.csv in data/raw/."
        )

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find sample_submission.csv at {SAMPLE_PATH}. "
            "Please place sample_submission.csv in data/raw/."
        )

    train = pd.read_csv(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    print(f"train shape: {train.shape}")
    print(f"sample shape: {sample.shape}")

    required_train_cols = {
        "Date", "Stt", "ItemCode", "Quantity", "UnitPrice",
        "SalesAmount", "Unit Cost", "Cost Amount",
    }
    missing_cols = required_train_cols - set(train.columns)
    if missing_cols:
        raise ValueError(f"train.csv is missing columns: {sorted(missing_cols)}")

    expected_sample_cols = ["id"] + [f"F{i}" for i in range(1, 29)]
    if list(sample.columns) != expected_sample_cols:
        raise ValueError(
            "sample_submission.csv columns are not exactly id,F1,...,F28. "
            f"Actual columns: {sample.columns.tolist()}"
        )

    if sample["id"].duplicated().any():
        raise ValueError("sample_submission.csv contains duplicate id values.")

    return train, sample


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Normal integer / numeric columns.
    for col in ["Quantity", "SalesAmount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columns that may contain comma decimal separators.
    for col in ["UnitPrice", "Unit Cost", "Cost Amount"]:
        df[col] = parse_number_series(df[col])

    required_numeric_cols = [
        "Quantity", "UnitPrice", "SalesAmount", "Unit Cost", "Cost Amount"
    ]
    missing_after_parse = df[required_numeric_cols].isna().sum()

    print("Missing values after numeric parsing:")
    print(missing_after_parse)

    if missing_after_parse.sum() > 0:
        raise ValueError(
            "Numeric parsing produced missing values. Please inspect raw data parsing.\n"
            f"{missing_after_parse}"
        )

    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Number of SKUs: {df['ItemCode'].nunique()}")

    return df


# ============================================================
# Daily panel and SKU activity
# ============================================================

def make_daily_panel(train_clean: pd.DataFrame) -> pd.DataFrame:
    df = train_clean.copy()

    df["qty_positive"] = df["Quantity"].clip(lower=0)
    df["qty_return"] = -df["Quantity"].clip(upper=0)
    df["profit"] = df["SalesAmount"] - df["Cost Amount"]

    daily = (
        df.groupby(["ItemCode", "Date"], as_index=False)
        .agg(
            y_net=("Quantity", "sum"),
            y_gross=("qty_positive", "sum"),
            y_return=("qty_return", "sum"),
            sales=("SalesAmount", "sum"),
            cost=("Cost Amount", "sum"),
            profit=("profit", "sum"),
            transaction_count=("Stt", "count"),
        )
    )

    daily["y_net_clip"] = daily["y_net"].clip(lower=0)

    all_skus = pd.Index(sorted(train_clean["ItemCode"].unique()), name="ItemCode")
    all_dates = pd.date_range(
        train_clean["Date"].min(),
        train_clean["Date"].max(),
        freq="D",
        name="Date",
    )

    full_index = pd.MultiIndex.from_product(
        [all_skus, all_dates], names=["ItemCode", "Date"]
    )

    daily_panel = (
        daily.set_index(["ItemCode", "Date"])
        .reindex(full_index)
        .reset_index()
    )

    fill_zero_cols = [
        "y_net", "y_gross", "y_return", "sales", "cost", "profit",
        "transaction_count", "y_net_clip",
    ]
    for col in fill_zero_cols:
        daily_panel[col] = daily_panel[col].fillna(0)

    daily_panel["dayofweek"] = daily_panel["Date"].dt.dayofweek
    daily_panel["is_saturday"] = (daily_panel["dayofweek"] == 5).astype(int)
    daily_panel["is_sunday"] = (daily_panel["dayofweek"] == 6).astype(int)
    daily_panel["month"] = daily_panel["Date"].dt.month
    daily_panel["day"] = daily_panel["Date"].dt.day
    daily_panel["weekofyear"] = daily_panel["Date"].dt.isocalendar().week.astype(int)
    daily_panel["year"] = daily_panel["Date"].dt.year

    expected_rows = train_clean["ItemCode"].nunique() * (
        (train_clean["Date"].max() - train_clean["Date"].min()).days + 1
    )
    if len(daily_panel) != expected_rows:
        raise ValueError(
            f"Unexpected daily_panel rows: {len(daily_panel)}, expected {expected_rows}"
        )

    print(f"daily_panel shape: {daily_panel.shape}")
    return daily_panel


def make_sku_activity(daily_panel: pd.DataFrame) -> pd.DataFrame:
    max_train_date = daily_panel["Date"].max()

    sku_activity = (
        daily_panel.groupby("ItemCode", as_index=False)
        .agg(
            active_days=("y_gross", lambda x: int((x > 0).sum())),
            active_net_days=("y_net_clip", lambda x: int((x > 0).sum())),
            total_y_net=("y_net", "sum"),
            total_y_gross=("y_gross", "sum"),
            total_return=("y_return", "sum"),
            total_sales=("sales", "sum"),
            total_cost=("cost", "sum"),
            total_profit=("profit", "sum"),
            total_transactions=("transaction_count", "sum"),
        )
    )

    sale_dates = (
        daily_panel[daily_panel["y_gross"] > 0]
        .groupby("ItemCode", as_index=False)
        .agg(first_sale_date=("Date", "min"), last_sale_date=("Date", "max"))
    )

    transaction_dates = (
        daily_panel[daily_panel["transaction_count"] > 0]
        .groupby("ItemCode", as_index=False)
        .agg(
            first_transaction_date=("Date", "min"),
            last_transaction_date=("Date", "max"),
        )
    )

    sku_activity = (
        sku_activity
        .merge(sale_dates, on="ItemCode", how="left")
        .merge(transaction_dates, on="ItemCode", how="left")
    )

    sku_activity["has_ever_sold"] = sku_activity["last_sale_date"].notna().astype(int)

    sku_activity["days_since_last_sale"] = (
        max_train_date - sku_activity["last_sale_date"]
    ).dt.days.fillna(9999).astype(int)

    sku_activity["days_since_last_transaction"] = (
        max_train_date - sku_activity["last_transaction_date"]
    ).dt.days.fillna(9999).astype(int)

    sku_activity["positive_profit"] = sku_activity["total_profit"].clip(lower=0)

    sku_activity["profit_rank"] = (
        sku_activity["positive_profit"].rank(method="min", ascending=False).astype(int)
    )

    sku_activity["return_rate_qty"] = np.where(
        sku_activity["total_y_gross"] > 0,
        sku_activity["total_return"] / sku_activity["total_y_gross"],
        0,
    )

    print(f"sku_activity shape: {sku_activity.shape}")
    print(f"median active_days: {sku_activity['active_days'].median()}")
    print(f"median days_since_last_sale: {sku_activity['days_since_last_sale'].median()}")

    return sku_activity


# ============================================================
# Forecasting
# ============================================================

def make_recent_blend_prediction(
    panel: pd.DataFrame,
    train_end,
    forecast_start,
    horizon: int = 56,
    target_col: str = "y_net_clip",
    windows: tuple[int, ...] = (28,),
    weights: tuple[float, ...] = (1.0,),
) -> pd.DataFrame:
    train_end = pd.Timestamp(train_end)
    forecast_start = pd.Timestamp(forecast_start)
    forecast_dates = pd.date_range(forecast_start, periods=horizon, freq="D")

    itemcodes = sorted(panel["ItemCode"].unique())

    if len(windows) != len(weights):
        raise ValueError("windows and weights must have the same length.")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.")

    blended = pd.Series(0.0, index=itemcodes)

    for window, weight in zip(windows, weights):
        hist_start = train_end - pd.Timedelta(days=window - 1)
        hist = panel.loc[
            (panel["Date"] >= hist_start) & (panel["Date"] <= train_end),
            ["ItemCode", target_col],
        ]
        mean_by_sku = hist.groupby("ItemCode")[target_col].mean()
        blended += weight * mean_by_sku.reindex(itemcodes).fillna(0)

    pred = pd.DataFrame(
        np.repeat(blended.to_numpy()[:, None], horizon, axis=1),
        index=itemcodes,
        columns=forecast_dates,
    )

    return pred


def compute_sku_caps(
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    train_end,
    target_col: str = "y_net_clip",
) -> pd.Series:
    train_end = pd.Timestamp(train_end)
    itemcodes = sorted(panel["ItemCode"].unique())

    hist_365 = panel.loc[
        (panel["Date"] >= train_end - pd.Timedelta(days=364))
        & (panel["Date"] <= train_end),
        ["ItemCode", "Date", target_col],
    ].copy()

    hist_112 = panel.loc[
        (panel["Date"] >= train_end - pd.Timedelta(days=111))
        & (panel["Date"] <= train_end),
        ["ItemCode", "Date", target_col],
    ].copy()

    hist_56 = panel.loc[
        (panel["Date"] >= train_end - pd.Timedelta(days=55))
        & (panel["Date"] <= train_end),
        ["ItemCode", "Date", target_col],
    ].copy()

    positive_hist = hist_365[hist_365[target_col] > 0]

    pos_q95 = positive_hist.groupby("ItemCode")[target_col].quantile(0.95)
    max_56 = hist_56.groupby("ItemCode")[target_col].max()
    mean_112 = hist_112.groupby("ItemCode")[target_col].mean()

    cap = pd.DataFrame(index=itemcodes)
    cap["pos_q95"] = pos_q95.reindex(itemcodes).fillna(0)
    cap["max_56"] = max_56.reindex(itemcodes).fillna(0)
    cap["mean_112"] = mean_112.reindex(itemcodes).fillna(0)

    cap["base_cap"] = np.maximum.reduce(
        [
            cap["pos_q95"] * 2.0,
            cap["max_56"] * 1.2,
            cap["mean_112"] * 5.0,
        ]
    )

    meta = sku_activity.set_index("ItemCode").reindex(itemcodes)

    recently_active = meta["days_since_last_sale"].fillna(9999) <= 56
    cap.loc[recently_active, "base_cap"] = cap.loc[recently_active, "base_cap"].clip(
        lower=1.0
    )

    profit_rank = meta["profit_rank"].fillna(999999)

    cap_multiplier = pd.Series(1.0, index=itemcodes)
    cap_multiplier.loc[profit_rank <= 100] = 3.0
    cap_multiplier.loc[(profit_rank > 100) & (profit_rank <= 500)] = 2.0
    cap_multiplier.loc[(profit_rank > 500) & (profit_rank <= 1000)] = 1.5

    final_cap = cap["base_cap"] * cap_multiplier
    return final_cap.fillna(0)


def postprocess_prediction(
    pred: pd.DataFrame,
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    train_end,
    target_col: str = "y_net_clip",
    sunday_factor: float = 0.0,
    apply_cap: bool = True,
) -> pd.DataFrame:
    pred = pred.copy()
    pred[pred < 0] = 0

    # Sunday correction. Forecast window starts on Saturday, so Sundays are F2/F9/F16/F23
    # in each 28-day submission block.
    sunday_cols = [c for c in pred.columns if pd.Timestamp(c).dayofweek == 6]
    if sunday_cols:
        pred.loc[:, sunday_cols] *= sunday_factor

    meta = sku_activity.set_index("ItemCode").reindex(pred.index)

    days_since = meta["days_since_last_sale"].fillna(9999)
    active_days = meta["active_days"].fillna(0)
    profit_rank = meta["profit_rank"].fillna(999999)

    zero_mask = (
        ((days_since > 365) & (profit_rank > 500))
        | ((days_since > 180) & (active_days <= 3) & (profit_rank > 100))
        | ((days_since > 90) & (active_days <= 1) & (profit_rank > 100))
    )
    pred.loc[zero_mask, :] = 0

    shrink_mask = (
        (days_since > 90)
        & (days_since <= 365)
        & (active_days <= 5)
        & (profit_rank > 500)
        & (~zero_mask)
    )
    pred.loc[shrink_mask, :] *= 0.25

    if apply_cap:
        caps = compute_sku_caps(
            panel=panel,
            sku_activity=sku_activity,
            train_end=train_end,
            target_col=target_col,
        ).reindex(pred.index).fillna(0)
        pred = pred.clip(upper=caps, axis=0)

    pred = pred.fillna(0)
    pred[pred < 0] = 0

    return pred


def build_recent_forecast(
    panel: pd.DataFrame,
    sku_activity: pd.DataFrame,
    windows: tuple[int, ...],
    weights: tuple[float, ...],
    name: str,
) -> pd.DataFrame:
    print(f"Building forecast: {name} | windows={windows}, weights={weights}")

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


# ============================================================
# Submission generation and QA
# ============================================================

def make_kaggle_submission(
    pred_56: pd.DataFrame,
    sample: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    f_cols = [f"F{i}" for i in range(1, 29)]

    if pred_56.shape[1] != 56:
        raise ValueError(f"pred_56 must have 56 forecast columns, got {pred_56.shape[1]}")

    pred = pred_56.copy()
    pred = pred.sort_index()
    pred = pred.fillna(0).clip(lower=0).astype(float)

    validation_part = pred.iloc[:, :28].copy()
    validation_part.columns = f_cols
    validation_part.index = validation_part.index + "_validation"

    evaluation_part = pred.iloc[:, 28:56].copy()
    evaluation_part.columns = f_cols
    evaluation_part.index = evaluation_part.index + "_evaluation"

    pred_sub = (
        pd.concat([validation_part, evaluation_part], axis=0)
        .reset_index()
        .rename(columns={"index": "id"})
    )

    # Preserve exact sample row order.
    sub = sample[["id"]].merge(pred_sub, on="id", how="left", validate="one_to_one")

    assert sub.shape == sample.shape
    assert sub["id"].tolist() == sample["id"].tolist()
    assert sub["id"].nunique() == len(sub)

    values = sub[f_cols].to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError("Submission contains NaN values.")
    if np.isinf(values).any():
        raise ValueError("Submission contains infinite values.")
    if (values < 0).any():
        raise ValueError("Submission contains negative values.")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    return sub


def qa_submission(sub: pd.DataFrame, sample: pd.DataFrame, name: str) -> dict:
    f_cols = [f"F{i}" for i in range(1, 29)]

    validation_rows = sub["id"].str.endswith("_validation")
    evaluation_rows = sub["id"].str.endswith("_evaluation")
    sunday_f = ["F2", "F9", "F16", "F23"]

    summary = {
        "name": name,
        "shape": sub.shape,
        "unique_ids": sub["id"].nunique(),
        "missing": int(sub[f_cols].isna().sum().sum()),
        "negative": int((sub[f_cols] < 0).sum().sum()),
        "total": float(sub[f_cols].sum().sum()),
        "max": float(sub[f_cols].max().max()),
        "validation_sunday_total": float(sub.loc[validation_rows, sunday_f].sum().sum()),
        "evaluation_sunday_total": float(sub.loc[evaluation_rows, sunday_f].sum().sum()),
    }

    print(f"QA for {name}:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    assert sub.shape == sample.shape
    assert sub["id"].tolist() == sample["id"].tolist()
    assert summary["unique_ids"] == len(sample)
    assert summary["missing"] == 0
    assert summary["negative"] == 0
    assert abs(summary["validation_sunday_total"]) < 1e-9
    assert abs(summary["evaluation_sunday_total"]) < 1e-9

    return summary


def compare_with_existing_submission(generated_sub: pd.DataFrame, name: str) -> None:
    """
    Optional check: if outputs/submissions/submission_<name>.csv exists,
    compare the generated file in outputs/submissions1 with that existing file.
    """
    reference_path = OUTPUTS / "submissions" / f"submission_{name}.csv"
    if not reference_path.exists():
        print(f"No existing reference submission found for {name}. Skipping comparison.")
        return

    reference = pd.read_csv(reference_path)
    f_cols = [f"F{i}" for i in range(1, 29)]

    if generated_sub["id"].tolist() != reference["id"].tolist():
        print(f"Reference comparison for {name}: ID order differs.")
        return

    max_abs_diff = np.max(
        np.abs(generated_sub[f_cols].to_numpy(float) - reference[f_cols].to_numpy(float))
    )

    print(f"Reference comparison for {name}: max_abs_diff = {max_abs_diff}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    log_section("1. Load raw data")
    train, sample = load_raw_data()

    log_section("2. Clean train data")
    train_clean = clean_train(train)

    log_section("3. Build daily panel")
    daily_panel = make_daily_panel(train_clean)

    log_section("4. Build SKU activity")
    sku_activity = make_sku_activity(daily_panel)

    log_section("5. Generate final candidate submissions")
    final_configs = [
        {
            "name": "recent21_28blend_sunday0",
            "windows": (21, 28),
            "weights": (0.50, 0.50),
            "public_score": 0.49259,
        },
        {
            "name": "recent21_sunday0",
            "windows": (21,),
            "weights": (1.0,),
            "public_score": 0.49299,
        },
        {
            "name": "recent28_sunday0",
            "windows": (28,),
            "weights": (1.0,),
            "public_score": 0.49339,
        },
    ]

    qa_records: list[dict] = []

    for cfg in final_configs:
        print("\n" + "-" * 80)
        print(
            f"Submission: {cfg['name']} | "
            f"Public score: {cfg['public_score']} | "
            f"windows={cfg['windows']} | weights={cfg['weights']}"
        )

        pred_56 = build_recent_forecast(
            panel=daily_panel,
            sku_activity=sku_activity,
            windows=cfg["windows"],
            weights=cfg["weights"],
            name=cfg["name"],
        )

        output_path = SUB_DIR / f"submission_{cfg['name']}.csv"
        sub = make_kaggle_submission(pred_56=pred_56, sample=sample, output_path=output_path)
        qa_summary = qa_submission(sub=sub, sample=sample, name=cfg["name"])
        qa_summary["public_score"] = cfg["public_score"]
        qa_summary["output_path"] = str(output_path)
        qa_records.append(qa_summary)

        compare_with_existing_submission(sub, cfg["name"])

    log_section("6. Save QA summary")
    qa_df = pd.DataFrame(qa_records)
    qa_path = SUB_DIR / "final_submission_qa_summary.csv"
    qa_df.to_csv(qa_path, index=False)
    print(qa_df)
    print(f"Saved QA summary: {qa_path}")

    log_section("Done")
    print(f"Generated final submissions in: {SUB_DIR}")
    print("If outputs look correct, you can later change OUTPUT_SUBDIR_NAME to 'submissions'.")


if __name__ == "__main__":
    main()
