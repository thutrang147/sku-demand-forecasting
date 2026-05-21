from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from final_reproduce import (
    ROOT,
    load_raw_data,
    clean_train,
    make_daily_panel,
    make_sku_activity,
    build_recent_forecast,
    make_kaggle_submission,
)

# ============================================================
# CONFIG
# ============================================================

# Current best public candidate đang thấy trên Kaggle: 0.49179
ANCHOR_NAME = "recent14_18_21_w20_45_35_sunday0"
ANCHOR_WINDOWS = (14, 18, 21)
ANCHOR_WEIGHTS = (0.20, 0.45, 0.35)

# Private hedge candidate: nhẹ hơn, thêm 28/56 để đỡ overfit public recent noise
PRIVATE_NAME = "recent18_21_28_56_private_safe"
PRIVATE_WINDOWS = (18, 21, 28, 56)
PRIVATE_WEIGHTS = (0.30, 0.45, 0.20, 0.05)

OUT_DIR = ROOT / "outputs" / "submissions"
AUDIT_DIR = ROOT / "outputs" / "reproduce_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = (
    f"submission_PUBLIC_ANCHOR_{ANCHOR_NAME}"
    f"__eval_{PRIVATE_NAME}.csv"
)
OUT_PATH = OUT_DIR / OUT_NAME

F_COLS = [f"F{i}" for i in range(1, 29)]


# ============================================================
# QA helpers
# ============================================================

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_submission(sub: pd.DataFrame, name: str) -> dict:
    val_mask = sub["id"].str.endswith("_validation")
    eval_mask = sub["id"].str.endswith("_evaluation")
    sunday_cols = ["F2", "F9", "F16", "F23"]

    summary = {
        "name": name,
        "rows": len(sub),
        "cols": sub.shape[1],
        "unique_ids": sub["id"].nunique(),
        "missing": int(sub[F_COLS].isna().sum().sum()),
        "negative": int((sub[F_COLS] < 0).sum().sum()),
        "total": float(sub[F_COLS].sum().sum()),
        "validation_total": float(sub.loc[val_mask, F_COLS].sum().sum()),
        "evaluation_total": float(sub.loc[eval_mask, F_COLS].sum().sum()),
        "max": float(sub[F_COLS].max().max()),
        "validation_sunday_total": float(sub.loc[val_mask, sunday_cols].sum().sum()),
        "evaluation_sunday_total": float(sub.loc[eval_mask, sunday_cols].sum().sum()),
    }

    print("\n" + "=" * 80)
    print(f"QA: {name}")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    assert summary["rows"] == 31944, "Unexpected row count"
    assert summary["cols"] == 29, "Unexpected column count"
    assert summary["unique_ids"] == 31944, "Duplicate IDs exist"
    assert summary["missing"] == 0, "Submission has missing values"
    assert summary["negative"] == 0, "Submission has negative values"
    assert abs(summary["validation_sunday_total"]) < 1e-9, "Validation Sunday not zero"
    assert abs(summary["evaluation_sunday_total"]) < 1e-9, "Evaluation Sunday not zero"

    return summary


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, rows_mask=None) -> float:
    if rows_mask is None:
        rows_mask = np.ones(len(a), dtype=bool)

    a2 = a.loc[rows_mask, F_COLS].to_numpy(float)
    b2 = b.loc[rows_mask, F_COLS].to_numpy(float)
    return float(np.max(np.abs(a2 - b2)))


# ============================================================
# Main
# ============================================================

def main():
    print("ROOT:", ROOT)

    # 1. Load + rebuild from raw data
    train, sample = load_raw_data()
    train_clean = clean_train(train)
    daily_panel = make_daily_panel(train_clean)
    sku_activity = make_sku_activity(daily_panel)

    # 2. Generate anchor forecast from code
    print("\nBuilding anchor forecast...")
    anchor_pred = build_recent_forecast(
        panel=daily_panel,
        sku_activity=sku_activity,
        windows=ANCHOR_WINDOWS,
        weights=ANCHOR_WEIGHTS,
        name=ANCHOR_NAME,
    )
    anchor_sub = make_kaggle_submission(anchor_pred, sample)

    # 3. Generate private hedge forecast from code
    print("\nBuilding private hedge forecast...")
    private_pred = build_recent_forecast(
        panel=daily_panel,
        sku_activity=sku_activity,
        windows=PRIVATE_WINDOWS,
        weights=PRIVATE_WEIGHTS,
        name=PRIVATE_NAME,
    )
    private_sub = make_kaggle_submission(private_pred, sample)

    # 4. Combine:
    #    - validation/public = anchor
    #    - evaluation/private = private hedge
    print("\nCombining validation anchor + evaluation hedge...")

    out = anchor_sub.set_index("id").copy()
    private_idx = private_sub.set_index("id")

    eval_ids = [x for x in out.index if x.endswith("_evaluation")]
    out.loc[eval_ids, F_COLS] = private_idx.loc[eval_ids, F_COLS].values

    # Preserve exact Kaggle sample order
    out = out.loc[sample["id"].tolist()].reset_index()

    # 5. QA
    anchor_summary = summarize_submission(anchor_sub, f"generated_{ANCHOR_NAME}")
    private_summary = summarize_submission(private_sub, f"generated_{PRIVATE_NAME}")
    out_summary = summarize_submission(out, OUT_NAME)

    val_mask = out["id"].str.endswith("_validation")
    eval_mask = out["id"].str.endswith("_evaluation")

    val_diff_vs_anchor = max_abs_diff(out, anchor_sub, val_mask)
    eval_diff_vs_private = max_abs_diff(out, private_sub, eval_mask)

    print("\n" + "=" * 80)
    print("ANCHOR / HEDGE CONSISTENCY CHECK")
    print("=" * 80)
    print("validation max_abs_diff vs anchor:", val_diff_vs_anchor)
    print("evaluation max_abs_diff vs private:", eval_diff_vs_private)

    assert val_diff_vs_anchor == 0.0, "Validation part is not identical to anchor."
    assert eval_diff_vs_private == 0.0, "Evaluation part is not identical to private hedge."

    # 6. Optional: compare generated anchor with existing current-best file
    existing_anchor_path = OUT_DIR / f"submission_{ANCHOR_NAME}.csv"
    if existing_anchor_path.exists():
        existing_anchor = pd.read_csv(existing_anchor_path)

        if existing_anchor["id"].tolist() == anchor_sub["id"].tolist():
            diff_existing = max_abs_diff(anchor_sub, existing_anchor)
            print("\nExisting anchor file found:", existing_anchor_path)
            print("generated anchor vs existing anchor max_abs_diff:", diff_existing)
        else:
            print("\nWARNING: Existing anchor id order differs from generated anchor.")
    else:
        print("\nNo existing anchor file found. Skipping existing-anchor comparison.")

    # 7. Save files for audit
    anchor_audit_path = AUDIT_DIR / f"generated_submission_{ANCHOR_NAME}.csv"
    private_audit_path = AUDIT_DIR / f"generated_submission_{PRIVATE_NAME}.csv"

    anchor_sub.to_csv(anchor_audit_path, index=False)
    private_sub.to_csv(private_audit_path, index=False)
    out.to_csv(OUT_PATH, index=False)

    qa_df = pd.DataFrame([anchor_summary, private_summary, out_summary])
    qa_path = AUDIT_DIR / "anchor_private_hedge_qa.csv"
    qa_df.to_csv(qa_path, index=False)

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print("Final submission:", OUT_PATH)
    print("QA summary:", qa_path)
    print("SHA256:", file_sha256(OUT_PATH))

    print("\nExpected key QA for final file:")
    print("rows = 31944")
    print("missing = 0")
    print("negative = 0")
    print("validation_total ≈ 29907.4")
    print("evaluation_total ≈ 30665.992857")
    print("total ≈ 60573.392857")


if __name__ == "__main__":
    main()