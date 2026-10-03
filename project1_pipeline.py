#!/usr/bin/env python3
"""
DecodeLabs Data Science - Project 1: Advanced EDA & Feature Engineering
Input  : Dataset_for_Data_Analytics.xlsx  (E-commerce orders, 1200 rows)
Output : cleaned_dataset.csv (ML-ready), cleaning_report.md

Usage  : python project1_pipeline.py [path/to/Dataset_for_Data_Analytics.xlsx]
Needs  : pandas, numpy, openpyxl  
"""
import sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "Dataset_for_Data_Analytics.xlsx"
TARGET = "TotalPrice"          # assumption: predict order value
ID_COLS = ["OrderID", "CustomerID", "TrackingNumber", "ShippingAddress"]
CAT_COLS = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
log = []                       # lines for the markdown report


def say(msg=""):
    print(msg)
    log.append(msg)


# ---------------------------------------------------------------- 1. LOAD
df = pd.read_excel(SRC)
raw = df.copy()
say(f"# Cleaning report\n\nLoaded {df.shape[0]} rows x {df.shape[1]} columns")
say(f"Duplicate rows: {df.duplicated().sum()} | duplicate OrderIDs: {df.OrderID.duplicated().sum()}")
mism = (~np.isclose(df.Quantity * df.UnitPrice, df.TotalPrice, atol=0.02)).sum()
say(f"Rows where Quantity*UnitPrice != TotalPrice: {mism}")

# ------------------------------------------------- 2. MISSING VALUES (matrix)
say("\n## 1. Missing values")
miss = df.isna().mean() * 100
say(miss[miss > 0].round(2).to_string() if (miss > 0).any() else "none")


def impute(frame):
    """Apply the PDF decision matrix per column."""
    frame = frame.copy()
    for col in frame.columns[frame.isna().any()]:
        pct = frame[col].isna().mean() * 100
        numeric = pd.api.types.is_numeric_dtype(frame[col])
        if pct < 5:
            frame = frame.dropna(subset=[col])
            say(f"- {col}: {pct:.1f}% -> drop rows")
        elif pct <= 20 and numeric:
            frame[col] = frame[col].fillna(frame[col].median())
            say(f"- {col}: {pct:.1f}% -> global median")
        elif pct <= 20:
            frame[col] = frame[col].fillna(frame[col].mode()[0])
            say(f"- {col}: {pct:.1f}% -> mode")
        elif numeric:
            say(f"- {col}: {pct:.1f}% -> KNN (see knn_impute)")
        else:
            # >20% but categorical: KNN has no meaning for a nominal label.
            # Here NaN = "no coupon used" (structural), so make it explicit.
            frame[col] = frame[col].fillna("NONE")
            say(f"- {col}: {pct:.1f}% -> categorical & structural; filled 'NONE'")
    return frame


df = impute(df)

# ------- Mean vs Median vs KNN experiment (as the PDF conclusion suggests)
say("\n### Imputation experiment on UnitPrice (10% MCAR injected)")
rng = np.random.default_rng(42)
num = df[["Quantity", "ItemsInCart", "UnitPrice"]].astype(float)
truth = num["UnitPrice"].copy()
hole = rng.random(len(num)) < 0.10
holed = num.copy()
holed.loc[hole, "UnitPrice"] = np.nan
cands = {
    "mean": np.full(hole.sum(), holed["UnitPrice"].mean()),
    "median": np.full(hole.sum(), holed["UnitPrice"].median()),
}
# KNN imputation in plain NumPy (no scikit-learn needed, works in Termux):
# distance on standardized Quantity & ItemsInCart, average UnitPrice of k=5 neighbours.
feat = ((num[["Quantity", "ItemsInCart"]] - num[["Quantity", "ItemsInCart"]].mean())
        / num[["Quantity", "ItemsInCart"]].std()).to_numpy()
known = np.where(~hole)[0]
knn_vals = []
for i in np.where(hole)[0]:       # only the ~120 gap rows are looped
    d = np.sqrt(((feat[known] - feat[i]) ** 2).sum(axis=1))
    knn_vals.append(truth.to_numpy()[known[np.argsort(d, kind="stable")[:5]]].mean())
cands["knn(k=5)"] = np.array(knn_vals)
say("| method | RMSE | std after imputing (true std %.1f) |" % truth.std())
say("|---|---|---|")
for name, vals in cands.items():
    filled = truth.copy()
    filled[hole] = vals
    rmse = np.sqrt(np.mean((vals - truth[hole]) ** 2))
    say(f"| {name} | {rmse:.1f} | {filled.std():.1f} |")

# ------------------------------------------------------ 3. OUTLIERS (IQR/Z)
say("\n## 2. Outliers")
num_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
rows = []
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = ((df[c] - df[c].mean()) / df[c].std()).abs()
    rows.append((c, round(lo, 2), round(hi, 2),
                 int(((df[c] < lo) | (df[c] > hi)).sum()), int((z > 3).sum()),
                 round(df[c].skew(), 2)))
say(pd.DataFrame(rows, columns=["col", "IQR_low", "IQR_high", "IQR_out", "Z>3", "skew"])
    .to_string(index=False))

# Winsorize (np.clip keeps row count). Flag first so the information survives.
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag = (df[c] < lo) | (df[c] > hi)
    if flag.any():
        df[f"Is{c}Outlier"] = flag.astype(int)
        df[c] = np.clip(df[c], lo, hi)
        say(f"- {c}: {int(flag.sum())} values capped to [{lo:.2f}, {hi:.2f}]")

# ------------------------------------------------- 4. FEATURE ENGINEERING
say("\n## 3. Engineered features")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df.Date.dt.month                                    # 1
df["DayOfWeek"] = df.Date.dt.dayofweek                            # 2
df["IsWeekend"] = (df.DayOfWeek >= 5).astype(int)                 # 3
df["HasCoupon"] = (df["CouponCode"] != "NONE").astype(int)        # 4
df["QtyPerCartItem"] = df.Quantity / df.ItemsInCart               # 5
df["UnitPriceTier"] = pd.qcut(df.UnitPrice, 3, labels=["low", "mid", "high"])  # 6
df["CustomerOrderCount"] = df.groupby("CustomerID")["OrderID"].transform("count")  # 7
say("Month, DayOfWeek, IsWeekend, HasCoupon, QtyPerCartItem, "
    "UnitPriceTier (tertiles), CustomerOrderCount")

# ----------------------------------------------------------- 5. ENCODING
df = df.drop(columns=ID_COLS + ["Date"])      # identifiers / raw date: no signal
df["UnitPriceTier"] = df["UnitPriceTier"].astype(str)
df = pd.get_dummies(df, columns=CAT_COLS + ["UnitPriceTier"],
                    drop_first=True, dtype=int)

# --------------------------------------------- 6. COLLINEARITY ERADICATION
say("\n## 4. Collinearity (|r| > 0.80, drop the weaker link to target)")
feats = df.drop(columns=[TARGET])
corr = feats.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
target_r = df.corr()[TARGET].abs()
dropped = set()
for a in upper.columns:
    for b in upper.index:
        v = upper.loc[b, a]
        if pd.notna(v) and v > 0.80 and a not in dropped and b not in dropped:
            loser = a if target_r[a] < target_r[b] else b
            dropped.add(loser)
            say(f"- r={v:.2f}: {a} vs {b} -> drop {loser} "
                f"(|r| to target {target_r[a]:.2f} vs {target_r[b]:.2f})")
df = df.drop(columns=list(dropped))
if not dropped:
    say("none found")

# ------------------------------------------------------ 7. VALIDATION
assert df.isna().sum().sum() == 0, "NaNs remain"
assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes), "non-numeric col"
assert (df[TARGET] > 0).all(), "non-positive target"
say(f"\n## 5. Output\nValidation passed. Final shape: {df.shape}")

df.to_csv("cleaned_dataset.csv", index=False)
with open("cleaning_report.md", "w") as f:
    f.write("\n".join(log))
print("\nSaved cleaned_dataset.csv and cleaning_report.md")
