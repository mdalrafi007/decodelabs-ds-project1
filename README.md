# Advanced EDA & Feature Engineering: E-commerce Orders

DecodeLabs Data Science Internship, Project 1 (Batch 2026).

A single Python script that turns a raw e-commerce orders table into a
mathematically clean, ML-ready dataset, using rule-based statistical logic
rather than guesswork.

## Dataset
`Dataset_for_Data_Analytics.xlsx`: 1,200 orders x 14 columns (order ID, date,
customer, product, quantity, unit price, payment method, order status,
coupon code, referral source, total price, etc.).
Assumed modelling target: `TotalPrice`.

## What the pipeline does
| Step | Method | Result on this data |
|---|---|---|
| Missing values | PDF decision matrix (<5% drop, 5-20% median/mode, >20% KNN) | Only `CouponCode` is missing (25.8%). It is categorical and means "no coupon used", so it is filled with `NONE` rather than KNN |
| Imputation experiment | Mean vs median vs KNN on 10% artificial gaps in `UnitPrice` | Mean/median (RMSE ~199) beat KNN (~218): `UnitPrice` is uncorrelated with the other columns, so KNN has nothing to learn from |
| Outliers | IQR (1.5x) and Z-score (>3) | 8 `TotalPrice` outliers by IQR, 0 by Z-score. Winsorized with `np.clip` (no rows lost) and flagged in `IsTotalPriceOutlier` |
| Feature engineering | 7 new features | `Month`, `DayOfWeek`, `IsWeekend`, `HasCoupon`, `QtyPerCartItem`, `UnitPriceTier`, `CustomerOrderCount` |
| Encoding | One-hot (`drop_first=True`) | Avoids false ordering between categories |
| Collinearity | Drop one of each pair with abs(r) > 0.80, keeping the one more correlated with the target | Dropped `HasCoupon` (duplicate of `CouponCode_NONE`) and `UnitPriceTier_low` |
| Validation | Assertions: no NaN, all numeric, target > 0 | Passed, final shape 1200 x 32 |

All transformations are vectorized (pandas/NumPy), with no row-by-row loops.

## Documentation summary (project requirements)
**Missing value strategy.** Rules from the course matrix are coded in `impute()`
(drop / mean-median / KNN by missingness %). On this dataset only `CouponCode`
is missing and it is categorical, so it is filled with `NONE`. Mean, median and
KNN imputation are then compared on `UnitPrice` with 10% artificial gaps:

| method | RMSE | std after imputing (true 197.2) |
|---|---|---|
| mean | 199.2 | 187.6 |
| median | 199.1 | 187.6 |
| KNN (k=5) | 217.6 | 189.5 |

**Outlier strategy.** IQR (1.5x) and Z-score (>3) are both computed for every
numeric column. Outliers are winsorized (`np.clip`) to the IQR bounds, so no
rows are removed, and an `IsTotalPriceOutlier` flag is kept.

**New features.** `Month`, `DayOfWeek`, `IsWeekend`, `HasCoupon` (later dropped
as collinear), `QtyPerCartItem`, `UnitPriceTier`, `CustomerOrderCount`.

**Final cleaned dataset.** `cleaned_dataset.csv`: 1,200 rows x 32 columns, no
missing values, all numeric (one-hot encoded), ready for modelling.

## Run it
```bash
pip install pandas numpy openpyxl
python project1_pipeline.py Dataset_for_Data_Analytics.xlsx
```
Outputs: `cleaned_dataset.csv` and `cleaning_report.md`.

## Notes and limitations
- `TotalPrice` equals `Quantity x UnitPrice` in every row, so predicting it from
  those two columns is trivial. A real project would pick another target.
- The 8 capped orders look like genuine large purchases, so they are flagged,
  not deleted.
- Pandera schemas and a Feast feature store (mentioned in the course slides)
  are out of scope here; simple assertions are used instead.

## Built with
Python, pandas, NumPy (KNN imputation written in plain NumPy). Developed entirely on a phone
using Termux.
