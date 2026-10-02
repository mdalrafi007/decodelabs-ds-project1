# Cleaning report

Loaded 1200 rows x 14 columns
Duplicate rows: 0 | duplicate OrderIDs: 0
Rows where Quantity*UnitPrice != TotalPrice: 0

## 1. Missing values
CouponCode    25.75
- CouponCode: 25.8% -> categorical & structural; filled 'NONE'

### Imputation experiment on UnitPrice (10% MCAR injected)
scikit-learn not installed -> experiment skipped (pip install scikit-learn)

## 2. Outliers
        col  IQR_low  IQR_high  IQR_out  Z>3  skew
   Quantity    -1.00      7.00        0    0  0.03
  UnitPrice  -317.20   1024.83        0    0 -0.03
ItemsInCart    -0.50     11.50        0    0  0.00
 TotalPrice -1341.41   3330.41        8    0  0.89
- TotalPrice: 8 values capped to [-1341.41, 3330.41]

## 3. Engineered features
Month, DayOfWeek, IsWeekend, HasCoupon, QtyPerCartItem, UnitPriceTier (tertiles), CustomerOrderCount

## 4. Collinearity (|r| > 0.80, drop the weaker link to target)
- r=1.00: CouponCode_NONE vs HasCoupon -> drop HasCoupon (|r| to target 0.01 vs 0.01)
- r=0.82: UnitPriceTier_low vs UnitPrice -> drop UnitPriceTier_low (|r| to target 0.59 vs 0.72)

## 5. Output
Validation passed. Final shape: (1200, 32)