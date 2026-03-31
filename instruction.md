# Toronto Shelter Notebook — Student Build Instructions

Use this guide to recreate the full analysis and modeling workflow from scratch. Work through each section in order.

## Learning Goals
- Load and inspect shelter usage data with pandas.
- Clean raw columns into model-ready features.
- Create a safe occupancy-rate target variable.
- Build a robust ML pipeline with preprocessing + XGBoost.
- Tune model hyperparameters and save the best model artifact.

## Before You Start
1. Place `TorontoShelterUsage2020_Final.csv` in the same folder as this notebook.
2. Install required packages if needed:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `joblib`
3. Use Python 3.10+ if possible for package compatibility.

## Cell 1 — Load and Preview Data
**Purpose:** Verify that the source file loads correctly and understand the raw structure.

**What to write:**
- Import `pandas as pd`.
- Read the CSV into `shelter_df` with `pd.read_csv("TorontoShelterUsage2020_Final.csv")`.
- Display `shelter_df.head()` to inspect the first rows.

**Checks:**
- Confirm no file-not-found error.
- Confirm columns look like the source system export (many raw fields, including shelter/location fields).

## Cell 2 — Clean Data and Engineer Target
**Purpose:** Build a cleaned dataset and create a model target (`OCCUPANCY_Rate`).

**What to write and why:**
1. Reload the CSV (keeps this cell reproducible by itself).
2. Keep columns from index 5 onward using `.iloc[:, 5:]` to remove leading metadata columns.
3. Drop location columns not needed for modeling:
   - `SHELTER_CITY`
   - `SHELTER_PROVINCE`
   - `SHELTER_POSTAL_CODE`
4. Convert `AREA` to numeric safely:
   - Cast to string, extract numeric part using regex, then `pd.to_numeric(..., errors="coerce")`.
   - This handles mixed text/number values without crashing.
5. Standardize column names:
   - Replace non-alphanumeric characters with `_`.
   - Strip leading/trailing underscores.
6. Create target feature:
   - `OCCUPANCY_Rate = OCCUPANCY / CAPACITY`
   - Use a condition so rows with `CAPACITY == 0` become `NaN` (avoids divide-by-zero).
7. Drop the original `OCCUPANCY` column after target creation.
8. Print head of cleaned frame and export to `TorontoShelterUsage2020_Cleaned.csv`.

**Checks:**
- `AREA` should be numeric with invalid values converted to `NaN`.
- `OCCUPANCY_Rate` should exist and be numeric.
- Cleaned CSV should be written to disk.

## Cell 3 — Build and Tune the Model
**Purpose:** Train a high-quality regressor for occupancy rate using proper preprocessing and hyperparameter search.

**Important:** Add all required imports before modeling logic. Include:
- `numpy as np`
- `from sklearn.model_selection import train_test_split, RandomizedSearchCV`
- `from sklearn.compose import ColumnTransformer`
- `from sklearn.pipeline import Pipeline`
- `from sklearn.impute import SimpleImputer`
- `from sklearn.preprocessing import OneHotEncoder`
- `from sklearn.metrics import mean_squared_error, r2_score`
- `from xgboost import XGBRegressor`
- `import joblib`

**Model workflow:**
1. Create `model_df_tuned` by replacing infinities with `NaN` and dropping rows where target is missing.
2. Optional feature engineering: if both `Max_Temp_C` and `Min_Temp_C` exist, create `Temp_Range`.
3. Split predictors/target:
   - `X_tuned = model_df_tuned.drop(columns=["OCCUPANCY_Rate"])`
   - `y_tuned = model_df_tuned["OCCUPANCY_Rate"]`
4. Train/test split with `test_size=0.2` and `random_state=42`.
5. Detect categorical vs numeric columns from `X_tuned`.
6. Build preprocessing:
   - Categorical: most-frequent imputer + one-hot encoding (`handle_unknown="ignore"`, `min_frequency=10`).
   - Numeric: median imputer.
7. Build a full `Pipeline(preprocessor -> XGBRegressor)`.
8. Define `param_dist` for randomized hyperparameter tuning.
9. Run `RandomizedSearchCV` with:
   - `n_iter=30`, `cv=4`, `scoring="r2"`, `n_jobs=-1`, `random_state=42`.
10. Fit on training data, predict test set, compute RMSE and \(R^2\).
11. Print best params and metrics.
12. Save best estimator to `best_xgboost_occupancy_rate_model_tuned.joblib`.

**Checks:**
- Search completes without import/name errors.
- `best_params_` prints a parameter set.
- RMSE and \(R^2\) print successfully.
- Joblib file exists after execution.

## Suggested Run Order
1. Run Cell 1 and confirm data loads.
2. Run Cell 2 and confirm cleaned output file is created.
3. Run Cell 3 and confirm training, metrics, and model export.

## Common Troubleshooting
- **`NameError: np is not defined`** → Add `import numpy as np`.
- **`NameError` for sklearn/xgboost classes** → Add missing imports listed above.
- **File not found** → Ensure CSV is in the notebook folder or provide full path.
- **Slow tuning** → Reduce `n_iter` temporarily (e.g., 10) for quick tests.

## Deliverables for Students
- A cleaned dataset file: `TorontoShelterUsage2020_Cleaned.csv`.
- A tuned model file: `best_xgboost_occupancy_rate_model_tuned.joblib`.
- Printed model quality metrics (RMSE and \(R^2\)).