import pandas as pd
import numpy as np
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import (
    LassoCV,
    RidgeCV,
    ElasticNetCV,
    Lasso,
    Ridge,
    ElasticNet,
    LinearRegression
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

# =====================================================================
# Step 1: API Setup and Download
# =====================================================================

api_key = "KGAT_0c71a4e814c24267c8b14f00f5e6f6e1"
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(api_key)

api = KaggleApi()
api.authenticate()

COMPETITION_ID = "ecom-90025-2026-sm-2-ada-assignment-and-practice"
DESTINATION_PATH = "./data"
os.makedirs(DESTINATION_PATH, exist_ok=True)

print(f"Starting download for competition: {COMPETITION_ID}")
api.competition_download_files(competition=COMPETITION_ID, path=DESTINATION_PATH)
print("Download completed successfully.")

zip_filepath = os.path.join(DESTINATION_PATH, f"{COMPETITION_ID}.zip")
with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
    zip_ref.extractall(DESTINATION_PATH)
os.remove(zip_filepath)
print(f"Files extracted successfully to '{DESTINATION_PATH}' directory.")

# =====================================================================
# Step 2: Data Loading & Outlier Removal
# =====================================================================

train = pd.read_csv(f"{DESTINATION_PATH}/train_data.csv")
test = pd.read_csv(f"{DESTINATION_PATH}/test_data.csv")

X = train.drop(columns=["Y", "ID"])
y = train["Y"]
X_test = test.drop(columns=["ID"])

print("\nOriginal Train shape:", X.shape)

# Outlier removal
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

mask = (y >= lower) & (y <= upper)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print("Train shape after outlier removal:", X.shape)
print("Test shape:", X_test.shape)

# =====================================================================
# Step 3: Polynomial transformation & Standardisation
# =====================================================================

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

print(f"\nOriginal feature size: {X.shape[1]}")
print(f"Transformed feature size: {X_poly.shape[1]}")

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# =====================================================================
# Step 4: Parameter Tuning
# =====================================================================
print("\n===== Parameter Tuning =====")

lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42, n_jobs=-1)
lasso_cv.fit(X_poly_scaled, y)
best_lasso_alpha = lasso_cv.alpha_
print("LASSO alpha:", best_lasso_alpha)

ridge_alphas = np.logspace(-3, 3, 20)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_poly_scaled, y)
best_ridge_alpha = ridge_cv.alpha_
print("Ridge alpha:", best_ridge_alpha)

elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=np.logspace(-3, 1, 20),
    cv=5,
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)
elastic_cv.fit(X_poly_scaled, y)
best_elastic_alpha = elastic_cv.alpha_
best_elastic_l1 = elastic_cv.l1_ratio_
print("Elastic Net alpha:", best_elastic_alpha)
print("Elastic Net l1 ratio:", best_elastic_l1)

# =====================================================================
# Step 5: Fix tuned parameters for out-of-fold prediction
# =====================================================================

lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000, random_state=42)
ridge = Ridge(alpha=best_ridge_alpha, random_state=42)
elastic = ElasticNet(alpha=best_elastic_alpha, l1_ratio=best_elastic_l1, max_iter=10000, random_state=42)

# =====================================================================
# Step 6: Define KFold and Out-of-Fold predictions (Base Models)
# =====================================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n===== Base Model Out-of-Fold Predictions =====")

# 1. LASSO
oof_lasso = cross_val_predict(lasso, X_poly_scaled, y, cv=kf, n_jobs=-1)
mse_lasso = mean_squared_error(y, oof_lasso)
print("Model 1: LASSO CV MSE:", mse_lasso)

# Ridge and Elastic Net for ensembles
oof_ridge = cross_val_predict(ridge, X_poly_scaled, y, cv=kf, n_jobs=-1)
oof_elastic = cross_val_predict(elastic, X_poly_scaled, y, cv=kf, n_jobs=-1)

# Variable Selection & Selected OLS
lasso.fit(X_poly_scaled, y)
selected_idx = np.where(lasso.coef_ != 0)[0]
print(f"Number of LASSO-selected features: {len(selected_idx)}")

X_selected = X_poly_scaled[:, selected_idx]
X_test_selected = X_test_poly_scaled[:, selected_idx]

ols_selected = LinearRegression()
oof_ols = cross_val_predict(ols_selected, X_selected, y, cv=kf, n_jobs=-1)

# =====================================================================
# Step 7: Ensembles (Models 2 to 7)
# =====================================================================
print("\n===== Ensemble Out-of-Fold Predictions =====")

# Model 2: Simple Avg (LASSO + Ridge + EN)
oof_average_1 = (oof_lasso + oof_ridge + oof_elastic) / 3
mse_average_1 = mean_squared_error(y, oof_average_1)
print("Model 2: Simple Avg (LASSO+Ridge+EN) CV MSE:", mse_average_1)

# Model 3: Stacking OLS (LASSO + Ridge + EN)
X_stack_1 = np.column_stack([oof_lasso, oof_ridge, oof_elastic])
stack_regression_1 = LinearRegression()
oof_stack_1 = cross_val_predict(stack_regression_1, X_stack_1, y, cv=kf, n_jobs=-1)
mse_stack_1 = mean_squared_error(y, oof_stack_1)
print("Model 3: Stacking OLS (LASSO+Ridge+EN) CV MSE:", mse_stack_1)

# Model 4: Simple Avg (Selected OLS + LASSO + EN)
oof_average_2 = (oof_ols + oof_lasso + oof_elastic) / 3
mse_average_2 = mean_squared_error(y, oof_average_2)
print("Model 4: Simple Avg (Selected OLS+LASSO+EN) CV MSE:", mse_average_2)

# Model 5: Stacking OLS (Selected OLS + LASSO + EN)
X_stack_2 = np.column_stack([oof_ols, oof_lasso, oof_elastic])
stack_regression_2 = LinearRegression()
oof_stack_2 = cross_val_predict(stack_regression_2, X_stack_2, y, cv=kf, n_jobs=-1)
mse_stack_2 = mean_squared_error(y, oof_stack_2)
print("Model 5: Stacking OLS (Selected OLS+LASSO+EN) CV MSE:", mse_stack_2)

# Model 6: Stacking Ridge (LASSO + Ridge + EN)
stack_ridge_meta_1 = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5, scoring="neg_mean_squared_error")
oof_stack_6 = cross_val_predict(stack_ridge_meta_1, X_stack_1, y, cv=kf, n_jobs=-1)
mse_stack_6 = mean_squared_error(y, oof_stack_6)
print("Model 6: Stacking Ridge (LASSO+Ridge+EN) CV MSE:", mse_stack_6)

# Model 7: Stacking Ridge (Selected OLS + LASSO + EN)
stack_ridge_meta_2 = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5, scoring="neg_mean_squared_error")
oof_stack_7 = cross_val_predict(stack_ridge_meta_2, X_stack_2, y, cv=kf, n_jobs=-1)
mse_stack_7 = mean_squared_error(y, oof_stack_7)
print("Model 7: Stacking Ridge (Selected OLS+LASSO+EN) CV MSE:", mse_stack_7)


# =====================================================================
# Step 8: Full Model Comparison Table
# =====================================================================

results = pd.DataFrame({
    "Model": [
        "Model 1: LASSO Baseline",
        "Model 2: Simple Avg (LASSO+Ridge+EN)",
        "Model 3: Stacking OLS (LASSO+Ridge+EN)",
        "Model 4: Simple Avg (Selected OLS+LASSO+EN)",
        "Model 5: Stacking OLS (Selected OLS+LASSO+EN)",
        "Model 6: Stacking Ridge (LASSO+Ridge+EN)",
        "Model 7: Stacking Ridge (Selected OLS+LASSO+EN)"
    ],
    "CV MSE": [
        mse_lasso,
        mse_average_1,
        mse_stack_1,
        mse_average_2,
        mse_stack_2,
        mse_stack_6,
        mse_stack_7
    ]
})

results = results.sort_values(by="CV MSE").reset_index(drop=True)

print("\n===== Model Comparison (7 Models) =====")
print(results.to_string(index=False))

# =====================================================================
# Step 9: Fit final models on full training sample
# =====================================================================

ridge.fit(X_poly_scaled, y)
elastic.fit(X_poly_scaled, y)
ols_selected.fit(X_selected, y)

stack_regression_1.fit(X_stack_1, y)
stack_regression_2.fit(X_stack_2, y)
stack_ridge_meta_1.fit(X_stack_1, y)
stack_ridge_meta_2.fit(X_stack_2, y)

# =====================================================================
# Step 10: Predict Kaggle test sample
# =====================================================================

pred_lasso_test = lasso.predict(X_test_poly_scaled)
pred_ridge_test = ridge.predict(X_test_poly_scaled)
pred_elastic_test = elastic.predict(X_test_poly_scaled)
pred_ols_test = ols_selected.predict(X_test_selected)

pred_average_test_1 = (pred_lasso_test + pred_ridge_test + pred_elastic_test) / 3

X_stack_test_1 = np.column_stack([pred_lasso_test, pred_ridge_test, pred_elastic_test])
pred_stack_test_1 = stack_regression_1.predict(X_stack_test_1)

pred_average_test_2 = (pred_ols_test + pred_lasso_test + pred_elastic_test) / 3

X_stack_test_2 = np.column_stack([pred_ols_test, pred_lasso_test, pred_elastic_test])
pred_stack_test_2 = stack_regression_2.predict(X_stack_test_2)

pred_stack_test_6 = stack_ridge_meta_1.predict(X_stack_test_1)
pred_stack_test_7 = stack_ridge_meta_2.predict(X_stack_test_2)

models_predictions = {
    "Model 1: LASSO Baseline": pred_lasso_test,
    "Model 2: Simple Avg (LASSO+Ridge+EN)": pred_average_test_1,
    "Model 3: Stacking OLS (LASSO+Ridge+EN)": pred_stack_test_1,
    "Model 4: Simple Avg (Selected OLS+LASSO+EN)": pred_average_test_2,
    "Model 5: Stacking OLS (Selected OLS+LASSO+EN)": pred_stack_test_2,
    "Model 6: Stacking Ridge (LASSO+Ridge+EN)": pred_stack_test_6,
    "Model 7: Stacking Ridge (Selected OLS+LASSO+EN)": pred_stack_test_7
}

# =====================================================================
# Step 11: Save Kaggle submissions and submit the best one
# =====================================================================

submissions_dir = "./submissions"
os.makedirs(submissions_dir, exist_ok=True)

best_model_name = results.iloc[0]["Model"]
best_predictions = None

for idx, (model_name, preds) in enumerate(models_predictions.items(), start=1):
    filename = f"{submissions_dir}/submission_model_{idx}.csv"
    pd.DataFrame({"ID": test["ID"], "Y": preds}).to_csv(filename, index=False)
    
    if model_name == best_model_name:
        best_predictions = preds

final_submission_file = "./best_submission.csv"
pd.DataFrame({'ID': test['ID'], 'Y': best_predictions}).to_csv(final_submission_file, index=False)

print(f"\nSaved best submission to {final_submission_file}")
print(f"Submitting predictions for best model: {best_model_name} to Kaggle...")

result = api.competition_submit(
    file_name=final_submission_file,
    message=f"Best Model: {best_model_name}",
    competition=COMPETITION_ID
)
print("\n--- Kaggle Submission Response ---")
print(result)
