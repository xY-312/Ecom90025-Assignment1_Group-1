import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import (
    LassoCV,
    RidgeCV,
    ElasticNetCV,
    LinearRegression,
    Ridge
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

# =====================================================================
# Part 1: Data Preparation
# =====================================================================
print("===== Part 1: Data Preparation =====")

# Ensure directories exist
DATA_DIR = Path("./data")
SUBMISSIONS_DIR = Path("./submissions")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Authenticate Kaggle API
api_key = "KGAT_07005328f6852bee7a8b908d496828d2"
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    f.write('{"username":"student","key":"' + api_key + '"}')
with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(api_key)

api = KaggleApi()
api.authenticate()
COMPETITION_ID = "ecom-90025-2026-sm-2-ada-assignment-and-practice"

train_path = DATA_DIR / "train_data.csv"
if not train_path.exists():
    print("Downloading competition data...")
    api.competition_download_files(competition=COMPETITION_ID, path=str(DATA_DIR))
    zip_filepath = DATA_DIR / f"{COMPETITION_ID}.zip"
    if zip_filepath.exists():
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_filepath)

# 1. Read data and separate Y and ID
train = pd.read_csv(train_path)
test = pd.read_csv(DATA_DIR / "test_data.csv")

X_train_raw = train.drop(columns=["Y", "ID"])
y_train = train["Y"].reset_index(drop=True)
X_test_raw = test.drop(columns=["ID"])

# 2. Build feature matrix (original, squared, and interaction terms)
print("Building polynomial features (degree=2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_raw)
X_test_poly = poly.transform(X_test_raw)

# 3. Standardize all features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


# =====================================================================
# Part 2: Define 5 Base Models
# =====================================================================
print("\n===== Part 2: Defining Base Models =====")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set higher max_iter to prevent convergence warnings
MAX_ITER = 50000

# Base Model 1: LASSO
print("Fitting global LASSO to extract non-zero coefficients...")
lasso_model = LassoCV(cv=kf, random_state=42, n_jobs=-1, max_iter=MAX_ITER)
# Fit on the entire training set first to find non-zero coefficients for later models
lasso_model.fit(X_train_scaled, y_train)
selected_indices = np.where(lasso_model.coef_ != 0)[0]
zero_indices = np.where(lasso_model.coef_ == 0)[0]
print(f"LASSO selected {len(selected_indices)} non-zero variables.")

# Base Model 2: Ridge
ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=kf)

# Base Model 3: Elastic Net
elastic_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 20), 
                             cv=kf, random_state=42, n_jobs=-1, max_iter=MAX_ITER)

# Base Model 4: Selected OLS (Custom Wrapper)
class SelectedOLS(BaseEstimator, RegressorMixin):
    def __init__(self, selected_idx=None):
        self.selected_idx = selected_idx
        self.ols = LinearRegression()
        
    def fit(self, X, y):
        if len(self.selected_idx) > 0:
            self.ols.fit(X[:, self.selected_idx], y)
        else:
            self.ols.fit(np.zeros((X.shape[0], 1)), y)
        return self
        
    def predict(self, X):
        if len(self.selected_idx) > 0:
            return self.ols.predict(X[:, self.selected_idx])
        return self.ols.predict(np.zeros((X.shape[0], 1)))

sel_ols_model = SelectedOLS(selected_idx=selected_indices)

# Base Model 5: FWL Partialled Predictor (Custom Wrapper)
class FWLPartialledPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, key_idx=None, ctrl_idx=None):
        self.key_idx = key_idx
        self.ctrl_idx = ctrl_idx
        self.model_y_ctrl_ = None
        self.model_x_ctrl_ = None
        self.res_reg_ = None

    def fit(self, X, y):
        # If no control variables exist, FWL degenerates to standard OLS
        if len(self.ctrl_idx) == 0:
            self.res_reg_ = LinearRegression()
            self.res_reg_.fit(X[:, self.key_idx], y)
            return self
            
        if len(self.key_idx) == 0:
            self.res_reg_ = LinearRegression()
            self.res_reg_.fit(np.zeros((X.shape[0], 1)), y)
            return self

        X_key = X[:, self.key_idx]
        X_ctrl = X[:, self.ctrl_idx]

        # Use RidgeCV for controls to avoid instability with high-dimensional OLS
        alphas = np.logspace(-3, 3, 10)
        self.model_y_ctrl_ = RidgeCV(alphas=alphas)
        self.model_x_ctrl_ = RidgeCV(alphas=alphas)
        self.res_reg_ = LinearRegression()

        # 1. Regress Y on controls -> obtain Y residuals
        self.model_y_ctrl_.fit(X_ctrl, y)
        y_res = y - self.model_y_ctrl_.predict(X_ctrl)

        # 2. Regress key variables on controls -> obtain key variable residuals
        self.model_x_ctrl_.fit(X_ctrl, X_key)
        X_key_res = X_key - self.model_x_ctrl_.predict(X_ctrl)

        # 3. Regress Y residuals on key variable residuals
        self.res_reg_.fit(X_key_res, y_res)
        return self

    def predict(self, X):
        if len(self.ctrl_idx) == 0:
            return self.res_reg_.predict(X[:, self.key_idx])
        if len(self.key_idx) == 0:
            return self.res_reg_.predict(np.zeros((X.shape[0], 1)))

        X_key = X[:, self.key_idx]
        X_ctrl = X[:, self.ctrl_idx]

        # Prediction: Part predicted by controls + Part predicted by residual model
        y_ctrl_pred = self.model_y_ctrl_.predict(X_ctrl)
        X_key_res = X_key - self.model_x_ctrl_.predict(X_ctrl)
        y_res_pred = self.res_reg_.predict(X_key_res)

        return y_ctrl_pred + y_res_pred

fwl_model = FWLPartialledPredictor(key_idx=selected_indices, ctrl_idx=zero_indices)


# =====================================================================
# Part 3: Calculate 5-fold CV MSE for 14 Model Configurations
# =====================================================================
print("\n===== Part 3: 5-fold CV Calculations =====")

# Generate out-of-fold (OOF) predictions for the 5 base models
print("Generating out-of-fold predictions for base models (this may take some time)...")
oof_lasso = cross_val_predict(lasso_model, X_train_scaled, y_train, cv=kf, n_jobs=-1)
oof_ridge = cross_val_predict(ridge_model, X_train_scaled, y_train, cv=kf, n_jobs=-1)
oof_elastic = cross_val_predict(elastic_model, X_train_scaled, y_train, cv=kf, n_jobs=-1)
oof_sel_ols = cross_val_predict(sel_ols_model, X_train_scaled, y_train, cv=kf, n_jobs=-1)
oof_fwl = cross_val_predict(fwl_model, X_train_scaled, y_train, cv=kf, n_jobs=-1)

# Group 1: Baseline Models MSE
mse_lasso = mean_squared_error(y_train, oof_lasso)
mse_ridge = mean_squared_error(y_train, oof_ridge)
mse_elastic = mean_squared_error(y_train, oof_elastic)
mse_sel_ols = mean_squared_error(y_train, oof_sel_ols)
mse_fwl = mean_squared_error(y_train, oof_fwl)

# Prepare Stacking feature matrices
oof_stack_group2 = np.column_stack([oof_lasso, oof_ridge, oof_elastic])
oof_stack_group3 = np.column_stack([oof_sel_ols, oof_lasso, oof_elastic])
oof_stack_group4 = np.column_stack([oof_lasso, oof_ridge, oof_elastic, oof_sel_ols, oof_fwl])

# Define generic Stacking meta-models
meta_ols = LinearRegression()
meta_ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=kf)

# Helper function to calculate ensemble CV MSE
def get_ensemble_mse(X_stack, meta_model=None):
    if meta_model is None: # Averaging
        preds = np.mean(X_stack, axis=1)
        return mean_squared_error(y_train, preds)
    else: # OLS or Ridge Stacking
        preds = cross_val_predict(meta_model, X_stack, y_train, cv=kf, n_jobs=-1)
        return mean_squared_error(y_train, preds)

# Group 2: A3 3-Model Ensemble (LASSO + Ridge + Elastic Net)
mse_g2_avg = get_ensemble_mse(oof_stack_group2, None)
mse_g2_ols = get_ensemble_mse(oof_stack_group2, meta_ols)
mse_g2_ridge = get_ensemble_mse(oof_stack_group2, meta_ridge)

# Group 3: Alternative A3 3-Model Ensemble (Sel OLS + LASSO + Elastic Net)
mse_g3_avg = get_ensemble_mse(oof_stack_group3, None)
mse_g3_ols = get_ensemble_mse(oof_stack_group3, meta_ols)
mse_g3_ridge = get_ensemble_mse(oof_stack_group3, meta_ridge)

# Group 4: A4 New Ensemble (All 5 Base Models)
mse_g4_avg = get_ensemble_mse(oof_stack_group4, None)
mse_g4_ols = get_ensemble_mse(oof_stack_group4, meta_ols)
mse_g4_ridge = get_ensemble_mse(oof_stack_group4, meta_ridge)


# =====================================================================
# Part 4: Result Summary
# =====================================================================
print("\n===== Part 4: Result Summary =====")

results = [
    (1, "LASSO", "Baseline", mse_lasso),
    (2, "Ridge", "Baseline", mse_ridge),
    (3, "Elastic Net", "Baseline", mse_elastic),
    (4, "Selected OLS", "Baseline", mse_sel_ols),
    (5, "FWL Partialled Predictor", "Baseline", mse_fwl),
    
    (6, "LASSO + Ridge + ENet", "Averaging", mse_g2_avg),
    (7, "LASSO + Ridge + ENet", "Stacking (OLS)", mse_g2_ols),
    (8, "LASSO + Ridge + ENet", "Stacking (Ridge)", mse_g2_ridge),
    
    (9, "Selected OLS + LASSO + ENet", "Averaging", mse_g3_avg),
    (10, "Selected OLS + LASSO + ENet", "Stacking (OLS)", mse_g3_ols),
    (11, "Selected OLS + LASSO + ENet", "Stacking (Ridge)", mse_g3_ridge),
    
    (12, "LASSO + Ridge + ENet + Sel OLS + FWL", "Averaging", mse_g4_avg),
    (13, "LASSO + Ridge + ENet + Sel OLS + FWL", "Stacking (OLS)", mse_g4_ols),
    (14, "LASSO + Ridge + ENet + Sel OLS + FWL", "Stacking (Ridge)", mse_g4_ridge)
]

print("=====================================================================")
print(f" {'#':<2} | {'Model Spec':<47} | {'Method':<15} | {'CV MSE'}")
print("----|---------------------------------------------------|-----------------|--------")
best_mse = float('inf')
best_id = -1
for id_val, spec, method, mse in results:
    print(f" {id_val:<2} | {spec:<47} | {method:<15} | {mse:.5f}")
    if mse < best_mse:
        best_mse = mse
        best_id = id_val
print("=====================================================================")
print(f"Best model: #{best_id} with CV MSE = {best_mse:.5f}")


# =====================================================================
# Part 5: Generate Kaggle Submission Files
# =====================================================================
print("\n===== Part 5: Generating Kaggle Submissions =====")

# 1. Retrain all 5 base models on the entire training set
print("Refitting base models on full training data...")
lasso_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
elastic_model.fit(X_train_scaled, y_train)
sel_ols_model.fit(X_train_scaled, y_train)
fwl_model.fit(X_train_scaled, y_train)

# 2. Generate test set predictions for all base models
print("Generating test set predictions...")
pred_lasso = lasso_model.predict(X_test_scaled)
pred_ridge = ridge_model.predict(X_test_scaled)
pred_elastic = elastic_model.predict(X_test_scaled)
pred_sel_ols = sel_ols_model.predict(X_test_scaled)
pred_fwl = fwl_model.predict(X_test_scaled)

# Prepare combined feature matrices for the test set
X_test_g2 = np.column_stack([pred_lasso, pred_ridge, pred_elastic])
X_test_g4 = np.column_stack([pred_lasso, pred_ridge, pred_elastic, pred_sel_ols, pred_fwl])

# 3. Combine predictions and output CSV files
# --- #12: 5-Model Averaging ---
pred_12 = np.mean(X_test_g4, axis=1)
pd.DataFrame({'ID': test['ID'], 'Y': pred_12}).to_csv(SUBMISSIONS_DIR / "submission_5model_avg.csv", index=False)

# --- #13: 5-Model OLS Stacking ---
meta_ols.fit(oof_stack_group4, y_train)
pred_13 = meta_ols.predict(X_test_g4)
pd.DataFrame({'ID': test['ID'], 'Y': pred_13}).to_csv(SUBMISSIONS_DIR / "submission_5model_ols_stack.csv", index=False)

# --- #14: 5-Model Ridge Stacking ---
meta_ridge.fit(oof_stack_group4, y_train)
pred_14 = meta_ridge.predict(X_test_g4)
pd.DataFrame({'ID': test['ID'], 'Y': pred_14}).to_csv(SUBMISSIONS_DIR / "submission_5model_ridge_stack.csv", index=False)

# --- #7: 3-Model OLS Stacking (A3 Baseline) ---
meta_ols.fit(oof_stack_group2, y_train)
pred_7 = meta_ols.predict(X_test_g2)
pd.DataFrame({'ID': test['ID'], 'Y': pred_7}).to_csv(SUBMISSIONS_DIR / "submission_3model_ols_stack.csv", index=False)


# =====================================================================
# Part 6: Additional Requirements
# =====================================================================
print("\nAll submission files have been successfully generated.")
print("所有 submission 文件已生成")
