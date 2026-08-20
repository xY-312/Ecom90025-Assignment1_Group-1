import os
import zipfile
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Kaggle API Setup
# ==========================================
api_key = "KGAT_9c406361a43b659d0fc623dd25083604"
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(api_key)

api = KaggleApi()
api.authenticate()

COMPETITION_ID = "ecom-90025-2026-sm-2-ada-assignment-and-practice"
DESTINATION_PATH = "./data/A2"

print(f"Downloading dataset for {COMPETITION_ID}...")
api.competition_download_files(competition=COMPETITION_ID, path=DESTINATION_PATH)

zip_filepath = os.path.join(DESTINATION_PATH, f"{COMPETITION_ID}.zip")
if os.path.exists(zip_filepath):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(DESTINATION_PATH)
    os.remove(zip_filepath)

# ==========================================
# 2. Data Loading & Preparation
# ==========================================
print("Loading data...")
train_df = pd.read_csv(os.path.join(DESTINATION_PATH, "train_data.csv"))
test_df = pd.read_csv(os.path.join(DESTINATION_PATH, "test_data.csv"))

X_train = train_df.drop(columns=['ID', 'Y'])
y_train = train_df['Y']
X_test = test_df.drop(columns=['ID'])

# ==========================================
# 3. Feature Engineering 
# ==========================================
# Based on Week 3 open-ended modeling, we add interactions and polynomials
print("Generating Polynomial and Interaction Features...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print(f"Features expanded from {X_train.shape[1]} to {X_train_poly.shape[1]}")

# Feature Scaling (Crucial for Lasso/Regularization)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# ==========================================
# 4. Model Training (Lasso with K-Fold CV)
# ==========================================
print("Training Lasso Model with 5-Fold Cross Validation...")
# KFold CV concept from Week 3 combined with Lasso regularization
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_cv = LassoCV(cv=kf, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

best_alpha = lasso_cv.alpha_
features_kept = sum(lasso_cv.coef_ != 0)
print(f"Optimal Alpha selected: {best_alpha:.4f}")
print(f"Features kept by Lasso: {features_kept} / {X_train_poly.shape[1]}")

# ==========================================
# 4.1 Visualization of Alpha Selection
# ==========================================
print("Generating Alpha Selection Plots...")

# Plot 1: Lasso Cross-Validation Results (MSE vs Alpha)
plt.figure(figsize=(10, 6))
# Calculate mean and standard error of MSE across folds
mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
std_mse = np.std(lasso_cv.mse_path_, axis=1) / np.sqrt(lasso_cv.mse_path_.shape[1])

# Plot with error bars
plt.errorbar(lasso_cv.alphas_, mean_mse, yerr=std_mse, fmt='-o', color='#5E72E4', 
             ecolor='#5E72E4', capsize=3, markersize=4, label='Mean Squared Error')
plt.axvline(best_alpha, linestyle='--', color='red', label=f'Best Alpha (CV): {best_alpha:.4f}')
plt.xscale('log')
plt.gca().invert_xaxis()  # Alpha goes from larger to smaller
plt.xlabel('Alpha (Penalty Term)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Cross-Validation Results')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 4.2 Evaluation Metrics & Feature Importance
# ==========================================
# Training R2
r2_train = lasso_cv.score(X_train_scaled, y_train)
print(f"Training R²: {r2_train:.4f}")

# 5-fold CV R2 using the best alpha
lasso_best = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
cv_r2_scores = cross_val_score(lasso_best, X_train_scaled, y_train, cv=kf, scoring='r2')
print(f"5-Fold CV R² (each fold): {[round(score, 4) for score in cv_r2_scores]}")
print(f"5-Fold CV R² (mean): {np.mean(cv_r2_scores):.4f}")

# Top 10 features kept by Lasso (sorted by absolute coefficient)
feature_names = poly.get_feature_names_out(X_train.columns)
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_cv.coef_})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
top_10_features = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Abs_Coefficient', ascending=False).head(10)

print("\nTop 10 Features Kept by Lasso (by absolute coefficient):")
for idx, row in top_10_features.iterrows():
    print(f"{row['Feature']:>25}: {row['Coefficient']:.4f}")
print("\n" + "="*42 + "\n")

# ==========================================
# 5. Prediction and Kaggle Submission
# ==========================================
print("Predicting on test data...")
predictions = lasso_cv.predict(X_test_scaled)

submission_df = pd.DataFrame({
    'ID': test_df['ID'],
    'Y': predictions
})

submission_path = os.path.join(DESTINATION_PATH, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

print("Attempting to submit to Kaggle Leaderboard...")
try:
    api.competition_submit( 
        file_name=submission_path,
        message="Lasso Regression with Polynomial Features (Assignment 2)",
        competition=COMPETITION_ID
    )
    print("Successfully submitted to Kaggle!")
except Exception as e:
    print(f"Notice: Failed to submit via API (Possibly hit the daily limit).")
    print(f"Error details: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(e.response.text)
