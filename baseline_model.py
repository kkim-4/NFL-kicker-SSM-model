import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

print("Loading data for baseline model...")
df = pd.read_csv('nfl_kicks_1999_2024.csv')

# --- 1. DATA PREP ---
# Make sure this matches your CSV's exact column name for the kick length
dist_col = 'kick_distance' 
if dist_col not in df.columns:
    dist_col = 'distance' # Fallback common name

# Drop any weird rows where distance or outcome is missing
df_clean = df.dropna(subset=[dist_col, 'made']).copy()

# --- 2. FEATURE ENGINEERING ---
print("Training 2nd-Order Polynomial Baseline...")
# We use X and X^2 to capture the non-linear drop-off in kicker accuracy
X = np.column_stack((
    df_clean[dist_col], 
    df_clean[dist_col] ** 2
))
y = df_clean['made'].values

# --- 3. TRAIN & PREDICT ---
# Fit a standard Logistic Regression 
# (penalty=None ensures we get the pure mathematical fit without regularization)
baseline_model = LogisticRegression(penalty=None, max_iter=1000)
baseline_model.fit(X, y)

# .predict_proba returns [Prob_Miss, Prob_Make]. We want column index 1.
y_prob_baseline = baseline_model.predict_proba(X)[:, 1]

# --- 4. CALCULATE METRICS ---
brier = brier_score_loss(y, y_prob_baseline)
auc = roc_auc_score(y, y_prob_baseline)
acc = accuracy_score(y, (y_prob_baseline > 0.5))

print("\n" + "="*45)
print(f"DISTANCE-ONLY BASELINE (N={len(y)})")
print("-" * 45)
print(f"Baseline Brier Score:  {brier:.4f}")
print(f"Baseline AUC-ROC:      {auc:.4f}")
print(f"Baseline Accuracy:     {acc:.2%}")
print("="*45)

# Optional: Print the formula so you can see the math
intercept = baseline_model.intercept_[0]
coef_d = baseline_model.coef_[0][0]
coef_d2 = baseline_model.coef_[0][1]
print(f"\nFormula: P(Make) = Sigmoid({intercept:.4f} + {coef_d:.4f}*d + {coef_d2:.4f}*d^2)")