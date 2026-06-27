import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import xgboost as xgb

# =====================================================================
# STEP 1: LOAD ENRICHED DATASET
# =====================================================================
csv_path = "rich_f1_dataset_with_features.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please run feature_engineering.py first.")

df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# =====================================================================
# STEP 2: PREPROCESSING (ONE-HOT ENCODING)
# =====================================================================
print("\nPreprocessing categorical features...")

# One-hot encode circuit_type
if 'circuit_type' in df.columns:
    df = pd.get_dummies(df, columns=['circuit_type'], prefix='circuit_type', dtype=int)
else:
    for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
        df[f'circuit_type_{c_type}'] = 0

# List features we will use for modeling
feature_cols = [
    'GridPosition',
    'quali_position',
    'gap_to_pole_pct',
    'driver_elo_pre',
    'elo_vs_teammate_diff',
    'driver_avg_finish_roll5',
    'driver_avg_points_roll5',
    'driver_error_dnf_rate_roll5',
    'driver_avg_quali_roll5',
    'driver_avg_quali_gap_roll5',
    'team_avg_points_roll5',
    'team_mech_dnf_rate_roll5',
    'team_avg_quali_roll5',
    'is_wet_race',
    'is_sprint_weekend',
    'air_temp',
    'track_temp',
    'circuit_type_Street',
    'circuit_type_High-Downforce',
    'circuit_type_Power'
]

# Target column
target_col = 'podium_finish'

# Ensure all selected features are clean (no NaNs)
df[feature_cols] = df[feature_cols].fillna(0.0)

# =====================================================================
# STEP 3: CHRONOLOGICAL DATA SPLIT
# =====================================================================
print("Splitting data chronologically...")

train_df = df[df['season'] <= 2024].copy()
val_df = df[df['season'] == 2025].copy()
test_df = df[df['season'] == 2026].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"Train Set (2018-2024): X={X_train.shape}, y={y_train.shape}")
print(f"Validation Set (2025): X={X_val.shape}, y={y_val.shape}")
print(f"Test Set (2026):       X={X_test.shape}, y={y_test.shape}")

# =====================================================================
# STEP 4: 2D GRID SEARCH FOR RANDOM FOREST
# =====================================================================
print("\nRunning 2D Grid Search for Random Forest...")
print(f"{'Max Depth':10} | {'Num Trees':10} | {'Validation ROC-AUC':20}")
print("-" * 50)

best_auc = 0
best_depth = 0
best_trees = 0

for depth in [4, 6, 8, 10]:
    for trees in [50, 100, 150, 200]:
        temp_model = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42)
        temp_model.fit(X_train, y_train)
        
        probs = temp_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        print(f"{depth:10d} | {trees:10d} | {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_depth = depth
            best_trees = trees

print("-" * 50)
print(f"Optimal parameters: max_depth={best_depth}, n_estimators={best_trees} (ROC-AUC = {best_auc:.4f})")

# Retrain final model with the best parameters
rf_model = RandomForestClassifier(n_estimators=best_trees, max_depth=best_depth, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate on Validation set (2025)
rf_preds = rf_model.predict(X_val)
rf_probs = rf_model.predict_proba(X_val)[:, 1]

rf_auc = roc_auc_score(y_val, rf_probs)
rf_precision = precision_score(y_val, rf_preds, zero_division=0)
rf_recall = recall_score(y_val, rf_preds, zero_division=0)
rf_accuracy = accuracy_score(y_val, rf_preds)


# =====================================================================
# STEP 5: TRAIN CHALLENGER XGBOOST CLASSIFIER
# =====================================================================
print("\nTraining XGBoost challenger...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Evaluate on Validation set (2025)
xgb_preds = xgb_model.predict(X_val)
xgb_probs = xgb_model.predict_proba(X_val)[:, 1]

xgb_auc = roc_auc_score(y_val, xgb_probs)
xgb_precision = precision_score(y_val, xgb_preds, zero_division=0)
xgb_recall = recall_score(y_val, xgb_preds, zero_division=0)
xgb_accuracy = accuracy_score(y_val, xgb_preds)

print("XGBoost Validation Performance:")
print(f"  ROC-AUC:   {xgb_auc:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  Accuracy:  {xgb_accuracy:.4f}")

# =====================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# =====================================================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Random Forest Feature Importance
rf_importances = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False).reset_index(drop=True)

# XGBoost Feature Importance
xgb_importances = pd.DataFrame({
    'Feature': feature_cols,
    'XGB_Importance': xgb_model.feature_importances_
}).sort_values('XGB_Importance', ascending=False).reset_index(drop=True)

# Merge and print top features
merged_importances = rf_importances.merge(xgb_importances, on='Feature')
print(merged_importances.head(15).to_string(index=False))

# =====================================================================
# STEP 7: MONACO 2025 PROBABILITY SIMULATION
# =====================================================================
print("\n" + "="*50)
print("SIMULATION: 2025 MONACO GRAND PRIX PROBABILITY RANKING")
print("="*50)

# Filter validation set for Monaco GP
monaco_val = val_df[val_df['EventName'].str.contains('Monaco', na=False)].copy()

if not monaco_val.empty:
    X_monaco = monaco_val[feature_cols].fillna(0.0)
    
    # Predict probabilities using both models
    monaco_val['RF_Podium_Prob'] = rf_model.predict_proba(X_monaco)[:, 1]
    monaco_val['XGB_Podium_Prob'] = xgb_model.predict_proba(X_monaco)[:, 1]
    
    # Let's print the XGBoost ranking (our boosting challenger)
    monaco_ranking = monaco_val[[
        'Abbreviation', 'TeamName', 'GridPosition', 'quali_position', 
        'driver_elo_pre', 'XGB_Podium_Prob', 'RF_Podium_Prob', 'Position', 'podium_finish'
    ]].sort_values('XGB_Podium_Prob', ascending=False)
    
    print(monaco_ranking.to_string(index=False))
else:
    print("Could not find Monaco Grand Prix in the 2025 validation set.")

# =====================================================================
# STEP 8: FINAL EVALUATION ON 2026 TEST SET (HOLDOUT)
# =====================================================================
print("\n" + "="*50)
print("FINAL EVALUATION: UNSEEN 2026 HOLDOUT TEST SET")
print("="*50)

# Select the model with the higher ROC-AUC on validation set as final
if xgb_auc > rf_auc:
    final_model = xgb_model
    model_name = "XGBoost"
    final_probs = xgb_model.predict_proba(X_test)[:, 1]
    final_preds = xgb_model.predict(X_test)
else:
    final_model = rf_model
    model_name = "Random Forest"
    final_probs = rf_model.predict_proba(X_test)[:, 1]
    final_preds = rf_model.predict(X_test)

final_auc = roc_auc_score(y_test, final_probs)
final_precision = precision_score(y_test, final_preds, zero_division=0)
final_recall = recall_score(y_test, final_preds, zero_division=0)
final_accuracy = accuracy_score(y_test, final_preds)

print(f"Selected Final Model: {model_name}")
print(f"Test Performance on 2026 Season:")
print(f"  ROC-AUC:   {final_auc:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall:    {final_recall:.4f}")
print(f"  Accuracy:  {final_accuracy:.4f}")

# =====================================================================
# STEP 9: EXPORT MODEL
# =====================================================================
print("\n" + "="*50)
print("EXPORTING MODEL FOR STREAMLIT DEPLOYMENT")
print("="*50)

model_path = "f1_podium_model.joblib"
joblib.dump(final_model, model_path)
print(f"Model successfully saved to '{model_path}'.")

