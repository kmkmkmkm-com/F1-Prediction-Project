import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.metrics import ndcg_score

# =====================================================================
# STEP 1: LOAD ENRICHED DATASET
# =====================================================================
csv_path = "rich_f1_dataset_with_features.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please run feature_engineering.py first.")

df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# =====================================================================
# STEP 2: PREPROCESSING (ONE-HOT ENCODING & RANKING SETUP)
# =====================================================================
print("\nPreprocessing categorical features and generating rank variables...")

# One-hot encode circuit_type
if 'circuit_type' in df.columns:
    df = pd.get_dummies(df, columns=['circuit_type'], prefix='circuit_type', dtype=int)
else:
    for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
        if f'circuit_type_{c_type}' not in df.columns:
            df[f'circuit_type_{c_type}'] = 0

# Create Graded Relevance Score (3 for P1, 2 for P2, 1 for P3, 0 for rest)
def get_relevance(pos):
    try:
        p = float(pos)
        if p == 1: return 3
        if p == 2: return 2
        if p == 3: return 1
        return 0
    except:
        return 0

df['relevance_score'] = df['Position'].apply(get_relevance)

# Create Grouping Variable (Query ID) for XGBRanker
# The data is already chronologically sorted, but we must group it strictly by race
df['race_id'] = df.groupby(['season', 'race_round']).ngroup()

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
    'circuit_type_Power',
    'elite_wet_modifier'  # <--- NEW INTERACTION FEATURE
]

# Ensure all selected features are clean (no NaNs)
df[feature_cols] = df[feature_cols].fillna(0.0)

# Sort strictly by race_id as required by XGBRanker
df = df.sort_values('race_id').reset_index(drop=True)

# =====================================================================
# STEP 3: CHRONOLOGICAL DATA SPLIT
# =====================================================================
print("\nSplitting data chronologically...")

train_df = df[df['season'] <= 2024].copy()
val_df = df[df['season'] == 2025].copy()
test_df = df[df['season'] == 2026].copy()

X_train = train_df[feature_cols]
y_train = train_df['relevance_score']
qid_train = train_df['race_id']

X_val = val_df[feature_cols]
y_val = val_df['relevance_score']
qid_val = val_df['race_id']

X_test = test_df[feature_cols]
y_test = test_df['relevance_score']
qid_test = test_df['race_id']

print(f"Train Set (2018-2024): {len(train_df)} rows, {len(qid_train.unique())} races")
print(f"Validation Set (2025): {len(val_df)} rows, {len(qid_val.unique())} races")
print(f"Test Set (2026):       {len(test_df)} rows, {len(qid_test.unique())} races")

# =====================================================================
# STEP 4: TRAIN XGBOOST RANKER
# =====================================================================
print("\nTraining XGBoost Ranker (Learning to Rank)...")

# We use rank:ndcg which optimizes for Normalized Discounted Cumulative Gain
xgb_model = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Early stopping requires eval sets. We pass validation data with its qid
xgb_model.fit(
    X_train, y_train, qid=qid_train,
    eval_set=[(X_val, y_val)],
    eval_qid=[qid_val],
    verbose=50
)

# =====================================================================
# STEP 5: EVALUATION ON TEST SET
# =====================================================================
print("\n" + "="*50)
print("FINAL EVALUATION: UNSEEN 2026 HOLDOUT TEST SET")
print("="*50)

# To evaluate NDCG properly across all test races:
test_races = test_df.groupby('race_id')
ndcg_scores = []

for race_id, group in test_races:
    if group['relevance_score'].sum() == 0:
        continue # Skip if no positive relevance in this group (e.g. bad data)
        
    X_group = group[feature_cols]
    y_true = group['relevance_score'].values.reshape(1, -1)
    
    # Predict ranking scores
    y_pred = xgb_model.predict(X_group).reshape(1, -1)
    
    # Calculate NDCG for this race (k=3 for podium)
    score = ndcg_score(y_true, y_pred, k=3)
    ndcg_scores.append(score)

mean_ndcg = np.mean(ndcg_scores)
print(f"Test Set NDCG@3 (Average across all 2026 races): {mean_ndcg:.4f}")

# =====================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# =====================================================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(importances.head(15).to_string(index=False))

# =====================================================================
# STEP 7: MONACO 2026 PROBABILITY SIMULATION (WITH SOFTMAX)
# =====================================================================
print("\n" + "="*50)
print("SIMULATION: 2026 MONACO GRAND PRIX (RANKING SCORE TO POWER RATING)")
print("="*50)

# Filter test set for Monaco GP
monaco_test = test_df[test_df['EventName'].str.contains('Monaco', na=False)].copy()

if not monaco_test.empty:
    X_monaco = monaco_test[feature_cols].fillna(0.0)
    
    # Predict arbitrary ranking scores
    raw_scores = xgb_model.predict(X_monaco)
    
    # Apply Softmax to convert scores to a 0-100 "Power Rating"
    # We add a temperature parameter to prevent the scores from being too skewed
    temperature = 1.5 
    exp_scores = np.exp(raw_scores / temperature)
    power_ratings = exp_scores / np.sum(exp_scores)
    
    monaco_test['Raw_Ranking_Score'] = raw_scores
    monaco_test['Power_Rating_Pct'] = (power_ratings * 100).round(2)
    
    monaco_ranking = monaco_test[[
        'Abbreviation', 'TeamName', 'GridPosition', 'driver_elo_pre', 
        'Raw_Ranking_Score', 'Power_Rating_Pct', 'Position'
    ]].sort_values('Raw_Ranking_Score', ascending=False)
    
    print(monaco_ranking.to_string(index=False))
else:
    print("Could not find Monaco Grand Prix in the 2026 test set.")


# =====================================================================
# STEP 8: EXPORT MODEL
# =====================================================================
print("\n" + "="*50)
print("EXPORTING MODEL FOR STREAMLIT DEPLOYMENT")
print("="*50)

model_path = "f1_podium_model.joblib"
joblib.dump(xgb_model, model_path)
print(f"Model successfully saved to '{model_path}'.")
