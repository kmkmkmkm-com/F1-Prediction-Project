import pandas as pd
import joblib

df = pd.read_csv('rich_f1_dataset_with_features.csv')
model = joblib.load('f1_podium_model.joblib')
test_df = df[df['season'] == 2026].copy()

feature_cols = ['GridPosition', 'quali_position', 'gap_to_pole_pct', 'driver_elo_pre', 'elo_vs_teammate_diff', 'driver_avg_finish_roll5', 'driver_avg_points_roll5', 'driver_error_dnf_rate_roll5', 'driver_avg_quali_roll5', 'driver_avg_quali_gap_roll5', 'team_avg_points_roll5', 'team_mech_dnf_rate_roll5', 'team_avg_quali_roll5', 'is_wet_race', 'is_sprint_weekend', 'air_temp', 'track_temp', 'circuit_type_Street', 'circuit_type_High-Downforce', 'circuit_type_Power', 'elite_wet_modifier']

def get_rel(pos):
    try:
        p=float(pos)
        if p==1: return 3
        if p==2: return 2
        if p==3: return 1
        return 0
    except: return 0

test_df['relevance_score'] = test_df['Position'].apply(get_rel)
test_df['elite_wet_modifier'] = test_df['driver_elo_pre'] * test_df['is_wet_race']

if 'circuit_type' in test_df.columns:
    test_df = pd.get_dummies(test_df, columns=['circuit_type'], prefix='circuit_type', dtype=int)
    for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
        if f'circuit_type_{c_type}' not in test_df.columns:
            test_df[f'circuit_type_{c_type}'] = 0
else:
    for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
        if f'circuit_type_{c_type}' not in test_df.columns:
            test_df[f'circuit_type_{c_type}'] = 0

races = test_df.groupby(['season', 'race_round'])

top3_hits = 0
total_podium_spots = 0
winner_hits = 0
total_races = 0

for race_id, group in races:
    if group['relevance_score'].sum() == 0: continue
    X = group[feature_cols].fillna(0.0)
    scores = model.predict(X)
    group = group.copy()
    group['score'] = scores
    
    pred_top3 = group.nlargest(3, 'score')['Abbreviation'].values
    actual_top3 = group[group['relevance_score'] > 0]['Abbreviation'].values
    
    hits = len(set(pred_top3).intersection(set(actual_top3)))
    top3_hits += hits
    total_podium_spots += min(3, len(actual_top3))
    total_races += 1
    
    pred_winner = pred_top3[0]
    actual_winner = group[group['relevance_score'] == 3]['Abbreviation'].values
    if len(actual_winner) > 0 and pred_winner == actual_winner[0]: winner_hits += 1

print(f'\n--- 2026 TEST SET RESULTS (UNSEEN DATA) ---')
print(f'Podium Hit Rate (Top-3 Recall): {top3_hits}/{total_podium_spots} ({(top3_hits/total_podium_spots)*100:.1f}%)')
print(f'Winner Accuracy: {winner_hits}/{total_races} ({(winner_hits/total_races)*100:.1f}%)')

print("\n--- BASELINE: NAIVE GRID POSITION (POLE = WIN) ---")
baseline_correct_winner = 0
baseline_podium_hits = 0

for race_id, group in races:
    if group['relevance_score'].sum() == 0: continue
    
    pred_top3_base = group.nsmallest(3, 'GridPosition')['Abbreviation'].values
    actual_top3 = group[group['relevance_score'] > 0]['Abbreviation'].values
    
    hits = len(set(pred_top3_base).intersection(set(actual_top3)))
    baseline_podium_hits += hits
    
    pred_winner_base = pred_top3_base[0] if len(pred_top3_base) > 0 else None
    actual_winner = group[group['relevance_score'] == 3]['Abbreviation'].values
    
    if len(actual_winner) > 0 and pred_winner_base == actual_winner[0]:
        baseline_correct_winner += 1

print(f"Baseline Podium Hit Rate: {baseline_podium_hits}/{total_podium_spots} ({(baseline_podium_hits/total_podium_spots)*100:.1f}%)")
print(f"Baseline Winner Accuracy: {baseline_correct_winner}/{total_races} ({(baseline_correct_winner/total_races)*100:.1f}%)")
