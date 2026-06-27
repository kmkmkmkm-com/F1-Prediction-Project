import pandas as pd
import numpy as np 
import os 

csv_path = "rich_f1_dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}. Please ensure the extraction is complete.")

df = pd.read_csv(csv_path)
df = df.sort_values(['season','race_round']).reset_index(drop=True)

print(df.shape)

def calulate_quali_outcome(driver1,driver2):
    pos1=driver1['quali_position']
    pos2=driver2['quali_position']
    if pd.isna(pos1) and pd.isna(pos2):
        return 0.5, 0.5
    elif pd.isna(pos1) and pd.notna(pos2):
        return 0.0, 1.0
    elif pd.notna(pos1) and pd.isna(pos2):
        return 1.0, 0.0

    else:
        if float(pos1)<float(pos2):
            return 1.0,0.0
        elif float(pos2)<float(pos1):
            return 0.0,1.0
        else:
            return 0.5,0.5

def calculate_race_outcome(driver1,driver2):
    status1 = str(driver1['Status']).lower()
    status2 = str(driver2['Status']).lower()
    pos1=float(driver1['Position'])
    pos2=float(driver2['Position'])
    finisher1 = "finished" in status1 or "lap" in status1
    finisher2 = "finished" in status2 or "lap" in status2
    if finisher1 and finisher2:
        if pos1<pos2:
            return 1.0,0.0
        elif pos2<pos1:
            return 0.0,1.0
        else:
            return 0.5,0.5
    elif finisher1 and not finisher2:
        return 1.0,0.0
    elif not finisher1 and finisher2:
        return 0.0,1.0
    else:
        laps1=int(driver1['Laps'])
        laps2=int(driver2['Laps'])
        if laps1>laps2:
            return 1.0,0.0
        elif laps2>laps1:
            return 0.0,1.0
        else:
            return 0.5,0.5

driver_elo = {}
K_FACTOR = 32
pre_race_elos = []  # store tuples (row_index,pre_race_elo)
post_race_elos = [] # store tuples (row_index,post_race_elo)
teammate_elos = [] # store tuples (row_index,teammate_elo) to calculate elo diff bw teammates

races = df.groupby(['season', 'race_round'])

# We loop through every race weekend in history chronologically
for (season, rnd), race_group in races:
    # 1. Capture the pre-race ELO for all drivers in this weekend.
    # We must use pre-race ELO as our machine learning feature to avoid lookahead bias.
    weekend_pre_elo = {}
    for idx, row in race_group.iterrows():
        driver = row['Abbreviation']
        # If a driver has never raced before, they start at the baseline of 1500
        current_elo = driver_elo.get(driver, 1500.0)
        weekend_pre_elo[driver] = current_elo
        pre_race_elos.append((idx, current_elo))
        
    # 2. Compute ELO updates team-by-team
    weekend_post_elo = weekend_pre_elo.copy()
    
    for team, team_group in race_group.groupby('TeamName'):
        # Teammate comparison only works if both cars are present
        if len(team_group) == 2:
            drivers = team_group.to_dict('records')
            driver_a_row = drivers[0]
            driver_b_row = drivers[1]
            
            driver_a = driver_a_row['Abbreviation']
            driver_b = driver_b_row['Abbreviation']
            
            # Current ratings entering the weekend
            R_A = weekend_pre_elo[driver_a]
            R_B = weekend_pre_elo[driver_b]
            
            # Store teammate ELO for ELO difference features
            # driver_a_row has index, but if not we use the pandas index of team_group
            teammate_elos.append((team_group.index[0], R_B))
            teammate_elos.append((team_group.index[1], R_A))
            
            # Calculate Expected Scores (Standard ELO Chess Formula)
            E_A = 1.0 / (1.0 + 10.0 ** ((R_B - R_A) / 400.0))
            E_B = 1.0 - E_A
            
            # Calculate Actual Outcomes (using our custom helper functions)
            q_a, q_b = calulate_quali_outcome(driver_a_row, driver_b_row)
            r_a, r_b = calculate_race_outcome(driver_a_row, driver_b_row)
            
            # Combined score: 30% Qualifying weight, 70% Race weight
            S_A = 0.3 * q_a + 0.7 * r_a
            S_B = 0.3 * q_b + 0.7 * r_b
            
            # Update ratings using K = 32
            R_A_new = R_A + K_FACTOR * (S_A - E_A)
            R_B_new = R_B + K_FACTOR * (S_B - E_B)
            
            # Update our master dictionary and weekend post-race record
            driver_elo[driver_a] = R_A_new
            driver_elo[driver_b] = R_B_new
            weekend_post_elo[driver_a] = R_A_new
            weekend_post_elo[driver_b] = R_B_new
        else:
            # If a team has only 1 car running (e.g., driver withdrew), no ELO update occurs.
            # They carry over their current ELO and teammate ELO defaults to their own ELO (diff = 0).
            for idx, row in team_group.iterrows():
                driver = row['Abbreviation']
                teammate_elos.append((idx, weekend_pre_elo[driver]))

    # Store post-race ELOs
    for idx, row in race_group.iterrows():
        driver = row['Abbreviation']
        post_race_elos.append((idx, weekend_post_elo[driver]))

# Map the ELO lists back to our main DataFrame using the row indices
pre_race_dict = dict(pre_race_elos)
post_race_dict = dict(post_race_elos)
teammate_dict = dict(teammate_elos)

df['driver_elo_pre'] = df.index.map(pre_race_dict)
df['driver_elo_post'] = df.index.map(post_race_dict)
df['teammate_elo_pre'] = df.index.map(teammate_dict)
df['elo_vs_teammate_diff'] = df['driver_elo_pre'] - df['teammate_elo_pre']

print("ELO ratings successfully calculated.")

# =====================================================================
# SECTION 5: EVENT MAPPING & CIRCUIT CHARACTERISTICS
# =====================================================================
print("\nMapping event names and circuit characteristics...")
import fastf1
fastf1.Cache.enable_cache('f1_cache')

# Fetch schedules to map EventName to (season, race_round)
all_schedules = []
for year in range(2018, 2027):
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing'].copy()
        schedule['season'] = year
        schedule = schedule.rename(columns={'RoundNumber': 'race_round'})
        all_schedules.append(schedule[['season', 'race_round', 'EventName']])
    except Exception as e:
        print(f"Warning: Could not fetch schedule for {year}: {e}")

if all_schedules:
    schedule_df = pd.concat(all_schedules, ignore_index=True)
    schedule_df = schedule_df.drop_duplicates(subset=['season', 'race_round'])
    df = df.merge(schedule_df, on=['season', 'race_round'], how='left')
else:
    df['EventName'] = "Unknown"

# Define a robust, expert-level categorization of F1 circuits
circuit_type_map = {
    'Monaco Grand Prix': 'Street',
    'Singapore Grand Prix': 'Street',
    'Azerbaijan Grand Prix': 'Street',
    'Las Vegas Grand Prix': 'Street',
    'Miami Grand Prix': 'Street',
    'Saudi Arabian Grand Prix': 'Street',
    'Australian Grand Prix': 'Street',
    'Canadian Grand Prix': 'Street',
    
    'Hungarian Grand Prix': 'High-Downforce',
    'Dutch Grand Prix': 'High-Downforce',
    'Spanish Grand Prix': 'High-Downforce',
    'Barcelona Grand Prix': 'High-Downforce',
    'Emilia Romagna Grand Prix': 'High-Downforce',
    'Tuscan Grand Prix': 'High-Downforce',
    'Eifel Grand Prix': 'High-Downforce',
    'Turkish Grand Prix': 'High-Downforce',
    
    'Italian Grand Prix': 'Power',
    'Belgian Grand Prix': 'Power',
    'Austrian Grand Prix': 'Power',
    'Styrian Grand Prix': 'Power',
    'Sakhir Grand Prix': 'Power',
    'Bahrain Grand Prix': 'Power',
    
    'British Grand Prix': 'Balanced',
    'Japanese Grand Prix': 'Balanced',
    'United States Grand Prix': 'Balanced',
    'Brazilian Grand Prix': 'Balanced',
    'So Paulo Grand Prix': 'Balanced',
    'São Paulo Grand Prix': 'Balanced',
    'Chinese Grand Prix': 'Balanced',
    'Portuguese Grand Prix': 'Balanced',
    'Qatar Grand Prix': 'Balanced',
    'German Grand Prix': 'Balanced',
    'Russian Grand Prix': 'Balanced',
    'French Grand Prix': 'Balanced',
    'Abu Dhabi Grand Prix': 'Balanced',
    'Mexican Grand Prix': 'Balanced',
    'Mexico City Grand Prix': 'Balanced',
    '70th Anniversary Grand Prix': 'Balanced'
}

df['circuit_type'] = df['EventName'].map(circuit_type_map).fillna('Balanced')

# =====================================================================
# SECTION 6: ROLLING DRIVER FORM (5-RACE WINDOWS)
# =====================================================================
print("Calculating 5-race rolling driver form features...")

# Ensure Position and Points are numeric
df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
df['quali_position'] = pd.to_numeric(df['quali_position'], errors='coerce')
df['gap_to_pole_pct'] = pd.to_numeric(df['gap_to_pole_pct'], errors='coerce')

# Identify DNFs and categorize them
driver_error_keywords = ['collision', 'accident', 'spun off', 'disqualified']

def get_dnf_type(status):
    s = str(status).lower()
    if "finished" in s or "lap" in s:
        return 'none' # Not a DNF
    elif any(k in s for k in driver_error_keywords):
        return 'driver_error'
    else:
        return 'mechanical'

df['dnf_type'] = df['Status'].apply(get_dnf_type)
df['is_driver_dnf'] = (df['dnf_type'] == 'driver_error').astype(int)
df['is_mech_dnf'] = (df['dnf_type'] == 'mechanical').astype(int)
# Keeping general is_dnf for legacy/ELO checks
df['is_dnf'] = (df['dnf_type'] != 'none').astype(int)

# Group by driver and calculate rolling features shifted by 1 to prevent lookahead bias
driver_groups = df.groupby('Abbreviation')

df['driver_avg_finish_roll5'] = driver_groups['Position'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)
df['driver_avg_points_roll5'] = driver_groups['Points'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)
df['driver_error_dnf_rate_roll5'] = driver_groups['is_driver_dnf'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)
df['driver_avg_quali_roll5'] = driver_groups['quali_position'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)
df['driver_avg_quali_gap_roll5'] = driver_groups['gap_to_pole_pct'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)

# Fill NaNs with midfield baselines for robustness
df['driver_avg_finish_roll5'] = df['driver_avg_finish_roll5'].fillna(12.0)
df['driver_avg_points_roll5'] = df['driver_avg_points_roll5'].fillna(2.0)
df['driver_error_dnf_rate_roll5'] = df['driver_error_dnf_rate_roll5'].fillna(0.05)
df['driver_avg_quali_roll5'] = df['driver_avg_quali_roll5'].fillna(12.0)
df['driver_avg_quali_gap_roll5'] = df['driver_avg_quali_gap_roll5'].fillna(3.0)

# =====================================================================
# SECTION 7: ROLLING CONSTRUCTOR MOMENTUM (5-RACE WINDOWS)
# =====================================================================
print("Calculating 5-race rolling constructor momentum features...")

# Aggregate team performance per race weekend to avoid driver mixing
team_race = df.groupby(['season', 'race_round', 'TeamName']).agg(
    team_points_in_race=('Points', 'sum'),
    team_mech_dnfs_in_race=('is_mech_dnf', 'sum'),
    team_avg_quali_in_race=('quali_position', 'mean')
).reset_index()

# Sort team-level data chronologically
team_race = team_race.sort_values(['season', 'race_round']).reset_index(drop=True)

# Group by team and calculate rolling averages shifted by 1 to prevent lookahead bias
team_groups = team_race.groupby('TeamName')

team_race['team_avg_points_roll5'] = team_groups['team_points_in_race'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)
# Divide by 2 to express DNF rate as a ratio of retired cars per race (max 2 cars)
team_race['team_mech_dnf_rate_roll5'] = team_groups['team_mech_dnfs_in_race'].transform(
    lambda x: (x.rolling(window=5, min_periods=1).mean() / 2.0).shift(1)
)
team_race['team_avg_quali_roll5'] = team_groups['team_avg_quali_in_race'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)

# Fill NaNs with team midfield baselines
team_race['team_avg_points_roll5'] = team_race['team_avg_points_roll5'].fillna(4.0)
team_race['team_mech_dnf_rate_roll5'] = team_race['team_mech_dnf_rate_roll5'].fillna(0.1)
team_race['team_avg_quali_roll5'] = team_race['team_avg_quali_roll5'].fillna(12.0)

# Merge rolling team features back to driver rows
df = df.merge(
    team_race[['season', 'race_round', 'TeamName', 'team_avg_points_roll5', 'team_mech_dnf_rate_roll5', 'team_avg_quali_roll5']],
    on=['season', 'race_round', 'TeamName'],
    how='left'
)

# =====================================================================
# SECTION 8: INTERACTION FEATURES
# =====================================================================
print("Calculating advanced interaction features...")
# Elite Wet Modifier: Amplifies the importance of ELO when the race is wet
df['elite_wet_modifier'] = df['driver_elo_pre'] * df['is_wet_race']

# =====================================================================
# SECTION 9: SANITY CHECKS & LEADERBOARD EXPORT
# =====================================================================

# Print ELO Leaderboard at the end of 2025 (to verify Leclerc/Hamilton/Verstappen/Stroll ELO targets)
print("\n" + "="*50)
print("SANITY CHECK: ELO LEADERBOARD AT THE END OF 2025")
print("="*50)
df_2025 = df[df['season'] == 2025]
if not df_2025.empty:
    last_round_2025 = df_2025['race_round'].max()
    last_race_df = df_2025[df_2025['race_round'] == last_round_2025]
    leaderboard_2025 = last_race_df[['Abbreviation', 'TeamName', 'driver_elo_post']].sort_values('driver_elo_post', ascending=False)
    # Deduplicate to show one entry per driver (in case of sprint weekends or multi-mappings)
    leaderboard_2025 = leaderboard_2025.drop_duplicates(subset=['Abbreviation'])
    print(leaderboard_2025.to_string(index=False))
else:
    print("No 2025 data found to print leaderboard.")

# Print ELO Leaderboard at the end of the current dataset (2026)
print("\n" + "="*50)
print("SANITY CHECK: CURRENT ELO LEADERBOARD (2026)")
print("="*50)
current_leaderboard = pd.DataFrame([
    {'Driver': driver, 'ELO': elo} for driver, elo in driver_elo.items()
]).sort_values('ELO', ascending=False)
print(current_leaderboard.head(15).to_string(index=False))

# Export the final feature-engineered dataset
output_path = "rich_f1_dataset_with_features.csv"
df.to_csv(output_path, index=False)
print(f"\nFeature engineering complete. Final dataset shape: {df.shape}")
print(f"Dataset successfully exported to '{output_path}'.")
