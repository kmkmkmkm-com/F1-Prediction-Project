import fastf1
import fastf1.req
import pandas as pd
import numpy as np
import time
import os
import re
import concurrent.futures

# Enable the cache
fastf1.Cache.enable_cache('f1_cache')

# Monkey-patch FastF1's internal rate limiter to remove the artificial global limit
# on official F1 live timing data (weather, laps, sector times), while keeping a safe
# 200 calls/hour limit on the public Ergast/Jolpi community API.
fastf1.req._SessionWithRateLimiting._RATE_LIMITS = {
    re.compile(r"^https?://(\w+\.)?(ergast\.com|jolpi\.ca).*"): [
        fastf1.req._MinIntervalLimitDelay(0.25),
        fastf1.req._CallsPerIntervalLimitRaise(200, 60*60, "Ergast/Jolpi: 200 calls/h")
    ]
}


# =====================================================================
# SECTION 1: COMPREHENSIVE EXTRACTION FUNCTION
# =====================================================================

def get_final_quali_time(row):
    """Resolves the best qualifying time set by a driver."""
    if pd.notna(row['Q3']):
        return row['Q3']
    elif pd.notna(row['Q2']):
        return row['Q2']
    else:
        return row['Q1']

def extract_race_features(year, round_num):
    """Downloads results, qualifying times, sector gaps, FP2 pace, and weather."""
    try:
        # 1. Load Race Session (includes weather)
        race_session = fastf1.get_session(year, round_num, 'R')
        race_session.load(laps=False, telemetry=False, weather=True, messages=False)
        results = race_session.results

        if results is None or results.empty or 'Position' not in results.columns or results['Position'].notna().sum() < 3:
            raise ValueError("Race results are missing or incomplete")

        # 2. Load Qualifying Session (includes laps for sector times)
        quali_session = fastf1.get_session(year, round_num, 'Q')
        quali_session.load(laps=True, telemetry=False, weather=False, messages=False)
        quali_results = quali_session.results.copy()

        if quali_results is None or quali_results.empty or 'Position' not in quali_results.columns or quali_results['Position'].notna().sum() < 3:
            raise ValueError("Qualifying results are missing or incomplete")

        # 3. Extract Qualifying Times & Gaps
        quali_results['final_quali_time'] = quali_results.apply(get_final_quali_time, axis=1)
        quali_results['quali_time_s'] = quali_results['final_quali_time'].dt.total_seconds()
        quali_results['q3_appearance'] = quali_results['Q3'].notna().astype(int)

        pole_time = quali_results['quali_time_s'].min()
        quali_results['gap_to_pole_s'] = quali_results['quali_time_s'] - pole_time
        quali_results['gap_to_pole_pct'] = (quali_results['gap_to_pole_s'] / pole_time) * 100
        
        # Rename Position to avoid collision with race finish position
        quali_results = quali_results.rename(columns={'Position': 'quali_position'})

        # 4. Extract Qualifying Sector Gaps & Best Compound from Lap Data
        laps = quali_session.laps
        valid_laps = laps[laps['LapTime'].notna()]
        if not valid_laps.empty:
            idx = valid_laps.groupby('Driver')['LapTime'].idxmin()
            best_laps = valid_laps.loc[idx].copy()
            best_laps['sector1_s'] = best_laps['Sector1Time'].dt.total_seconds()
            best_laps['sector2_s'] = best_laps['Sector2Time'].dt.total_seconds()
            best_laps['sector3_s'] = best_laps['Sector3Time'].dt.total_seconds()
            
            # Gaps to the session's fastest sector times
            best_laps['quali_sector1_gap'] = best_laps['sector1_s'] - best_laps['sector1_s'].min()
            best_laps['quali_sector2_gap'] = best_laps['sector2_s'] - best_laps['sector2_s'].min()
            best_laps['quali_sector3_gap'] = best_laps['sector3_s'] - best_laps['sector3_s'].min()
            best_laps['best_compound_used'] = best_laps['Compound']
            
            quali_sector_data = best_laps[['Driver', 'quali_sector1_gap', 'quali_sector2_gap', 'quali_sector3_gap', 'best_compound_used']].rename(columns={'Driver': 'Abbreviation'})
        else:
            quali_sector_data = pd.DataFrame(columns=['Abbreviation', 'quali_sector1_gap', 'quali_sector2_gap', 'quali_sector3_gap', 'best_compound_used'])

        # 5. Extract FP2 Long-Run Pace & Tyre Deg Rate
        fp2_avg_long_run_pace = {}
        fp2_tyre_deg_rate = {}
        try:
            fp2_session = fastf1.get_session(year, round_num, 'FP2')
            fp2_session.load(laps=True, telemetry=False, weather=False, messages=False)
            fp2_laps = fp2_session.laps
            
            flying_laps = fp2_laps[fp2_laps['IsAccurate'] == True].copy()
            if not flying_laps.empty:
                for driver in flying_laps['Driver'].unique():
                    driver_laps = flying_laps[flying_laps['Driver'] == driver].copy()
                    stint_groups = driver_laps.groupby(['Stint', 'Compound'])
                    
                    best_stint_len = 0
                    best_pace = None
                    best_deg = None
                    
                    for (stint, compound), group in stint_groups:
                        if len(group) >= 5:
                            group = group.sort_values('LapNumber')
                            times = group['LapTime'].dt.total_seconds().values
                            laps_idx = np.arange(len(times))
                            
                            pace = np.mean(times)
                            slope = np.polyfit(laps_idx, times, 1)[0] if len(times) >= 3 else 0.0
                            
                            if len(group) > best_stint_len:
                                best_stint_len = len(group)
                                best_pace = pace
                                best_deg = slope
                                
                    if best_pace is not None:
                        fp2_avg_long_run_pace[driver] = best_pace
                        fp2_tyre_deg_rate[driver] = best_deg
        except Exception:
            pass

        # Convert FP2 dictionaries to a DataFrame
        fp2_drivers = list(set(list(fp2_avg_long_run_pace.keys()) + list(fp2_tyre_deg_rate.keys())))
        fp2_data = pd.DataFrame({
            'Abbreviation': fp2_drivers,
            'fp2_avg_long_run_pace': [fp2_avg_long_run_pace.get(d) for d in fp2_drivers],
            'fp2_tyre_deg_rate': [fp2_tyre_deg_rate.get(d) for d in fp2_drivers]
        })

        # 6. Extract Race Weather
        weather = race_session.weather_data
        if weather is not None and not weather.empty:
            air_temp = weather['AirTemp'].mean()
            track_temp = weather['TrackTemp'].mean()
            humidity = weather['Humidity'].mean()
            wind_speed = weather['WindSpeed'].mean()
            is_wet_race = int(weather['Rainfall'].any())
        else:
            air_temp, track_temp, humidity, wind_speed, is_wet_race = None, None, None, None, 0

        # 7. Merge All Data
        race_data = results[['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status', 'Points', 'Laps']].copy()
        quali_data = quali_results[['Abbreviation', 'quali_position', 'quali_time_s', 'gap_to_pole_s', 'gap_to_pole_pct', 'q3_appearance']].copy()

        merged = race_data.merge(quali_data, on='Abbreviation', how='left')
        merged = merged.merge(quali_sector_data, on='Abbreviation', how='left')
        
        if not fp2_data.empty:
            merged = merged.merge(fp2_data, on='Abbreviation', how='left')
        else:
            merged['fp2_avg_long_run_pace'] = None
            merged['fp2_tyre_deg_rate'] = None
            
        # Add weather columns
        merged['air_temp'] = air_temp
        merged['track_temp'] = track_temp
        merged['humidity'] = humidity
        merged['wind_speed'] = wind_speed
        merged['is_wet_race'] = is_wet_race
        
        # Derived on-the-fly features
        merged['quali_position_delta'] = merged['quali_position'] - merged['GridPosition']
        merged['podium_finish'] = merged['Position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        
        # Metadata
        merged['season'] = year
        merged['race_round'] = round_num
        
        # Sprint Weekend
        event_format = race_session.event['EventFormat']
        merged['is_sprint_weekend'] = 1 if event_format in ['sprint', 'sprint_shootout', 'sprint_qualifying'] else 0
        
        # Regulation Era
        if year < 2022:
            merged['reg_era'] = 'pre_2022'
        elif year <= 2025:
            merged['reg_era'] = '2022_2025'
        else:
            merged['reg_era'] = '2026_plus'

        return merged

    except Exception as e:
        print(f"FAILED: {year} Round {round_num} — {e}")
        return None

# =====================================================================
# SECTION 2: CALENDAR SETUP
# =====================================================================

print("Building calendar...")
all_events = []
for year in range(2018, 2027):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    schedule['season'] = year
    all_events.append(schedule[['season', 'RoundNumber', 'EventName', 'EventDate']])

calendar = pd.concat(all_events, ignore_index=True)
calendar['EventDate'] = pd.to_datetime(calendar['EventDate'])

today = pd.Timestamp.now()
past_races = calendar[calendar['EventDate'] < today].reset_index(drop=True)

# =====================================================================
# SECTION 3: CHECKPOINT LOADING AND EXTRACTION LOOP
# =====================================================================

checkpoint_path = 'rich_f1_dataset.csv'
expected_columns = [
    'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status', 'Points', 'Laps', 
    'quali_position', 'quali_time_s', 'gap_to_pole_s', 'gap_to_pole_pct', 'q3_appearance', 
    'quali_sector1_gap', 'quali_sector2_gap', 'quali_sector3_gap', 'best_compound_used',
    'fp2_avg_long_run_pace', 'fp2_tyre_deg_rate', 
    'air_temp', 'track_temp', 'humidity', 'wind_speed', 'is_wet_race', 'is_sprint_weekend',
    'quali_position_delta', 'podium_finish', 'season', 'race_round', 'reg_era'
]

# Load existing rich checkpoint if it exists
try:
    checkpoint_df = pd.read_csv(checkpoint_path)
    checkpoint_df['season'] = checkpoint_df['season'].astype(int)
    checkpoint_df['race_round'] = checkpoint_df['race_round'].astype(int)
    existing_keys = set(zip(checkpoint_df['season'], checkpoint_df['race_round']))
    total_rows = len(checkpoint_df)
    print(f"Found existing rich checkpoint with {total_rows} rows from {len(existing_keys)} races.")
except FileNotFoundError:
    checkpoint_df = pd.DataFrame()
    existing_keys = set()
    total_rows = 0
    print("No existing rich checkpoint found. Starting fresh.")

# Find remaining races to extract
all_past_races = list(zip(past_races['season'], past_races['RoundNumber'], past_races['EventName']))
missing_races = [(y, r, n) for y, r, n in all_past_races if (int(y), int(r)) not in existing_keys]

print(f"Total past races in calendar: {len(all_past_races)}")
print(f"Races remaining to extract: {len(missing_races)}\n")

# Start loop
still_failed = []
recovered_count = 0

for i, (year, rnd, name) in enumerate(missing_races):
    print(f"[{i+1}/{len(missing_races)}] Extracting {year} Round {rnd} - {name}...")
    
    # Run the rich extraction function inside a thread pool with a 25-second hard timeout
    df = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_race_features, year, rnd)
        try:
            df = future.result(timeout=25)
        except concurrent.futures.TimeoutError:
            print(f"   TIMEOUT: {year} Round {rnd} took longer than 25 seconds. Skipping...")
            still_failed.append((year, rnd, name))
            df = None
            
    if df is not None:
        df = df[expected_columns]
        
        # Append immediately
        if os.path.exists(checkpoint_path) and total_rows > 0:
            df.to_csv(checkpoint_path, mode='a', header=False, index=False)
        else:
            df.to_csv(checkpoint_path, index=False)
            
        recovered_count += 1
        total_rows += len(df)
        print(f"   Success: Added {len(df)} rows. Total rows in file: {total_rows}")
    elif df is None and (year, rnd, name) not in still_failed:
        still_failed.append((year, rnd, name))
        print(f"   FAILED: {year} Round {rnd}")
        
    # Throttled sleep to stay perfectly under the 500 calls/hour limit
    time.sleep(22.0)

print(f"\nRich extraction run complete.")
print(f"Successfully extracted: {recovered_count} races")
print(f"Failed or Timed Out: {len(still_failed)} races")
if still_failed:
    print("Races that need a retry:", still_failed)
