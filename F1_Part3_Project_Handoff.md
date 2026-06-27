# F1 ML Prediction Project — Part 3 Handoff Document

**Purpose of this document:** This is a complete context dump for continuing this project in a new AI chat session. It contains everything about the project goals, current technical state, the exact code written so far, the exact problem hit, the fix already applied, and instructions on how to work with this specific person. Read this fully before responding to anything.

---

## 1. Who This Is For

A second-year engineering student at **BITS Pilani Hyderabad** (2024 batch), preparing for internship season (September–October). Internship targets are big tech companies broadly (Google, Microsoft, Meta, Amazon, etc.).

Three parallel projects this summer, in priority order:
1. **F1 ML prediction project** (this doc covers this — main resume signal)
2. Mini shell in C (scaffolded, compiled, tested, pushed to GitHub — Tier 1 features like I/O redirection, pipes, background execution are next)
3. DSA prep — remaining topics: backtracking, greedy, tries, heap applications, remaining graph algorithms (toposort, DSU, SCC), DP (1D, 2D, knapsack, strings, trees) — Striver A2Z + LeetCode

He is a **deep F1 fan** — do not explain F1 concepts (tyres, DRS, sprint formats, etc.) to him. He needs help with the **data science / ML / CS side only**.

Certifications are lower priority than DSA and projects for the internship pipeline. System design matters more at placement stage (later).

---

## 2. How To Work With This Person (read before responding)

- **Explain concepts before writing code.** He wants to understand *why*, not just copy-paste. This was explicitly requested when Part 3 (ML) started — he said he was "at 0" on ML despite being comfortable with Part 1/2 (EDA, visualization).
- **Go slow, one step at a time.** Don't dump a giant pipeline in one message. Give one small piece, have him run it, see the actual output, confirm it's correct, then move to the next piece.
- **Validate with real output at every step.** Ask for `.shape`, `.value_counts()`, `.head()` results — don't assume code worked. He appreciates being asked to paste back actual numbers.
- **Test small before scaling.** Before looping over 180 races, we tested the extraction function on 4 races spanning different eras (2019, 2024, 2026) first. This pattern should continue — test on 1, then a handful, then the full set.
- **Be direct. Low tolerance for fluff, preamble, or unsolicited advice.** Don't moralize, don't pad responses, don't repeat back what he just said before answering.
- **He corrects assumptions in real time — and expects the correction to stick permanently.** Example: he got frustrated when data range assumptions seemed forgotten and explicitly said "WE HAVE ALL DATA TILL 2026 LATEST RACES EVERYTHING." This is now permanent: **always assume 2018–2026 full dataset, fully cached, no artificial cutoffs.**
- **Don't guess on factual/current things — verify.** When he mentioned the 2026 Bahrain/Saudi cancellations, the right move was to web-search and confirm rather than just agreeing or disagreeing blindly. Same applies to library errors (e.g. the FastF1 rate limit error was confirmed via a GitHub issue search, not guessed at).
- **He gets anxious when he feels context/continuity might be lost.** Reassure with action (fix it, save it), not just words.
- **When he says "save this to memory," that information is permanent context for all future conversations** — treat it as locked in.

---

## 3. Project Goal Recap

Build a **podium prediction model** for F1 races using FastF1 data (2018–2026). Pipeline:

1. Build dataset (FastF1 → flat table)
2. Clean & prepare data
3. Train models (Random Forest + XGBoost)
4. Evaluate (accuracy, F1, confusion matrix, feature importance)
5. Deploy via Streamlit app — user selects a race, gets predicted podium (P1/P2/P3) with confidence % and feature importance chart

**We are currently in Stage 1 (Build Dataset)**, specifically in the data extraction loop.

### Two-pass architecture (confirmed design decision)
- **Pass 1**: Raw extraction — loop through every race 2018–2026, pull raw `session.results` (grid position, finish position, status, points) + qualifying data. No rolling/derived features yet. Save to CSV checkpoint.
- **Pass 2**: Derived/rolling features computed on top of the assembled Pass 1 table — things like rolling ELO, last-5-race form, etc. These require chronological order across the full dataset, so they can only be computed *after* Pass 1 is fully assembled, not during extraction.

**Why split this way:** Debugging a 180-race loop in one pass is painful — one error far into the loop means starting over. Two passes let Pass 1 run once, cache to CSV, and Pass 2 iterate freely without re-hitting the FastF1 API.

---

## 4. Complete Feature Engineering Plan (target state — Pass 2 will build all of this)

One row = one driver per race. Binary target `podium_finish` (1 = P1/P2/P3) is primary; multi-class is a stretch goal.

### Category 1 — Grid & Qualifying
`grid_position`, `quali_time_s`, `gap_to_pole_s`, `gap_to_pole_pct`, `q3_appearance`, `quali_position_delta`

### Category 2 — Driver Form & Momentum
`driver_podiums_last5`, `driver_points_last5`, `driver_dnf_last5`, `driver_avg_finish_last5`, `driver_elo` (rolling ELO vs teammate: 30% quali + 70% race), `driver_circuit_avg_finish`, `driver_circuit_podium_rate`, `driver_wet_podium_rate`, `driver_season_points`, `driver_championship_pos`

### Category 3 — Constructor
`constructor_podiums_last5`, `constructor_points_last5`, `constructor_avg_quali_gap`, `constructor_dnf_rate`, `constructor_circuit_avg_finish`, `constructor_season_points`, `constructor_pit_stop_avg_ms`, `power_unit_supplier`

### Category 4 — Circuit Characteristics
`circuit_type` (street/power/technical/mixed), `overtaking_difficulty` (1–5), `drs_zones_count`, `circuit_lap_length_km`, `avg_safety_car_prob`, `circuit_avg_dnfs`, `is_street_circuit`

### Category 5 — Race Weekend Context
`race_round`, `season`, `reg_era` (`pre_2022` / `2022_2025` / `2026_plus`), `is_sprint_weekend`, `air_temp`, `track_temp`, `humidity`, `is_wet_race`, `wind_speed`

### Category 6 — Telemetry & Sector
`fp2_avg_long_run_pace`, `fp2_tyre_deg_rate`, `quali_sector1_gap`, `quali_sector2_gap`, `quali_sector3_gap`, `best_compound_used`

### Category 7 — Target
Primary: `podium_finish` (binary, 1 if P1/P2/P3). Stretch: `result_class` (multi-class: P1/P2/P3/points/no_points).

### What sets this apart from typical student projects
Driver ELO, FP2 long-run pace, tyre degradation rate, circuit-specific safety car probability, `reg_era` + 2026 sample upweighting, sector gaps.

### Deliberately excluded (and why)
- Pit strategy prediction — separate model territory, too complex
- Safety car occurrence prediction — unpredictable by definition
- Live telemetry — not available pre-race, which is when the prediction needs to run

---

## 5. 2026 Regulation Handling (critical, do not skip)

2026 introduced a major regulatory overhaul: new PU formula (electric-heavy hybrid), new aero rules, ground-effect changes, **new constructors (Audi entry, Cadillac entry)**, changed team identities. Confirmed live in the actual data already (2026 Round 1 results show `Cadillac` and `Audi` as constructors).

**Handling approach (confirmed):**
1. Add `reg_era` feature (`pre_2022` / `2022_2025` / `2026_plus`) so the model knows which ruleset applies
2. Upweight 2026 samples during training so recent performance patterns matter more than diluted historical averages
3. Be careful with constructor historical features for 2026 — pre-2026 constructor stats may be misleading since some are new entities entirely

**Also confirmed (verified via web search, not assumption):** The 2026 Bahrain and Saudi Arabian Grands Prix were **cancelled** due to the regional conflict, reducing the 2026 calendar to 22 races (not replaced). This is real-world data absence, not a FastF1 bug — confirmed the schedule API already excludes them entirely (they don't appear as skipped rounds 4/5, they're just absent).

---

## 6. Current Technical State — EXACT CODE WRITTEN SO FAR

### Environment
- Windows, **Antigravity IDE** (new — first time using it this session)
- Project folder: `C:\Users\Kshitij\Desktop\F1_Project\`
- Virtual environment: `.venv` at `C:\Users\Kshitij\Desktop\F1_Project\.venv\Scripts\python.exe`
- fastf1 version 3.8.3, installed correctly into `.venv` after initial mismatch where it landed in a global AppData Python 3.13 install instead
- Notebook name: `f1_part3_modeling.ipynb`
- Cache: `fastf1.Cache.enable_cache('f1_cache')`

### Step 1 — Calendar built
```python
import fastf1
import pandas as pd

fastf1.Cache.enable_cache('f1_cache')

all_events = []

for year in range(2018, 2027):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule['EventFormat'] != 'testing']
    schedule['season'] = year
    all_events.append(schedule[['season', 'RoundNumber', 'EventName', 'EventDate', 'EventFormat']])

calendar = pd.concat(all_events, ignore_index=True)
```
Result: **195 rows**. `EventFormat` values: `conventional`, `sprint`, `sprint_shootout`, `sprint_qualifying` (the last three are all sprint weekends, just renamed across seasons — all map to `is_sprint_weekend = True`).

Season counts: 2018=21, 2019=21, 2020=17 (COVID-shortened, correct), 2021=22, 2022=22, 2023=22, 2024=24, 2025=24, 2026=22 (2 cancelled: Bahrain, Saudi — confirmed real, not a bug).

### Step 2 — Past vs future race split
```python
today = pd.Timestamp.now()
calendar['EventDate'] = pd.to_datetime(calendar['EventDate'])

past_races = calendar[calendar['EventDate'] < today]
future_races = calendar[calendar['EventDate'] >= today]
past_races = past_races.reset_index(drop=True)
```
Result: **180 past races** usable (current date context: late June 2026).

### Step 3 — Single-race extraction logic (validated on 2024 Round 1 Bahrain — VER/PER/SAI podium matched real results)

```python
def get_final_quali_time(row):
    if pd.notna(row['Q3']):
        return row['Q3']
    elif pd.notna(row['Q2']):
        return row['Q2']
    else:
        return row['Q1']
```

### Step 4 — Wrapped into a function, tested on 4 races (2024 R1, 2024 R5, 2019 R3, 2026 R1) — all passed, podium counts validated (12 podiums across 4 races, exactly right)

### Step 5 — ⚠️ CRITICAL BUG HIT AND FIXED ⚠️

**The bug:** Ran the full 180-race loop. Got `FAILED: 2026 Round 7 — any API: 500 calls/h`. Only **20 races succeeded (402 rows)**, **160 races failed**.

**Root cause (confirmed via web search of FastF1's GitHub issues):** `session.load()` by default pulls laps, telemetry, weather, AND messages — not just `results`. Telemetry alone triggers dozens of API calls per driver per session against FastF1's backend (Ergast/Jolpica), which has a hard **500 calls/hour** quota. We were over-fetching massively — only needed `.results`, which is a tiny fraction of what was being pulled.

**The fix (already written, NOT YET RE-RUN AT SCALE):** Add `laps=False, telemetry=False, weather=False, messages=False` to every `.load()` call:

```python
def extract_race_features(year, round_num):
    try:
        race_session = fastf1.get_session(year, round_num, 'R')
        race_session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = race_session.results

        quali_session = fastf1.get_session(year, round_num, 'Q')
        quali_session.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_results = quali_session.results.copy()

        quali_results['final_quali_time'] = quali_results.apply(get_final_quali_time, axis=1)
        quali_results['quali_time_s'] = quali_results['final_quali_time'].dt.total_seconds()
        quali_results['q3_appearance'] = quali_results['Q3'].notna().astype(int)

        pole_time = quali_results['quali_time_s'].min()
        quali_results['gap_to_pole_s'] = quali_results['quali_time_s'] - pole_time
        quali_results['gap_to_pole_pct'] = (quali_results['gap_to_pole_s'] / pole_time) * 100

        race_data = results[['Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status', 'Points', 'Laps']].copy()
        quali_data = quali_results[['Abbreviation', 'quali_time_s', 'gap_to_pole_s', 'gap_to_pole_pct', 'q3_appearance']].copy()

        merged = race_data.merge(quali_data, on='Abbreviation', how='left')
        merged['podium_finish'] = merged['Position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        merged['season'] = year
        merged['race_round'] = round_num

        return merged

    except Exception as e:
        print(f"FAILED: {year} Round {round_num} — {e}")
        return None
```

**Important framing for next session:** The 160 "failed" races are NOT a data quality problem. They failed purely due to rate-limiting, not because the data doesn't exist or the logic is wrong. With the fix applied, most/all of them should succeed on retry — likely in 1-2 more batches given the ~80-90% reduction in API calls per race.

### Checkpoint saved
The 402 successfully-extracted rows (≈20 races) were saved to:
```
partial_race_data_checkpoint.csv
```
in the project working directory. **This file exists and must not be lost or overwritten carelessly.**

### The exact list of 160 failed races (need retry with the fixed function)

```python
failed_races = [(2018, 17, 'Japanese Grand Prix'), (2018, 18, 'United States Grand Prix'), (2018, 19, 'Mexican Grand Prix'), (2018, 20, 'Brazilian Grand Prix'), (2018, 21, 'Abu Dhabi Grand Prix'), (2019, 1, 'Australian Grand Prix'), (2019, 2, 'Bahrain Grand Prix'), (2019, 4, 'Azerbaijan Grand Prix'), (2019, 5, 'Spanish Grand Prix'), (2019, 6, 'Monaco Grand Prix'), (2019, 7, 'Canadian Grand Prix'), (2019, 8, 'French Grand Prix'), (2019, 9, 'Austrian Grand Prix'), (2019, 10, 'British Grand Prix'), (2019, 11, 'German Grand Prix'), (2019, 12, 'Hungarian Grand Prix'), (2019, 13, 'Belgian Grand Prix'), (2019, 14, 'Italian Grand Prix'), (2019, 15, 'Singapore Grand Prix'), (2019, 16, 'Russian Grand Prix'), (2019, 17, 'Japanese Grand Prix'), (2019, 18, 'Mexican Grand Prix'), (2019, 19, 'United States Grand Prix'), (2019, 20, 'Brazilian Grand Prix'), (2019, 21, 'Abu Dhabi Grand Prix'), (2020, 1, 'Austrian Grand Prix'), (2020, 2, 'Styrian Grand Prix'), (2020, 3, 'Hungarian Grand Prix'), (2020, 4, 'British Grand Prix'), (2020, 5, '70th Anniversary Grand Prix'), (2020, 6, 'Spanish Grand Prix'), (2020, 7, 'Belgian Grand Prix'), (2020, 8, 'Italian Grand Prix'), (2020, 9, 'Tuscan Grand Prix'), (2020, 10, 'Russian Grand Prix'), (2020, 11, 'Eifel Grand Prix'), (2020, 12, 'Portuguese Grand Prix'), (2020, 13, 'Emilia Romagna Grand Prix'), (2020, 14, 'Turkish Grand Prix'), (2020, 15, 'Bahrain Grand Prix'), (2020, 16, 'Sakhir Grand Prix'), (2020, 17, 'Abu Dhabi Grand Prix'), (2021, 1, 'Bahrain Grand Prix'), (2021, 2, 'Emilia Romagna Grand Prix'), (2021, 3, 'Portuguese Grand Prix'), (2021, 4, 'Spanish Grand Prix'), (2021, 5, 'Monaco Grand Prix'), (2021, 6, 'Azerbaijan Grand Prix'), (2021, 7, 'French Grand Prix'), (2021, 8, 'Styrian Grand Prix'), (2021, 9, 'Austrian Grand Prix'), (2021, 10, 'British Grand Prix'), (2021, 11, 'Hungarian Grand Prix'), (2021, 12, 'Belgian Grand Prix'), (2021, 13, 'Dutch Grand Prix'), (2021, 14, 'Italian Grand Prix'), (2021, 15, 'Russian Grand Prix'), (2021, 16, 'Turkish Grand Prix'), (2021, 17, 'United States Grand Prix'), (2021, 18, 'Mexico City Grand Prix'), (2021, 19, 'São Paulo Grand Prix'), (2021, 20, 'Qatar Grand Prix'), (2021, 21, 'Saudi Arabian Grand Prix'), (2021, 22, 'Abu Dhabi Grand Prix'), (2022, 1, 'Bahrain Grand Prix'), (2022, 2, 'Saudi Arabian Grand Prix'), (2022, 3, 'Australian Grand Prix'), (2022, 4, 'Emilia Romagna Grand Prix'), (2022, 5, 'Miami Grand Prix'), (2022, 6, 'Spanish Grand Prix'), (2022, 7, 'Monaco Grand Prix'), (2022, 8, 'Azerbaijan Grand Prix'), (2022, 9, 'Canadian Grand Prix'), (2022, 10, 'British Grand Prix'), (2022, 11, 'Austrian Grand Prix'), (2022, 12, 'French Grand Prix'), (2022, 13, 'Hungarian Grand Prix'), (2022, 14, 'Belgian Grand Prix'), (2022, 15, 'Dutch Grand Prix'), (2022, 16, 'Italian Grand Prix'), (2022, 17, 'Singapore Grand Prix'), (2022, 18, 'Japanese Grand Prix'), (2022, 19, 'United States Grand Prix'), (2022, 20, 'Mexico City Grand Prix'), (2022, 21, 'São Paulo Grand Prix'), (2022, 22, 'Abu Dhabi Grand Prix'), (2023, 1, 'Bahrain Grand Prix'), (2023, 2, 'Saudi Arabian Grand Prix'), (2023, 3, 'Australian Grand Prix'), (2023, 4, 'Azerbaijan Grand Prix'), (2023, 5, 'Miami Grand Prix'), (2023, 6, 'Monaco Grand Prix'), (2023, 7, 'Spanish Grand Prix'), (2023, 8, 'Canadian Grand Prix'), (2023, 9, 'Austrian Grand Prix'), (2023, 10, 'British Grand Prix'), (2023, 11, 'Hungarian Grand Prix'), (2023, 12, 'Belgian Grand Prix'), (2023, 13, 'Dutch Grand Prix'), (2023, 14, 'Italian Grand Prix'), (2023, 15, 'Singapore Grand Prix'), (2023, 16, 'Japanese Grand Prix'), (2023, 17, 'Qatar Grand Prix'), (2023, 18, 'United States Grand Prix'), (2023, 19, 'Mexico City Grand Prix'), (2023, 20, 'São Paulo Grand Prix'), (2023, 21, 'Las Vegas Grand Prix'), (2023, 22, 'Abu Dhabi Grand Prix'), (2024, 2, 'Saudi Arabian Grand Prix'), (2024, 3, 'Australian Grand Prix'), (2024, 4, 'Japanese Grand Prix'), (2024, 6, 'Miami Grand Prix'), (2024, 7, 'Emilia Romagna Grand Prix'), (2024, 8, 'Monaco Grand Prix'), (2024, 9, 'Canadian Grand Prix'), (2024, 10, 'Spanish Grand Prix'), (2024, 11, 'Austrian Grand Prix'), (2024, 12, 'British Grand Prix'), (2024, 13, 'Hungarian Grand Prix'), (2024, 14, 'Belgian Grand Prix'), (2024, 15, 'Dutch Grand Prix'), (2024, 16, 'Italian Grand Prix'), (2024, 17, 'Azerbaijan Grand Prix'), (2024, 18, 'Singapore Grand Prix'), (2024, 19, 'United States Grand Prix'), (2024, 20, 'Mexico City Grand Prix'), (2024, 21, 'São Paulo Grand Prix'), (2024, 22, 'Las Vegas Grand Prix'), (2024, 23, 'Qatar Grand Prix'), (2024, 24, 'Abu Dhabi Grand Prix'), (2025, 1, 'Australian Grand Prix'), (2025, 2, 'Chinese Grand Prix'), (2025, 3, 'Japanese Grand Prix'), (2025, 4, 'Bahrain Grand Prix'), (2025, 5, 'Saudi Arabian Grand Prix'), (2025, 6, 'Miami Grand Prix'), (2025, 7, 'Emilia Romagna Grand Prix'), (2025, 8, 'Monaco Grand Prix'), (2025, 9, 'Spanish Grand Prix'), (2025, 10, 'Canadian Grand Prix'), (2025, 11, 'Austrian Grand Prix'), (2025, 12, 'British Grand Prix'), (2025, 13, 'Belgian Grand Prix'), (2025, 14, 'Hungarian Grand Prix'), (2025, 15, 'Dutch Grand Prix'), (2025, 16, 'Italian Grand Prix'), (2025, 17, 'Azerbaijan Grand Prix'), (2025, 18, 'Singapore Grand Prix'), (2025, 19, 'United States Grand Prix'), (2025, 20, 'Mexico City Grand Prix'), (2025, 21, 'São Paulo Grand Prix'), (2025, 22, 'Las Vegas Grand Prix'), (2025, 23, 'Qatar Grand Prix'), (2025, 24, 'Abu Dhabi Grand Prix'), (2026, 2, 'Chinese Grand Prix'), (2026, 3, 'Japanese Grand Prix'), (2026, 4, 'Miami Grand Prix'), (2026, 5, 'Canadian Grand Prix'), (2026, 6, 'Monaco Grand Prix'), (2026, 7, 'Barcelona Grand Prix')]
```

---

## 7. IMMEDIATE NEXT STEP (where to resume)

1. Confirm kernel state — check if `all_race_data`, `failed_races`, and the helper functions are still in memory, OR reload from `partial_race_data_checkpoint.csv` + the static `failed_races` list above if starting fresh.
2. Retry **only the failed_races list** (not all 180 again — 20 are already cached and done) using the FIXED `extract_race_features` function (with `laps=False, telemetry=False, weather=False, messages=False`).
3. Add a small defensive delay between calls even with the fix (`time.sleep(1.5)` or similar) — not strictly required now but good practice.
4. Expect this to require 1-2 retry passes max given the ~80-90% reduction in API calls per race versus before.
5. Once all/most of 180 races are extracted, concatenate everything into one `full_dataset`, save it as the Pass 1 output CSV.
6. Move to Pass 2 — derived/rolling feature engineering (ELO, last-5 form, circuit history, etc.) — this requires the FULL chronological Pass 1 table to exist first.

Suggested retry code:
```python
import time

retry_results = []
still_failed = []

for year, rnd, name in failed_races:
    print(f"Retrying {year} Round {rnd} - {name}")
    data = extract_race_features(year, rnd)
    if data is not None:
        retry_results.append(data)
    else:
        still_failed.append((year, rnd, name))
    time.sleep(1.5)

print(f"Recovered: {len(retry_results)}")
print(f"Still failed: {len(still_failed)}")
```

---

## 8. Other Key Technical Learnings (apply throughout, not just Part 3)

- Never use `laps[Position==1]` to find the race winner — use `session.results` instead (silent error risk)
- Never use `'white'` for HARD tyre compound on dark-background visualizations
- `fig.show()` fails in VSCode with an nbformat error — always use `fig.write_html()` and open in browser
- Always validate with real printed output (`.shape`, `.value_counts()`) — never assume code worked
- Never hardcode outcomes or assume results — always derive from actual data

---

## 9. Instructions To The Next AI Assistant

1. **Read this entire document before responding to anything.** Don't ask him to re-explain context that's already here.
2. **Resume at Section 7** unless he says otherwise.
3. **Keep teaching concept-first, step-by-step**, exactly as described in Section 2. Do not write the entire Pass 2 pipeline in one shot — build it incrementally, validating with real output at each stage, same as Pass 1 was built (single race → small batch → full loop).
4. **Do not relitigate the 2018–2026 full dataset assumption** — this is settled and was a sore point before. Always use the full range.
5. **Do not relitigate 2026 regulation handling** — `reg_era` + sample upweighting is the confirmed approach (Section 5).
6. **When something might be a current/factual claim you're unsure about (library behavior, API quirks, real-world F1 events), search and verify rather than guessing** — this exact pattern (verifying the rate-limit error via GitHub issues, verifying the Bahrain/Saudi cancellation via web search) built trust earlier in this project. Keep doing it.
7. **He wants this project to be genuinely excellent** — a strong differentiator for big tech internship applications, not a generic student project. The "what sets this apart" notes in Section 4 reflect that ambition — keep holding that bar.
8. **Once Pass 1 (raw extraction) is fully complete and saved**, move to Pass 2 feature engineering, following the categories in Section 4 — also incrementally, one feature category at a time, not all at once.
