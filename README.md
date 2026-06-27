# Formula 1 Podium Predictor & Interactive Simulator

An end-to-end Machine Learning pipeline that predicts Formula 1 race outcomes using advanced driver ELO modeling, team momentum rolling averages, and custom aerodynamic gap heuristics. The project features an interactive web dashboard built in Streamlit.

## 🏎️ Overview

Predicting modern Formula 1 is incredibly difficult because a "dumb baseline" (predicting that drivers finish exactly where they qualify) is often highly accurate. To beat this baseline and accurately predict the *chaos* of the race, this model implements state-of-the-art predictive features:

* **Custom Driver ELO Ratings:** Dynamic rating system evaluating a driver's true skill independently of their car by heavily weighting their performance against their direct teammate.
* **Team Momentum (5-Race Rolling Average):** Captures mid-season aerodynamic development by tracking a team's pace and point-scoring momentum over the previous 5 races.
* **Regulation Reset Logic:** The model mathematically understands when the FIA radically changes car regulations (e.g., 2022) and automatically discounts stale historical momentum.
* **Qualifying Penalty Tracking:** Accurately assesses the true pace of a driver who qualified highly but took an engine penalty and started further back on the grid.

## 🛠️ Technology Stack
* **Python** (Core Data Processing)
* **XGBoost (XGBRanker)** (Machine Learning Engine)
* **Pandas / NumPy** (Feature Engineering)
* **Streamlit** (Web Application & UI)
* **FastF1 API** (Historical Telemetry & Race Data)

## 📁 Project Structure

* `app.py`: The main Streamlit web application featuring a Historical Race Dashboard and a manual Interactive Simulator.
* `modeling.py`: The core ML training script that builds, trains, and evaluates the `XGBRanker` model.
* `feature_engineering.py` / `rich_extractor.py`: Data pipelines responsible for downloading raw FastF1 telemetry and engineering the ELO and momentum features.
* `f1_podium_model.joblib`: The serialized, production-ready XGBoost model.

## 🚀 How to Run Locally

1. **Install Dependencies:**
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `streamlit`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, and `fastf1` installed)*

2. **Launch the Dashboard:**
   Start the Streamlit application from your terminal:
   ```bash
   python -m streamlit run app.py
   ```

3. **Navigate the App:**
   * **Historical Race Dashboard:** Select a past Grand Prix to instantly generate AI win probabilities based on the exact grid conditions and momentum of that specific weekend.
   * **Interactive Simulator:** Manually tweak grid positions, driver ELOs, and rain conditions to simulate custom "what-if" scenarios (e.g., mid-season upgrades or wet races).

## 🧠 Model Architecture

The model uses a Pairwise Ranking algorithm (`XGBRanker`). Rather than trying to predict the absolute lap time of a driver, the model is trained to rank drivers against each other. It takes the aerodynamic pace, ELO, and grid position of all 20 drivers and outputs a Softmax Probability distribution (Power Rating) dictating their likelihood of winning. 

* **Why XGBRanker?** Standard regression fails in F1 because track speeds vary wildly (Monza vs. Monaco). Ranking models understand that the goal isn't to be fast; the goal is to be faster than the other 19 cars on the track.
