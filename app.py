import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="F1 Podium Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #0A0A0F;
            color: #F5F5F5;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #F5F5F5 !important;
            font-weight: 600;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }
        
        /* Subtitle */
        .subtitle {
            color: #9A9AA5;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Hide anchor links */
        .stMarkdown a.anchor-link {
            display: none !important;
        }
        
        /* Cards */
        .card {
            background-color: #15151C;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .card-header {
            font-size: 0.9rem;
            color: #9A9AA5;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 8px;
        }
        
        /* Labels for inputs */
        .stSelectbox label, .stSlider label, .stCheckbox label {
            color: #9A9AA5 !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        
        /* Primary Button */
        .stButton>button {
            background-color: #E10600 !important;
            color: #ffffff !important;
            border-radius: 6px;
            border: none !important;
            font-weight: 600;
            padding: 0.5rem 2rem;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #FF1A14 !important;
            transform: translateY(-1px);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background-color: transparent;
            border: none;
            color: #9A9AA5;
            font-weight: 500;
            padding: 0 8px;
            white-space: pre-wrap;
        }
        .stTabs [aria-selected="true"] {
            color: #F5F5F5 !important;
            border-bottom: 2px solid #E10600 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        
        /* Sliders */
        .stSlider [data-baseweb="slider"] > div {
            background-color: rgba(255,255,255,0.1);
            height: 4px;
        }
        .stSlider [data-baseweb="slider"] > div > div {
            background-color: #E10600;
        }
        .stSlider [role="slider"] {
            background-color: #E10600;
            border: 2px solid #15151C;
            box-shadow: none;
            width: 16px;
            height: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- DATA & COLORS ---
TEAM_COLORS = {
    'McLaren': '#FF8000',
    'Ferrari': '#E80020',
    'Red Bull Racing': '#3671C6',
    'Mercedes': '#27F4D2',
    'Aston Martin': '#00665E',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'Racing Bulls': '#6692FF',
    'Haas F1 Team': '#E6002B',
    'Haas': '#E6002B',
    'Kick Sauber': '#00E701',
    'Audi': '#F50537',
    'Cadillac': '#1A1A24',
    'AlphaTauri': '#2B4562',
    'Alfa Romeo': '#900000',
}

@st.cache_resource
def load_model():
    model_path = "f1_podium_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    df = pd.read_csv("rich_f1_dataset_with_features.csv")
    if 'circuit_type' in df.columns:
        df = pd.get_dummies(df, columns=['circuit_type'], prefix='circuit_type', dtype=int)
    else:
        for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
            if f'circuit_type_{c_type}' not in df.columns:
                df[f'circuit_type_{c_type}'] = 0
    return df

FEATURE_COLS = [
    'GridPosition', 'quali_position', 'gap_to_pole_pct', 'driver_elo_pre',
    'elo_vs_teammate_diff', 'driver_avg_finish_roll5', 'driver_avg_points_roll5',
    'driver_error_dnf_rate_roll5', 'driver_avg_quali_roll5', 'driver_avg_quali_gap_roll5',
    'team_avg_points_roll5', 'team_mech_dnf_rate_roll5', 'team_avg_quali_roll5',
    'is_wet_race', 'is_sprint_weekend', 'air_temp', 'track_temp',
    'circuit_type_Street', 'circuit_type_High-Downforce', 'circuit_type_Power',
    'elite_wet_modifier'
]

# --- MAIN APP LOGIC ---
def main():
    inject_custom_css()
    
    st.markdown("<h1>F1 Podium Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Powered by Machine Learning & ELO Ratings</div>", unsafe_allow_html=True)
    
    model = load_model()
    df = load_data()
    
    if model is None:
        st.error("Model file not found! Please run `python modeling.py` first.")
        return
    if df.empty:
        st.error("Dataset not found!")
        return

    tab1, tab2 = st.tabs(["Race Dashboard", "Interactive Simulator"])
    
    with tab1:
        render_race_dashboard(df, model)
        
    with tab2:
        render_interactive_simulator(df, model)


def highlight_podium(row):
    try:
        # Check if Actual Finish is a valid integer string 1-3
        val = row.get('Actual Finish', '')
        if str(val).isdigit() and int(val) <= 3:
            # Subtle left border + soft background tint
            return ['background-color: rgba(225, 6, 0, 0.08); border-left: 3px solid #E10600;' if i == 0 else 'background-color: rgba(225, 6, 0, 0.08);' for i in range(len(row))]
    except:
        pass
    return ['' for _ in row]


def render_race_dashboard(df, model):
    st.markdown("""
        <div class="card">
            <div class="card-header">Predict Race Outcome</div>
            <p style="color: #9A9AA5; font-size: 0.95rem; margin-bottom: 20px;">Select a historical race to visualize the model's pre-race predictions.</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        seasons = sorted(df['season'].unique(), reverse=True)
        selected_season = st.selectbox("Season", seasons)
    with col2:
        season_df = df[df['season'] == selected_season]
        events = season_df['EventName'].unique()
        selected_event = st.selectbox("Grand Prix", events)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Generate Predictions"):
        race_df = season_df[season_df['EventName'] == selected_event].copy()
        for col in FEATURE_COLS:
            if col not in race_df.columns:
                race_df[col] = 0.0
        X = race_df[FEATURE_COLS].fillna(0.0)
        
        # XGBRanker outputs arbitrary scores, use Softmax to get Power Ratings
        raw_scores = model.predict(X)
        temperature = 1.5 
        exp_scores = np.exp(raw_scores / temperature)
        power_ratings = exp_scores / np.sum(exp_scores)
        
        race_df['Power_Rating'] = power_ratings
        
        display_df = race_df[['Abbreviation', 'TeamName', 'GridPosition', 'Power_Rating', 'Position']].copy()
        display_df['Power_Rating_Pct'] = (display_df['Power_Rating'] * 100).round(1).astype(str) + '%'
        display_df = display_df.sort_values('Power_Rating', ascending=False).reset_index(drop=True)
        
        # Clean dataframe for rendering
        styled_df = display_df[['Abbreviation', 'TeamName', 'GridPosition', 'Power_Rating_Pct', 'Position']].copy()
        styled_df = styled_df.rename(columns={'Position': 'Actual Finish', 'Power_Rating_Pct': 'Power Rating'})
        
        # Format floats to int strings
        styled_df['Actual Finish'] = styled_df['Actual Finish'].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.','',1).isdigit() else '')
        styled_df['GridPosition'] = styled_df['GridPosition'].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.','',1).isdigit() else '')
        
        # Split layout for results
        res_col1, res_col2 = st.columns([3, 2])
        
        with res_col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>Predicted Power Ratings</div>", unsafe_allow_html=True)
            
            fig = px.bar(
                display_df, 
                x='Abbreviation', 
                y='Power_Rating', 
                color='TeamName', 
                color_discrete_map=TEAM_COLORS,
                text=display_df['Power_Rating_Pct']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font_family='Inter',
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, showticklabels=False, title=''),
                xaxis=dict(showgrid=False, zeroline=False, title='', tickfont=dict(color='#9A9AA5'))
            )
            fig.update_traces(textposition='outside', textfont=dict(color='#F5F5F5', size=11))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>Grid Breakdown</div>", unsafe_allow_html=True)
            
            # Apply styling
            final_table = styled_df.style.apply(highlight_podium, axis=1)
            
            # Muted header row styling is applied globally via streamlit theme but we can add specific CSS if needed.
            st.dataframe(final_table, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

def render_interactive_simulator(df, model):
    latest_season = df['season'].max()
    latest_df = df[df['season'] == latest_season]
    drivers = sorted(latest_df['Abbreviation'].unique())
    
    st.markdown("""
        <div class="card" style="margin-bottom: 24px;">
            <div class="card-header">Baseline Context</div>
            <p style="color: #9A9AA5; font-size: 0.95rem; margin-bottom: 16px;">Pre-load the most recent historical stats for any driver on the grid.</p>
    """, unsafe_allow_html=True)
    selected_driver = st.selectbox("Driver Baseline", drivers, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    driver_stats = latest_df[latest_df['Abbreviation'] == selected_driver].iloc[-1]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='card'><div class='card-header'>Race Weekend Info</div>", unsafe_allow_html=True)
        grid_pos = st.slider("Starting Grid Position", 1, 20, int(driver_stats['GridPosition']) if pd.notna(driver_stats['GridPosition']) else 5)
        quali_pos = st.slider("Qualifying Position", 1, 20, int(driver_stats['quali_position']) if pd.notna(driver_stats['quali_position']) else 5)
        circuit_type = st.selectbox("Circuit Type", ["Balanced", "High-Downforce", "Power", "Street"])
        is_wet = st.checkbox("Wet Race", value=False)
        is_sprint = st.checkbox("Sprint Weekend", value=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'><div class='card-header'>Driver Form & ELO</div>", unsafe_allow_html=True)
        driver_elo = st.slider("Pre-Race ELO", 1000, 2200, int(driver_stats['driver_elo_pre']) if pd.notna(driver_stats['driver_elo_pre']) else 1500)
        teammate_diff = st.slider("ELO vs Teammate", -500, 500, int(driver_stats['elo_vs_teammate_diff']) if pd.notna(driver_stats['elo_vs_teammate_diff']) else 0)
        driver_finish = st.slider("Avg Finish (Last 5)", 1.0, 20.0, float(driver_stats['driver_avg_finish_roll5']) if pd.notna(driver_stats['driver_avg_finish_roll5']) else 10.0, step=0.1)
        driver_error = st.slider("Driver Error DNF Rate (Last 5)", 0.0, 1.0, float(driver_stats['driver_error_dnf_rate_roll5']) if pd.notna(driver_stats['driver_error_dnf_rate_roll5']) else 0.05, step=0.01)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><div class='card-header'>Constructor Form</div>", unsafe_allow_html=True)
        team_points = st.slider("Team Avg Points (Last 5)", 0.0, 44.0, float(driver_stats['team_avg_points_roll5']) if pd.notna(driver_stats['team_avg_points_roll5']) else 5.0, step=0.5)
        team_mech = st.slider("Team Mechanical DNF Rate", 0.0, 1.0, float(driver_stats['team_mech_dnf_rate_roll5']) if pd.notna(driver_stats['team_mech_dnf_rate_roll5']) else 0.1, step=0.01)
        team_quali = st.slider("Team Avg Quali (Last 5)", 1.0, 20.0, float(driver_stats['team_avg_quali_roll5']) if pd.notna(driver_stats['team_avg_quali_roll5']) else 10.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    # Build input vector
    input_data = {col: 0.0 for col in FEATURE_COLS}
    input_data['GridPosition'] = grid_pos
    input_data['quali_position'] = quali_pos
    input_data['driver_elo_pre'] = driver_elo
    input_data['elo_vs_teammate_diff'] = teammate_diff
    input_data['driver_avg_finish_roll5'] = driver_finish
    input_data['driver_error_dnf_rate_roll5'] = driver_error
    input_data['team_avg_points_roll5'] = team_points
    input_data['team_mech_dnf_rate_roll5'] = team_mech
    input_data['team_avg_quali_roll5'] = team_quali
    input_data['is_wet_race'] = int(is_wet)
    input_data['is_sprint_weekend'] = int(is_sprint)
    
    if circuit_type != "Balanced":
        input_data[f'circuit_type_{circuit_type}'] = 1.0
        
    input_data['air_temp'] = 25.0
    input_data['track_temp'] = 35.0
    input_data['gap_to_pole_pct'] = 1.5
    input_data['driver_avg_quali_roll5'] = quali_pos
    input_data['driver_avg_quali_gap_roll5'] = 1.5

    input_data['elite_wet_modifier'] = driver_elo * int(is_wet)

    # To calculate a true Softmax Power Rating, we must predict the entire grid
    sim_grid = latest_df.copy()
    
    # Fill defaults for the whole grid if missing
    for col in FEATURE_COLS:
        if col not in sim_grid.columns:
            sim_grid[col] = 0.0
            
    # Apply global race conditions to the entire grid
    sim_grid['is_wet_race'] = int(is_wet)
    sim_grid['is_sprint_weekend'] = int(is_sprint)
    sim_grid['elite_wet_modifier'] = sim_grid['driver_elo_pre'] * int(is_wet)
    
    for c_type in ['Street', 'High-Downforce', 'Power', 'Balanced']:
        if f'circuit_type_{c_type}' in sim_grid.columns:
            sim_grid[f'circuit_type_{c_type}'] = 1.0 if c_type == circuit_type else 0.0
            
    # Replace the selected driver's specific stats with our slider inputs
    driver_idx = sim_grid[sim_grid['Abbreviation'] == selected_driver].index
    if not driver_idx.empty:
        idx = driver_idx[0]
        for key, val in input_data.items():
            sim_grid.loc[idx, key] = val

    X_sim = sim_grid[FEATURE_COLS].fillna(0.0)
    
    # Predict and Softmax
    raw_scores = model.predict(X_sim)
    temperature = 1.5
    exp_scores = np.exp(raw_scores / temperature)
    power_ratings = exp_scores / np.sum(exp_scores)
    
    sim_grid['Power_Rating'] = power_ratings
    
    if not driver_idx.empty:
        prob = sim_grid.loc[driver_idx[0], 'Power_Rating']
    else:
        prob = 0.0
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "", 'font': {'size': 64, 'color': '#F5F5F5', 'family': 'Inter'}},
        title = {'text': f"{selected_driver} Power Rating (Out of 100)", 'font': {'size': 14, 'color': '#9A9AA5', 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)", 'tickfont': {'color': 'rgba(0,0,0,0)'}},
            'bar': {'color': "#E10600", 'thickness': 1},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': "rgba(255, 255, 255, 0.05)"}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        font={'family': 'Inter'}, 
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
