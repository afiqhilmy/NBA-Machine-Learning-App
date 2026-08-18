import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import os

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NBA Expected Points Per Game (xPPG) Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .main-header { font-size: 32px; font-weight: bold; color: #1E293B; margin-bottom: 5px; }
    .sub-header { font-size: 16px; color: #64748B; margin-bottom: 25px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏀 NBA Expected Points Per Game (xPPG) Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decision-support system for player baseline evaluation and scouting projections.</div>', unsafe_allow_html=True)

# --- 2. DATA GENERATOR & MODEL PIPELINE ---
@st.cache_data
def get_dataset_and_models():
    """
    Generates synthetic benchmark dataset matching NBA player-season distributions
    and trains the three candidate models if external CSV is missing.
    """
    np.random.seed(42)

    # Define feature columns expected by the model
    feature_cols = ['GP', 'MPG', 'FGA_pg', '3PA_pg', 'FTA_pg', 'FG%', '3P%', 'FT%', 'APG', 'RPG', 'TOPG']

    # Try to load real dataset if present; otherwise fall back to synthetic generation
    try:
        df = pd.read_csv('nba_player_seasons_processed.csv')
        # Ensure required columns exist
        required = set(feature_cols + ['PPG', 'team', 'player', 'season'])
        if not required.issubset(df.columns):
            raise ValueError('CSV missing required columns')
        # Keep only needed columns (preserve extra for display)
        df = df.copy()
        teams = sorted(df['team'].dropna().unique())
    except Exception:
        # Fallback: synthetic data
        teams = [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]
        n_samples = 1200
        player_name_pool = [f"Player_{i}" for i in range(1000, 1300)]
        player_names = np.random.choice(player_name_pool, size=n_samples)
        data = []
        for pname in player_names:
            team = np.random.choice(teams)
            season = np.random.choice([2022, 2023, 2024, 2025, 2026])
            gp = np.random.randint(15, 82)
            mpg = np.random.uniform(10.0, 38.0)
            fga = mpg * np.random.uniform(0.35, 0.65)
            three_pa = fga * np.random.uniform(0.1, 0.5)
            fta = fga * np.random.uniform(0.15, 0.4)
            fg_pct = np.random.uniform(0.38, 0.58)
            three_pct = np.random.uniform(0.28, 0.42)
            ft_pct = np.random.uniform(0.60, 0.90)
            apg = mpg * np.random.uniform(0.05, 0.3)
            rpg = mpg * np.random.uniform(0.08, 0.35)
            topg = mpg * np.random.uniform(0.03, 0.15)
            ppg = (fga * fg_pct * 2) + (three_pa * three_pct * 1) + (fta * ft_pct * 0.75) + np.random.normal(0, 1.2)
            ppg = max(2.0, ppg)
            total_pts = int(ppg * gp)
            data.append({
                'player': pname,
                'team': team,
                'season': season,
                'GP': gp,
                'MPG': mpg,
                'Total_PTS': total_pts,
                'PPG': ppg,
                'FGA_pg': fga,
                '3PA_pg': three_pa,
                'FTA_pg': fta,
                'FG%': fg_pct,
                '3P%': three_pct,
                'FT%': ft_pct,
                'APG': apg,
                'RPG': rpg,
                'TOPG': topg
            })
        df = pd.DataFrame(data)
    # Prepare feature matrix and labels
    X = df[feature_cols]
    y = df['PPG']
    # Use player name for grouping so UI selection by player name aligns with splits
    groups = df['player']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
    X_test_scaled = scaler.transform(X.iloc[test_idx])
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train 3 models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)
    }
    
    eval_metrics = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        eval_metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
    return df, models, scaler, feature_cols, eval_metrics, teams

# Try to read CSV upfront to prefer real player names in the UI
csv_path = 'nba_player_seasons_processed.csv'
csv_available = False
df_csv = None
if os.path.exists(csv_path):
    try:
        df_csv = pd.read_csv(csv_path)
        # minimal validation
        if {'player', 'team', 'season'}.issubset(df_csv.columns):
            csv_available = True
    except Exception:
        csv_available = False

df, models, scaler, feature_cols, eval_metrics, all_teams = get_dataset_and_models()

# Mapping from team code to full team name
TEAM_FULL_NAMES = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Display names and reverse lookup
TEAM_DISPLAY_NAMES = [TEAM_FULL_NAMES.get(code, code) for code in all_teams]
TEAM_CODE_BY_NAME = {v: k for k, v in TEAM_FULL_NAMES.items()}

# If CSV is available, build team/player lists from it for the UI so names are real
if csv_available:
    csv_teams = sorted(df_csv['team'].dropna().unique())
    CSV_TEAM_DISPLAY = [TEAM_FULL_NAMES.get(code, code) for code in csv_teams]
    CSV_TEAM_CODE_BY_NAME = {v: k for k, v in TEAM_FULL_NAMES.items()}
    CSV_PLAYERS_BY_TEAM = {code: sorted(df_csv[df_csv['team'] == code]['player'].unique()) for code in csv_teams}

# UI controls: allow forcing a reload from CSV (clears cache and reruns)
# (Sidebar reload button and data-source info removed)

# --- 3. GLOBAL MODEL SELECTION (MAIN BODY) ---
st.markdown("### ⚙️ Select Machine Learning Model")
selected_model_name = st.radio(
    "Choose algorithm for expected scoring baseline prediction:",
    ["Linear Regression", "Random Forest", "XGBoost"],
    horizontal=True
)

active_model = models[selected_model_name]
active_metrics = eval_metrics[selected_model_name]

st.markdown("---")

# --- 4. MAIN PAGE NAVIGATION ---
page_tab1, page_tab2 = st.tabs(["👤 Specific Player Analysis", "🔮 General Scouting & Custom Player"])

# Helper function to render model performance card
def render_model_eval_card(name, metrics):
    st.markdown("#### 📐 Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f} PPG")
    c2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f} PPG")
    c3.metric("R² Score", f"{metrics['R2']:.3f}")

# ==============================================================================
# PAGE 1: SPECIFIC PLAYER ANALYSIS
# ==============================================================================
with page_tab1:
    st.markdown("### 1. Filter & Select Player")
    col_team, col_player = st.columns(2)
    
    with col_team:
        if csv_available:
            selected_team_display = st.selectbox("Select Team:", sorted(set(CSV_TEAM_DISPLAY)), key="p1_team")
            selected_team = CSV_TEAM_CODE_BY_NAME.get(selected_team_display, selected_team_display)
            team_players = CSV_PLAYERS_BY_TEAM.get(selected_team, [])
        else:
            selected_team_display = st.selectbox("Select Team:", sorted(set(TEAM_DISPLAY_NAMES)), key="p1_team")
            selected_team = TEAM_CODE_BY_NAME.get(selected_team_display, selected_team_display)
            team_players = sorted(df[df['team'] == selected_team]['player'].unique())
    with col_player:
        selected_player = st.selectbox("Select Player:", team_players if team_players else ["No Players"], key="p1_player")
        
    if selected_player != "No Players":
        if csv_available:
            player_df = df_csv[(df_csv['team'] == selected_team) & (df_csv['player'] == selected_player)].sort_values('season', ascending=False)
        else:
            player_df = df[(df['team'] == selected_team) & (df['player'] == selected_player)].sort_values('season', ascending=False)
        latest_season_data = player_df.iloc[0]
        # Build safe defaults for season features (use player's latest season when available,
        # otherwise fall back to dataset mean or sensible global defaults)
        global_defaults = {'MPG':25.0,'FGA_pg':10.0,'3PA_pg':4.0,'FTA_pg':3.0,'FG%':0.45,'3P%':0.35,'FT%':0.78,'APG':3.0,'RPG':4.0,'TOPG':1.8}
        season_defaults = {}
        for feat in feature_cols:
            if feat in player_df.columns:
                v = player_df.iloc[0].get(feat, np.nan)
                if pd.notna(v):
                    season_defaults[feat] = float(v)
                    continue
            if feat in df.columns:
                try:
                    season_defaults[feat] = float(df[feat].mean())
                except Exception:
                    season_defaults[feat] = global_defaults.get(feat, 0.0)
            else:
                season_defaults[feat] = global_defaults.get(feat, 0.0)
        
        st.markdown("---")
        st.markdown(f"### 2. Historical Career KPIs: {selected_player}")
        
        # Ensure Total_PTS exists (compute from PPG * GP when missing)
        player_df = player_df.copy()
        if 'Total_PTS' not in player_df.columns and {'PPG', 'GP'}.issubset(player_df.columns):
            player_df['Total_PTS'] = (player_df['PPG'] * player_df['GP']).round().astype(int)

        # Display Season KPI Table (show full team names)
        cols_to_show = [c for c in ['season', 'team', 'Total_PTS', 'PPG', 'MPG', 'GP'] if c in player_df.columns]
        kpi_display = player_df[cols_to_show].copy()
        if 'team' in kpi_display.columns:
            kpi_display['team'] = kpi_display['team'].map(TEAM_FULL_NAMES)
        # Rename columns for display (only for those present)
        rename_map = {}
        if 'season' in kpi_display.columns: rename_map['season'] = 'Season'
        if 'team' in kpi_display.columns: rename_map['team'] = 'Team'
        if 'Total_PTS' in kpi_display.columns: rename_map['Total_PTS'] = 'Team Total Points'
        if 'PPG' in kpi_display.columns: rename_map['PPG'] = 'PPG'
        if 'MPG' in kpi_display.columns: rename_map['MPG'] = 'Minutes Played (MPG)'
        if 'GP' in kpi_display.columns: rename_map['GP'] = 'Games Played'
        kpi_display = kpi_display.rename(columns=rename_map)
        st.dataframe(kpi_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 3. Input Player Workload & Metrics")
        st.info("Inputs default to the player's most recent season baseline. Adjust features to simulate scenarios.")
        
        # Input 1: Games Played FIRST
        input_gp = st.number_input(
            "1. Games Played (GP)", 
            min_value=1, max_value=82, 
            value=int(latest_season_data['GP']), 
            step=1
        )
        
        # Other Features Layout
        st.markdown("##### Workload & Secondary Stats")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            input_mpg = st.slider("Minutes Per Game (MPG)", 5.0, 42.0, float(season_defaults.get('MPG',25.0)), 0.5)
            input_fga = st.slider("Field Goal Attempts (FGA/G)", 1.0, 30.0, float(season_defaults.get('FGA_pg',10.0)), 0.5)
        with col_w2:
            input_3pa = st.slider("3-Point Attempts (3PA/G)", 0.0, 15.0, float(season_defaults.get('3PA_pg',4.0)), 0.5)
            input_fta = st.slider("Free Throw Attempts (FTA/G)", 0.0, 15.0, float(season_defaults.get('FTA_pg',3.0)), 0.5)
        with col_w3:
            input_apg = st.slider("Assists Per Game (APG)", 0.0, 12.0, float(season_defaults.get('APG',3.0)), 0.2)
            input_rpg = st.slider("Rebounds Per Game (RPG)", 0.0, 15.0, float(season_defaults.get('RPG',4.0)), 0.2)
            input_topg = st.slider("Turnovers Per Game (TOPG)", 0.0, 6.0, float(season_defaults.get('TOPG',1.8)), 0.1)
            
        st.markdown("##### Shooting Efficiency")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            input_fg_pct = st.number_input("Field Goal % (0-100)", 0.0, 100.0, float(season_defaults.get('FG%',0.45)) * 100.0, 0.1, format="%.1f")
        with col_e2:
            input_3p_pct = st.number_input("3-Point % (0-100)", 0.0, 100.0, float(season_defaults.get('3P%',0.35)) * 100.0, 0.1, format="%.1f")
        with col_e3:
            input_ft_pct = st.number_input("Free Throw % (0-100)", 0.0, 100.0, float(season_defaults.get('FT%',0.78)) * 100.0, 0.1, format="%.1f")

        # Assemble Input Array
        # Convert percent inputs (0-100) to decimal (0-1) for the model
        input_dict = {
            'GP': input_gp, 'MPG': input_mpg, 'FGA_pg': input_fga, '3PA_pg': input_3pa,
            'FTA_pg': input_fta, 'FG%': input_fg_pct / 100.0, '3P%': input_3p_pct / 100.0, 'FT%': input_ft_pct / 100.0,
            'APG': input_apg, 'RPG': input_rpg, 'TOPG': input_topg
        }
        
        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_scaled = scaler.transform(input_df)
        predicted_ppg = active_model.predict(input_scaled)[0]
        actual_ppg = float(latest_season_data['PPG'])
        xpts = actual_ppg - predicted_ppg
        
        st.markdown("---")
        st.markdown("### 4. Output & Performance Evaluation")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Expected Baseline (xPPG)", f"{predicted_ppg:.1f} PPG")
        with res_col2:
            st.metric("Actual Baseline", f"{actual_ppg:.1f} PPG")
        with res_col3:
            st.metric("Points Over Expected (xPTS)", f"{xpts:+.1f} PPG")
            
        # Performance Badge Evaluation
        if xpts > 1.2:
            st.success("🔥 **Performance Status:** OUTPERFORMED baseline (Hyper-Efficient Scorer)")
        elif xpts < -1.2:
            st.error("⚠️ **Performance Status:** UNDERPERFORMED baseline (Inefficient / Usage Heavy)")
        else:
            st.info("🎯 **Performance Status:** PERFECTLY PERFORMED (Aligns with Workload Baseline)")

        st.markdown("---")
        render_model_eval_card(selected_model_name, active_metrics)

# ==============================================================================
# PAGE 2: GENERAL SCOUTING & CUSTOM PLAYER
# ==============================================================================
with page_tab2:
    st.markdown("### 1. Select Team Context")
    scout_team_display = st.selectbox("Target Team Context:", sorted(set(TEAM_DISPLAY_NAMES)), key="p2_team")
    scout_team = TEAM_CODE_BY_NAME.get(scout_team_display, scout_team_display)
    
    st.markdown("---")
    st.markdown("### 2. Key In Custom / Hypothetical Player Workload")
    
    # Input 1: Games Played FIRST
    scout_gp = st.number_input(
        "1. Projected Games Played (GP)", 
        min_value=1, max_value=82, 
        value=65, 
        step=1,
        key="scout_gp"
    )
    
    st.markdown("##### Projected Workload & Secondary Metrics")
    sc_w1, sc_w2, sc_w3 = st.columns(3)
    with sc_w1:
        scout_mpg = st.slider("Minutes Per Game (MPG)", 5.0, 42.0, 28.0, 0.5, key="scout_mpg")
        scout_fga = st.slider("Field Goal Attempts (FGA/G)", 1.0, 30.0, 12.0, 0.5, key="scout_fga")
    with sc_w2:
        scout_3pa = st.slider("3-Point Attempts (3PA/G)", 0.0, 15.0, 4.5, 0.5, key="scout_3pa")
        scout_fta = st.slider("Free Throw Attempts (FTA/G)", 0.0, 15.0, 3.5, 0.5, key="scout_fta")
    with sc_w3:
        scout_apg = st.slider("Assists Per Game (APG)", 0.0, 12.0, 3.0, 0.2, key="scout_apg")
        scout_rpg = st.slider("Rebounds Per Game (RPG)", 0.0, 15.0, 4.0, 0.2, key="scout_rpg")
        scout_topg = st.slider("Turnovers Per Game (TOPG)", 0.0, 6.0, 1.8, 0.1, key="scout_topg")
        
    st.markdown("##### Projected Shooting Efficiency")
    sc_e1, sc_e2, sc_e3 = st.columns(3)
    with sc_e1:
        scout_fg_pct = st.number_input("Field Goal % (0-100)", 0.0, 100.0, 46.0, 0.1, format="%.1f", key="scout_fg")
    with sc_e2:
        scout_3p_pct = st.number_input("3-Point % (0-100)", 0.0, 100.0, 36.0, 0.1, format="%.1f", key="scout_3p")
    with sc_e3:
        scout_ft_pct = st.number_input("Free Throw % (0-100)", 0.0, 100.0, 80.0, 0.1, format="%.1f", key="scout_ft")

    # Predict Custom Player xPPG
    # Convert percent inputs (0-100) to decimal (0-1) for the model
    scout_dict = {
        'GP': scout_gp, 'MPG': scout_mpg, 'FGA_pg': scout_fga, '3PA_pg': scout_3pa,
        'FTA_pg': scout_fta, 'FG%': scout_fg_pct / 100.0, '3P%': scout_3p_pct / 100.0, 'FT%': scout_ft_pct / 100.0,
        'APG': scout_apg, 'RPG': scout_rpg, 'TOPG': scout_topg
    }
    
    scout_df = pd.DataFrame([scout_dict])[feature_cols]
    scout_scaled = scaler.transform(scout_df)
    scout_pred_ppg = active_model.predict(scout_scaled)[0]
    
    st.markdown("---")
    st.markdown("### 3. Expected Scoring Target Output")
    
    c_out1, c_out2 = st.columns(2)
    with c_out1:
        st.metric("Projected Expected PPG (xPPG)", f"{scout_pred_ppg:.1f} PPG")
    with c_out2:
        projected_total_points = int(scout_pred_ppg * scout_gp)
        st.metric("Projected Total Season Points", f"{projected_total_points:,} PTS ({scout_gp} Games)")

    st.markdown("---")
    render_model_eval_card(selected_model_name, active_metrics)