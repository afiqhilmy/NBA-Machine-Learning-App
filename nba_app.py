import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NBA Expected Scoring (xPPG) Simulator",
    page_icon="🏀",
    layout="wide"
)

# --- LOAD ARTIFACTS & DATA ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('xppg_model.joblib')
    scaler = joblib.load('xppg_scaler.joblib')
    features = joblib.load('xppg_features.joblib')
    return model, scaler, features

@st.cache_data
def load_data():
    df = pd.read_csv('nba_player_seasons_processed.csv')
    return df

try:
    model, scaler, feature_cols = load_artifacts()
    df = load_data()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False

st.title("🏀 NBA Expected Scoring (xPPG) & Workload Simulator")
st.caption("Predict player scoring baselines and evaluate Points Over Expected (xPTS) based on workload and efficiency.")

if not artifacts_loaded:
    st.error("Model artifacts not found. Please run the training and export script first.")
    st.stop()

# --- SIDEBAR: MODE & INPUT CONTROLS ---
st.sidebar.header("🎯 Simulation Controls")
app_mode = st.sidebar.radio("Select Workflow Mode:", ["Player Lookup & Scenario", "Custom Workload Builder"])

# Default slider values
defaults = {
    'MPG': 25.0, 'FGA_pg': 10.0, '3PA_pg': 4.0, 'FTA_pg': 3.0,
    'FG%': 0.450, '3P%': 0.350, 'FT%': 0.780,
    'APG': 3.0, 'RPG': 4.0, 'TOPG': 1.8, 'PTS_Std': 4.5
}

selected_player_name = None
actual_ppg = None

if app_mode == "Player Lookup & Scenario":
    st.sidebar.markdown("---")
    # Player Selection
    players_list = sorted(df['player'].unique())
    selected_player_name = st.sidebar.selectbox("Select Player:", players_list)
    
    player_seasons = df[df['player'] == selected_player_name]['season'].unique()
    selected_season = st.sidebar.selectbox("Select Season:", sorted(player_seasons, reverse=True))
    
    # Extract selected player's actual row
    player_row = df[(df['player'] == selected_player_name) & (df['season'] == selected_season)].iloc[0]
    actual_ppg = player_row['PPG']
    
    st.sidebar.subheader("Adjust Workload Sliders")
    for feat in feature_cols:
        if feat in player_row:
            defaults[feat] = float(player_row[feat])

# Interactive Sliders
st.sidebar.markdown("### Workload & Efficiency Inputs")
input_data = {}

input_data['MPG'] = st.sidebar.slider("Minutes Per Game (MPG)", 5.0, 42.0, float(defaults['MPG']), 0.5)
input_data['FGA_pg'] = st.sidebar.slider("Field Goal Attempts (FGA/G)", 1.0, 30.0, float(defaults['FGA_pg']), 0.5)
input_data['3PA_pg'] = st.sidebar.slider("3-Point Attempts (3PA/G)", 0.0, 15.0, float(defaults['3PA_pg']), 0.5)
input_data['FTA_pg'] = st.sidebar.slider("Free Throw Attempts (FTA/G)", 0.0, 15.0, float(defaults['FTA_pg']), 0.5)

st.sidebar.markdown("### Shooting Efficiency")
input_data['FG%'] = st.sidebar.slider("Field Goal %", 0.300, 0.650, float(defaults['FG%']), 0.005, format="%.3f")
input_data['3P%'] = st.sidebar.slider("3-Point %", 0.200, 0.500, float(defaults['3P%']), 0.005, format="%.3f")
input_data['FT%'] = st.sidebar.slider("Free Throw %", 0.500, 0.950, float(defaults['FT%']), 0.005, format="%.3f")

st.sidebar.markdown("### Secondary Stats")
input_data['APG'] = st.sidebar.slider("Assists Per Game (APG)", 0.0, 12.0, float(defaults['APG']), 0.2)
input_data['RPG'] = st.sidebar.slider("Rebounds Per Game (RPG)", 0.0, 15.0, float(defaults['RPG']), 0.2)
input_data['TOPG'] = st.sidebar.slider("Turnovers Per Game (TOPG)", 0.0, 6.0, float(defaults['TOPG']), 0.1)

if 'PTS_Std' in feature_cols:
    input_data['PTS_Std'] = defaults.get('PTS_Std', 4.5)

# --- MODEL PREDICTION ---
# Build input DataFrame then reindex to match model feature order,
# filling any missing features with zeros (or sensible defaults).
try:
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols, fill_value=0)
except Exception as e:
    st.error("Error building model input DataFrame — columns mismatch.")
    st.write("feature_cols expected by model:", feature_cols)
    st.write("input keys provided:", list(input_data.keys()))
    st.stop()

try:
    input_scaled = scaler.transform(input_df)
    predicted_ppg = model.predict(input_scaled)[0]
except Exception as e:
    st.error("Error during model scaling/prediction:")
    st.write(str(e))
    st.write("Input DataFrame:")
    st.dataframe(input_df)
    st.stop()

# --- MAIN DASHBOARD DISPLAY ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Expected PPG (xPPG)", f"{predicted_ppg:.1f} PPG")

with col2:
    if actual_ppg is not None:
        st.metric("Actual PPG", f"{actual_ppg:.1f} PPG")
    else:
        st.metric("Simulated Workload", f"{input_data['FGA_pg']:.1f} FGA | {input_data['MPG']:.1f} MIN")

with col3:
    if actual_ppg is not None:
        xpts = actual_ppg - predicted_ppg
        delta_color = "normal" if abs(xpts) <= 1.0 else ("inverse" if xpts < 0 else "normal")
        st.metric("Points Over Expected (xPTS)", f"{xpts:+.1f} PPG", delta_color=delta_color)
    else:
        st.metric("Expected Scoring Range", f"{max(0, predicted_ppg - 1.5):.1f} - {predicted_ppg + 1.5:.1f}")

st.markdown("---")

# --- PLOTLY VISUALIZATION ---
st.subheader("📊 Workload vs. Scoring Visualization")

if actual_ppg is not None:
    fig = go.Figure(data=[
        go.Bar(name='Actual PPG', x=[selected_player_name], y=[actual_ppg], marker_color='#1d4ed8'),
        go.Bar(name='Expected PPG Baseline', x=[selected_player_name], y=[predicted_ppg], marker_color='#059669')
    ])
    fig.update_layout(
        barmode='group', 
        title=f"Actual vs. Expected PPG Baseline ({selected_player_name})",
        yaxis_title="Points Per Game",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Adjust the sidebar sliders to see real-time updates to the expected scoring baseline.")