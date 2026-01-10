import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

# 1. PAGE SETUP
st.set_page_config(page_title="NFL Sharp Pro Predictor", layout="wide")
st.title("🏈 NFL Sharp: Analytics & Parlay Builder")

@st.cache_data(show_spinner="Updating NFL Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas() # Needed for Defense EPA
        
        # Standardize Team Columns
        if 'recent_team' not in weekly.columns:
            team_col = 'team' if 'team' in weekly.columns else 'team_abbr'
            weekly = weekly.rename(columns={team_col: 'recent_team'})
        
        # Clean & Numeric
        weekly = weekly.dropna(subset=['player_name', 'position'])
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'targets']
        for m in metrics: weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        weekly['total_scrimmage_tds'] = weekly['rushing_tds'] + weekly['receiving_tds']
        
        # Defense Stats (EPA Allowed)
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Rolling Averages
        weekly = weekly.sort_values(['player_name', 'season', 'week'])
        weekly['yards_roll3'] = weekly.groupby('player_name')['total_scrimmage_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['pass_roll3'] = weekly.groupby('player_name')['passing_yards'].transform(lambda x: x.rolling(3, 1).mean())
        weekly['td_roll3'] = weekly.groupby('player_name')['total_scrimmage_tds'].transform(lambda x: x.rolling(3, 1).mean())

        # Merge Environment & Defense
        sched_cols = ['season', 'week', 'home_team', 'temp', 'surface', 'wind']
        df = weekly.merge(sched[sched_cols], left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')
        
        df['wind'] = df['wind'].fillna(0)
        df['temp'] = df['temp'].fillna(70)
        df['def_epa_allowed'] = df['def_epa_allowed'].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception:
        return pd.DataFrame()

# --- INITIALIZE ---
data = load_nfl_data_pro()

if data.empty:
    st.error("Data failed to load.")
    st.stop()

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("Game Environment")
curr_wind = st.sidebar.slider("Wind Speed (MPH)", 0, 40, 5)
curr_temp = st.sidebar.slider("Temperature (F)", 0, 100, 65)
is_grass_val = 1 if st.sidebar.radio("Field Type", ["Grass", "Turf"]) == "Grass" else 0

# --- PLAYER & OPPONENT SELECTION ---
player_list = sorted(data['player_name'].unique())
selected_player = st.selectbox("Select Player for Detailed View", player_list)
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Opponent (Defense)", opp_list)

player_subset = data[data['player_name'] == selected_player]
player_pos = player_subset['position'].iloc[-1]
vegas_line = st.sidebar.number_input("Enter Sportsbook Line", value=225.5 if player_pos == 'QB' else 65.5)

# --- PREDICTION ENGINE (Stable Version) ---
def get_stable_prediction(df, player_name, target_stat, temp, wind, is_grass, opp_team):
    pos = df[df['player_name'] == player_name]['position'].iloc[-1]
    pos_data = df[df['position'] == pos].copy()
    
    # Feature setup
    feature_cols = ['temp', 'wind', 'is_grass', 'def_epa_allowed']
    roll_col = 'pass_roll3' if pos == 'QB' else 'yards_roll3'
    
    X = pos_data[feature_cols + [roll_col]].fillna(0)
    y = pos_data[target_stat]
    
    # Train Stable Model
    model = XGBRegressor(n_estimators=45, max_depth=3, reg_lambda=10).fit(X, y)
    
    # Get current opponent's average defensive EPA
    opp_epa = df[df['opponent_team'] == opp_team]['def_epa_allowed'].mean()
    p_latest = df[df['player_name'] == player_name].iloc[-1]
    
    input_df = pd.DataFrame([[temp, wind, is_grass, opp_epa, p_latest[roll_col]]], 
                             columns=feature_cols + [roll_col])
    
    pred = model.predict(input_df)[0]
    # Safety floor (60% of rolling avg)
    return max(pred, p_latest[roll_col] * 0.6)

# --- DASHBOARD LAYOUT ---
st.divider()
target_stat = 'passing_yards' if player_pos == 'QB' else 'total_scrimmage_yards'
proj = get_stable_prediction(data, selected_player, target_stat, curr_temp, curr_wind, is_grass_val, selected_opp)

col1, col2, col3 = st.columns(3)
with col1:
    st.success(f"🎯 RECOMMENDED: {int(proj*0.85/5)*5}+ Yards")
    st.metric("Model Projection", f"{proj:.1f}")

with col2:
    med = player_subset[target_stat].median()
    if proj > med * 1.5 or (curr_wind > 18 and player_pos != 'RB'):
        st.error("⚠️ FADE ALERT: Avoid 'Over'")
    else:
        st.warning("⚖️ NEUTRAL: No strong Fade signals.")

with col3:
    edge = proj - vegas_line
    st.metric("Vegas Line Edge", f"{(edge/vegas_line)*100:.1f}%", delta=f"{edge:.1f} yds")

# Visuals
st.divider()
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(px.line(player_subset, x='week', y=[target_stat, 'yards_roll3'], title="Yardage Velocity"), use_container_width=True)
with g2:
    st.plotly_chart(px.scatter(player_subset, x='total_scrimmage_yards', y='total_scrimmage_tds', trendline="ols", title="TD Efficiency"), use_container_width=True)

# --- PARLAY BUILDER (TD Probs) ---
st.divider()
st.header("🎟️ Parlay Builder")
parlay_players = st.multiselect("Add Scorers to Ticket", player_list, default=[selected_player])

if parlay_players:
    probs = []
    for p in parlay_players:
        p_pos = data[data['player_name'] == p]['position'].iloc[-1]
        stat = 'passing_tds' if p_pos == 'QB' else 'total_scrimmage_tds'
        exp = get_stable_prediction(data, p, stat, curr_temp, curr_wind, is_grass_val, selected_opp)
        hit_prob = (1 - poisson.pmf(0, exp))
        probs.append(hit_prob)
    
    total_prob = np.prod(probs) * 100
    st.metric("Parlay Win Probability", f"{total_prob:.2f}%")
    st.progress(min(total_prob/100, 1.0))

# --- HISTORICAL MATCHUP TABLE ---
st.divider()
st.subheader(f"🏟️ {selected_player} Career vs {selected_opp}")

# 1. Filter data for the specific matchup
matchup_history = player_subset[player_subset['opponent_team'] == selected_opp].copy()

# 2. Safety Check: Only show if they have actually played each other
if not matchup_history.empty:
    # Select and rename columns for a clean look
    display_cols = ['season', 'week', target_stat, 'total_scrimmage_tds', 'temp', 'wind']
    history_table = matchup_history[display_cols].sort_values('season', ascending=False)
    
    # Format the table for readability
    st.dataframe(
        history_table,
        column_config={
            "season": "Year",
            "week": "Wk",
            target_stat: "Yards",
            "total_scrimmage_tds": "TDs",
            "temp": "Temp",
            "wind": "Wind"
        },
        hide_index=True,
        use_container_width=True
    )
else:
    # 3. Graceful Fallback (Prevents NoneType Error)
    st.info(f"No previous career games found for {selected_player} against the {selected_opp}.")
