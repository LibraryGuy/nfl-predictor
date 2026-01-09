import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="NFL Sharp: Ultimate Prediction Engine", layout="wide")

# --- DEBUG & RESET TOOLS ---
if st.sidebar.button("Clear Cache & Reload"):
    st.cache_data.clear()
    st.rerun()

st.title("🏈 NFL Sharp: Ultimate Prediction Engine")

@st.cache_data(show_spinner="Initializing NFL Database...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        
        # 1. Fetch Data
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        
        # 2. Safety Check: If any dataframe is empty, stop early
        if weekly.empty or pbp.empty:
            return pd.DataFrame()

        # 3. Robust ID Alignment (The previous KeyError fix)
        # Using .get() ensures we don't crash if a column is missing
        pbp['player_id'] = pbp['receiver_player_id'].fillna(
            pbp.get('rusher_player_id', np.nan)).fillna(pbp.get('passer_player_id', np.nan))
        
        # 4. Feature Engineering: Red Zone & Defense
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        rz_touches = pbp[pbp['yardline_100'] <= 20].groupby(['season', 'week', 'player_id']).size().reset_index(name='rz_touches')
        
        # 5. Merge Strategy
        df = weekly.merge(rz_touches, on=['season', 'week', 'player_id'], how='left').fillna(0)
        
        # 6. Team & Metric Standardization
        team_col = 'recent_team' if 'recent_team' in df.columns else 'team'
        df = df.rename(columns={team_col: 'recent_team'})
        
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for m in metrics: df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)
        
        df['total_scrimmage_yards'] = df['rushing_yards'] + df['receiving_yards']
        df['total_scrimmage_tds'] = df['rushing_tds'] + df['receiving_tds']
        
        # 7. Rolling Calculations
        df = df.sort_values(['player_name', 'season', 'week'])
        for col in ['total_scrimmage_yards', 'rz_touches']:
            df[f'{col}_roll3'] = df.groupby('player_name')[col].transform(lambda x: x.rolling(3, 1).mean())

        # 8. Environmental Merge
        df = df.merge(sched[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                      left_on=['season', 'week', 'recent_team'], 
                      right_on=['season', 'week', 'home_team'], how='left')
        
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')

        # Final Fill & Cleanup
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df

    except Exception as e:
        # If any internal error happens, we return an empty DF so line 84 doesn't crash
        return pd.DataFrame()

# Execution
data = load_nfl_data_pro()

# --- THE FIX FOR LINE 84 ---
# We check if data is valid before doing ANY operations
if isinstance(data, pd.DataFrame) and not data.empty:
    
    player_list = sorted(data['player_name'].dropna().unique())
    selected_player = st.selectbox("Select Player", player_list)
    
    # Sidebar Filters
    st.sidebar.header("Matchup Conditions")
    curr_wind = st.sidebar.slider("Expected Wind", 0, 40, 5)
    curr_temp = st.sidebar.slider("Expected Temp", 0, 100, 60)
    
    # Core Logic
    p_info = data[data['player_name'] == selected_player].iloc[-1]
    pos = p_info['position']
    
    # Simple Prediction Engine
    def predict_stat(target):
        p_data = data[data['player_name'] == selected_player]
        features = ['temp', 'wind', 'is_grass', 'rz_touches_roll3', 'def_epa_allowed']
        model = XGBRegressor(n_estimators=30).fit(p_data[features], p_data[target])
        
        # Latest situational input
        latest_feat = p_data[features].iloc[[-1]].copy()
        latest_feat['temp'], latest_feat['wind'] = curr_temp, curr_wind
        return model.predict(latest_feat)[0]

    # Display Results
    st.subheader(f"Predictions for {selected_player} ({pos})")
    target_col = 'passing_yards' if pos == 'QB' else 'total_scrimmage_yards'
    proj = predict_stat(target_col)
    
    c1, c2 = st.columns(2)
    c1.metric("Projected Output", f"{proj:.1f} Yds")
    c2.metric("Red Zone Usage (Avg)", f"{p_info['rz_touches_roll3']:.1f}")

    # Visuals
    st.plotly_chart(px.line(data[data['player_name'] == selected_player], x='week', y=target_col, title="Performance History"))

else:
    st.error("⚠️ NFL Data could not be loaded.")
    st.info("Try clicking 'Clear Cache & Reload' in the sidebar. This can happen if the NFL API is temporarily rate-limiting your app.")
    st.stop()
