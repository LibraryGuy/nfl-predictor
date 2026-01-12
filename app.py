import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import requests
from nfl_stadiums import NFLStadiums

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Sharp: Fixed", layout="wide", page_icon="🏈")

# --- 2. BULLETPROOF DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_nfl_data_stable():
    try:
        # Core Stats
        w_raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        s_raw = nfl.load_schedules(seasons=[2024, 2025]).to_pandas()

        # A. Flatten Headers immediately
        if isinstance(w_raw.columns, pd.MultiIndex):
            w_raw.columns = ["_".join(filter(None, map(str, col))).strip() for col in w_raw.columns.values]

        # B. Map Columns (Flexible Search)
        mapping = {
            'player_display_name': 'player_name',
            'player_name': 'player_name',
            'recent_team': 'team',
            'team': 'team',
            'team_abbr': 'team',
            'opponent_team': 'opponent',
            'opponent': 'opponent'
        }
        
        # Rename based on what exists
        found_map = {col: mapping[col] for col in w_raw.columns if col in mapping}
        w_raw = w_raw.rename(columns=found_map)

        # C. Create Scrimmage Yards (Safe calculation)
        w_raw['total_scrimmage_yards'] = w_raw.get('rushing_yards', 0).fillna(0) + w_raw.get('receiving_yards', 0).fillna(0)

        # D. The "Safe" Weather Merge
        # We wrap this in a try/except so if weather fails, the app still works
        try:
            merged = w_raw.merge(
                s_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
                left_on=['season', 'week', 'team'], 
                right_on=['season', 'week', 'home_team'], 
                how='left'
            )
            # If merge worked, use it. If it returned empty, fall back to w_raw.
            final_df = merged if not merged.empty else w_raw
        except:
            final_df = w_raw

        # E. Defense Engine
        dvp = final_df.groupby(['opponent', 'position'])['total_scrimmage_yards'].mean().reset_index()
        
        return final_df.fillna(0), dvp
        
    except Exception as e:
        st.error(f"Critical Sync Failure: {e}")
        # Return empty dataframes so the rest of the app doesn't TypeError
        return pd.DataFrame(columns=['player_name', 'team', 'position']), pd.DataFrame()

# IMPORTANT: Always unpack both returns
data, dvp_data = load_nfl_data_stable()

# --- 3. UI LOGIC ---
if not data.empty and 'player_name' in data.columns:
    # Safely get player names
    player_list = sorted([p for p in data['player_name'].unique() if p and str(p) != 'nan'])
    
    selected_p = st.selectbox("Select Player", player_list)
    p_df = data[data['player_name'] == selected_p]
    
    # Simple Visual to confirm it works
    st.subheader(f"Performance for {selected_p}")
    st.plotly_chart(px.bar(p_df, x='week', y='total_scrimmage_yards'), use_container_width=True)
    
    # Defense Dropdown
    all_defenses = sorted(data['team'].unique())
    target_def = st.selectbox("Project against Defense", all_defenses)
    
    # Display raw data for peace of mind
    if st.checkbox("Show Raw Data Debugger"):
        st.write(data.head())
else:
    st.error("No data found. Please check your internet connection or nflreadpy version.")
