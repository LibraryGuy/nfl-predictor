import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# ... (Config & Auth stay the same) ...

# --- 4. DATA LOADING (REPAIRED) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and immediately convert
        weekly = nfl.load_player_stats(seasons=years).to_pandas()
        sched = nfl.load_schedules(seasons=years).to_pandas()
        pbp = nfl.load_pbp(seasons=years).to_pandas() 

        # --- THE FIX: FLATTEN MULTIINDEX ---
        # If columns are nested, we take the lowest level (the actual stat name)
        for df_obj in [weekly, sched, pbp]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Explicitly rename to avoid 2026 data mismatches
        name_col = 'player_display_name' if 'player_display_name' in weekly.columns else 'player_name'
        weekly = weekly.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})

        # Clean Player Names (Now guaranteed to be a Series)
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        weekly = weekly.dropna(subset=['player_name', 'position'])

        # Merge & Weather
        def_epa = pbp.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = weekly.merge(sched[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')

        # FINAL YARDAGE REPAIR: Force total yards
        for m in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)
        
        df['total_scrimmage_yards'] = df['rushing_yards'] + df['receiving_yards']
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# ... (Rest of dashboard follows) ...
