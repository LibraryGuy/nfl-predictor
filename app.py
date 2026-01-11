import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# ... [CONFIG & AUTH SECTION REMAINS THE SAME] ...

# --- 4. DATA LOADING (REPAIRED & FLATTENED) ---
@st.cache_data(ttl=3600, show_spinner="Fetching Latest NFL Stats...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and convert to Pandas
        weekly_raw = nfl.load_player_stats(seasons=years).to_pandas()
        sched_raw = nfl.load_schedules(seasons=years).to_pandas()
        pbp_raw = nfl.load_pbp(seasons=years).to_pandas() 
        
        # --- THE CRITICAL FIX: FLATTEN MULTIINDEX ---
        # collapsed ('passing', 'passing_yards') -> 'passing_yards'
        for df_obj in [weekly_raw, sched_raw, pbp_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # We take the lowest level name which contains the actual stat
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean, single-level strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names & Teams
        name_col = 'player_display_name' if 'player_display_name' in weekly_raw.columns else 'player_name'
        weekly = weekly_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Force Series for string operations (Stops the AttributeError)
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        weekly = weekly.dropna(subset=['player_name', 'position'])
        
        # Yardage cleanup - Ensure we have TOTAL yards
        for m in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
        
        # Defense EPA
        def_epa = pbp_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        
        # Final Merge
        df = weekly.merge(sched_raw[['season', 'week', 'home_team', 'temp', 'surface', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')
        
        df[['wind', 'temp', 'def_epa_allowed']] = df[['wind', 'temp', 'def_epa_allowed']].fillna(0)
        df['is_grass'] = df['surface'].str.lower().str.contains('grass', na=False).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# ... [REST OF DASHBOARD LOGIC REMAINS THE SAME] ...
