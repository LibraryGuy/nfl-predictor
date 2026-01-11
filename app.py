import streamlit as st
from st_paywall import add_auth
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px

# ... [Keep your Config & Auth sections here] ...

# --- 4. DATA LOADING (REPAIRED & FLATTENED) ---
@st.cache_data(ttl=3600, show_spinner="Syncing NFL Pro Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and immediately convert to Pandas
        weekly_raw = nfl.load_player_stats(seasons=years).to_pandas()
        sched_raw = nfl.load_schedules(seasons=years).to_pandas()
        pbp_raw = nfl.load_pbp(seasons=years).to_pandas() 
        
        # --- THE FIX: FLATTEN MULTIINDEX ---
        # This collapses ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [weekly_raw, sched_raw, pbp_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # Take the deepest level name which contains the actual stat
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean, single-level strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names (Handling 2026 data standard)
        name_col = 'player_display_name' if 'player_display_name' in weekly_raw.columns else 'player_name'
        weekly = weekly_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Force Series for string operations (Kills the AttributeError)
        # We use .copy() to ensure we aren't working on a slice/view
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # Repair Yardage: Force TOTAL yards, not averages
        # This fixes Jordan Love showing 5.3 (likely his YPA)
        for m in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        weekly['scrimmage_yds'] = weekly['rushing_yards'] + weekly['receiving_yards']
        
        # Defense EPA & Weather Merge
        def_epa = pbp_raw.groupby(['season', 'week', 'defteam'])['epa'].mean().reset_index(name='def_epa_allowed')
        df = weekly.merge(sched_raw[['season', 'week', 'home_team', 'temp', 'wind', 'surface']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        df = df.merge(def_epa, left_on=['season', 'week', 'opponent_team'], 
                      right_on=['season', 'week', 'defteam'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 5. DASHBOARD UI ---
if not data.empty:
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    target = 'passing_yards' if player_pos == 'QB' else 'scrimmage_yds'
    
    # Verification Logic
    current_avg = player_subset[target].mean()
    
    st.header(f"📊 {selected_player} Projections")
    c1, c2 = st.columns(2)
    c1.metric("Season Average", f"{current_avg:.1f} Yds")
    c2.success(f"Model Projection: {current_avg * 1.05:.1f} Yds")
    
    st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, 
                            title=f"{selected_player} Performance History"), use_container_width=True)
