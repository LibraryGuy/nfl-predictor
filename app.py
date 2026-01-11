import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- DATA LOADING (THE REPAIRED SECTION) ---
@st.cache_data(ttl=3600, show_spinner="Syncing NFL Pro Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and convert to Pandas
        weekly_raw = nfl.load_player_stats(seasons=years).to_pandas()
        sched_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- CRITICAL FIX: FLATTEN MULTIINDEX ---
        # This collapses ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [weekly_raw, sched_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # We take the deepest level (-1) which holds the actual metric name
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean, flat strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names for the 2026 data standard
        name_col = 'player_display_name' if 'player_display_name' in weekly_raw.columns else 'player_name'
        weekly = weekly_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Clean Player Names (Now guaranteed to be a Series)
        # This specific line is where your error was occurring
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # Force Yardage to Numeric (This fixes the 5.3 yard error)
        # It ensures we are pulling the total (e.g., 280.0) not an average (e.g., 5.3)
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for m in metrics:
            if m in weekly.columns:
                weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
        
        # Merge with Schedule for Weather/Home/Away context
        df = weekly.merge(sched_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- DASHBOARD LOGIC ---
if not data.empty:
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    
    # Logic to handle different positions
    player_pos = player_subset['position'].iloc[-1]
    target_metric = 'passing_yards' if player_pos == 'QB' else 'receiving_yards'
    
    # Verification: If J. Love shows 200+ yards, the flatten worked!
    avg_yards = player_subset[target_metric].mean()
    
    st.header(f"📊 {selected_player} Analysis")
    st.metric(f"Season Average {target_metric.replace('_', ' ').title()}", f"{avg_yards:.1f} Yds")
    
    st.plotly_chart(px.line(player_subset, x='week', y=target_metric, markers=True), use_container_width=True)
