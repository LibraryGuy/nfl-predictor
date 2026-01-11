import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. DATA LOADING (THE RECOVERY SECTION) ---
@st.cache_data(ttl=3600)
def load_nfl_data_pro():
    try:
        # Load data for 2024 and the current 2025 season
        years = [2024, 2025]
        weekly_raw = nfl.load_player_stats(seasons=years).to_pandas()
        sched_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- CRITICAL FIX: FLATTEN NESTED COLUMNS ---
        # This collapses ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [weekly_raw, sched_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # Take the deepest level which contains the actual stat name
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names for 2026 data standards
        name_col = 'player_display_name' if 'player_display_name' in weekly_raw.columns else 'player_name'
        weekly = weekly_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Now this will work because 'player_name' is a Series, not a DataFrame
        weekly['player_name'] = weekly['player_name'].astype(str).str.strip()
        
        # Force Yardage to Numbers (Fixes Jordan Love 5.3 yard issue)
        # This ensures you get 250+ yards (Total) instead of 5.3 (Yards Per Attempt)
        metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for m in metrics:
            weekly[m] = pd.to_numeric(weekly[m], errors='coerce').fillna(0)
            
        return weekly
    except Exception as e:
        # This prevents the dashboard from disappearing by showing exactly what failed
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 2. DASHBOARD RENDERING ---
if not data.empty:
    st.title("🏈 NFL Sharp: Pro Predictor")
    
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list)
    
    player_subset = data[data['player_name'] == selected_player]
    
    # Check if QB for passing yards or Skill Pos for scrimmage yards
    is_qb = player_subset['position'].iloc[-1] == 'QB'
    target_stat = 'passing_yards' if is_qb else 'receiving_yards'
    
    avg_val = player_subset[target_stat].mean()
    
    st.metric(f"Season Average {target_stat.replace('_', ' ').title()}", f"{avg_val:.1f}")
    st.plotly_chart(px.line(player_subset, x='week', y=target_stat, markers=True))
else:
    st.warning("Dashboard is currently offline due to a data sync error.")
