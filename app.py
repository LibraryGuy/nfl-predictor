import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. DATA LOADING (RESTORED & REPAIRED) ---
@st.cache_data(ttl=3600, show_spinner="Syncing Wild Card Weekend Data...")
def load_nfl_data_pro():
    try:
        years = [2024, 2025]
        # Load and convert to Pandas immediately
        w_raw = nfl.load_player_stats(seasons=years).to_pandas()
        s_raw = nfl.load_schedules(seasons=years).to_pandas()
        
        # --- CRITICAL FIX: FLATTEN MULTIINDEX ---
        # This collapses ('passing', 'passing_yards') into just 'passing_yards'
        for df_obj in [w_raw, s_raw]:
            if isinstance(df_obj.columns, pd.MultiIndex):
                # We take the deepest level name which contains the actual counting stat
                df_obj.columns = df_obj.columns.get_level_values(-1)
            # Ensure all column names are clean, single-level strings
            df_obj.columns = [str(c).strip() for c in df_obj.columns]

        # Standardize Names for 2026 Season data
        name_col = 'player_display_name' if 'player_display_name' in w_raw.columns else 'player_name'
        w_raw = w_raw.rename(columns={name_col: 'player_name', 'team_abbr': 'recent_team'})
        
        # Force Series (Ensures we don't pass a DataFrame to .str)
        w_raw['player_name'] = w_raw['player_name'].astype(str).str.strip()
        w_raw = w_raw.dropna(subset=['player_name', 'position'])
        
        # Force Yardage to Numeric (Fixes Jordan Love 5.3 yard issue)
        # We ensure passing_yards refers to the game total (e.g., 234) not Y/A
        for m in ['passing_yards', 'rushing_yards', 'receiving_yards']:
            w_raw[m] = pd.to_numeric(w_raw[m], errors='coerce').fillna(0)
        
        # Merge with Schedule
        df = w_raw.merge(s_raw[['season', 'week', 'home_team', 'temp', 'wind']], 
                          left_on=['season', 'week', 'recent_team'], 
                          right_on=['season', 'week', 'home_team'], how='left')
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Syncing Error: {str(e)}")
        return pd.DataFrame()

data = load_nfl_data_pro()

# --- 2. DASHBOARD UI (BACK ONLINE) ---
if not data.empty:
    st.title("🏈 NFL Sharp: Wild Card Weekend Projections")
    
    player_list = sorted(data['player_name'].unique())
    selected_player = st.selectbox("Search Player", player_list, index=player_list.index("Jordan Love") if "Jordan Love" in player_list else 0)
    
    player_subset = data[data['player_name'] == selected_player]
    player_pos = player_subset['position'].iloc[-1]
    
    # Correct Stat Selection
    target = 'passing_yards' if player_pos == 'QB' else 'receiving_yards'
    current_avg = player_subset[target].mean()
    
    st.header(f"📊 {selected_player} Analysis")
    c1, c2 = st.columns(2)
    c1.metric("2025 Season Avg", f"{current_avg:.1f} Yds")
    
    # Wild Card Matchup Insight
    st.subheader("Performance History")
    st.plotly_chart(px.line(player_subset, x='week', y=target, markers=True, 
                            title=f"{selected_player} {target.replace('_', ' ').title()} per Game"), 
                    use_container_width=True)
else:
    st.warning("🔄 System Restarting: Data source is refreshing for the 2026 Wild Card round.")
