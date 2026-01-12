import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# --- 1. SETTINGS ---
st.set_page_config(page_title="NFL Genius: Matchup Pro", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_data_safe():
    try:
        # Load Stats
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. Flatten MultiIndex (Critical for 2026 data schema)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        # B. Robust Mapping (Look for variants of 'opponent' and 'team')
        rename_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent'
        }
        # Only rename if the key exists to avoid KeyError
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # C. Mandatory Columns Guard
        # If 'opponent' or 'position' are still missing, find them by fuzzy matching
        if 'opponent' not in df.columns:
            # Often 'defteam' or 'opp'
            opp_col = [c for c in df.columns if 'opp' in c.lower() or 'def' in c.lower()]
            if opp_col: df = df.rename(columns={opp_col[0]: 'opponent'})
            
        # D. Clean duplicates and calculate yards
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df['total_yds'] = df.get('rushing_yards', 0).fillna(0) + df.get('receiving_yards', 0).fillna(0)
        
        # Drop rows where critical data is missing to prevent line 56 crashes
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Data Sync Failure: {e}")
        return pd.DataFrame()

data = load_data_safe()

# --- 2. THE REPAIRED DEFENSE ENGINE ---
def get_safe_defense_metrics(df):
    """Safely calculates DvP without crashing on missing columns."""
    if 'opponent' not in df.columns or 'position' not in df.columns:
        return pd.DataFrame(), {}
    
    # Calculate Benchmarks (League Wide)
    benchmarks = df.groupby('position')['total_yds'].mean().to_dict()
    
    # Calculate Team Defenses
    def_stats = df.groupby(['opponent', 'position'])['total_yds'].mean().reset_index()
    return def_stats, benchmarks

# --- 3. UI LOGIC ---
if not data.empty:
    def_data, league_benchmarks = get_safe_defense_metrics(data)
    
    st.title("🏈 NFL Genius: Matchup Predictor")
    
    # Sidebar Search
    st.sidebar.header("Selection")
    players = sorted(data['player_name'].unique())
    selected_p = st.sidebar.selectbox("Choose Player", players)
    
    opponents = sorted(data['opponent'].unique())
    selected_opp = st.sidebar.selectbox("Choose Opponent Defense", opponents)

    # Filtering
    p_df = data[data['player_name'] == selected_p]
    p_pos = p_df['position'].iloc[-1]
    p_avg = p_df['total_yds'].mean()
    
    # --- 4. CALCULATE MATCHUP MULTIPLIER ---
    # Find how this opponent handles this specific position
    opp_match = def_data[(def_data['opponent'] == selected_opp) & (def_data['position'] == p_pos)]
    
    bench = league_benchmarks.get(p_pos, 1.0)
    matchup_mod = (opp_match['total_yds'].iloc[0] / bench) if not opp_match.empty else 1.0
    
    # Final Projection
    proj_yds = p_avg * matchup_mod

    # --- 5. DISPLAY ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Player Avg", f"{p_avg:.1f} Yds")
    c2.metric("Defense Mod", f"{matchup_mod:.2f}x", 
              delta="Easy Matchup" if matchup_mod > 1.05 else "Tough Matchup" if matchup_mod < 0.95 else "Neutral")
    c3.metric("Projected", f"{proj_yds:.1f} Yds")

    st.plotly_chart(px.line(p_df, x='week', y='total_yds', title=f"{selected_p} Performance Trend"), use_container_width=True)
else:
    st.error("Data could not be processed. Please check your nflreadpy connection.")
