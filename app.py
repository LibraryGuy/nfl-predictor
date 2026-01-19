import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson, lognorm

# --- 1. THE REGISTRY (2025-2026 UPDATED) ---
@st.cache_data(ttl=3600)
def get_active_player_list():
    try:
        curr_season = nfl.get_current_season()
        # load_players() is now the recommended source over load_rosters()
        players = nfl.load_players().to_pandas()
        
        # In 2025/2026, the key column is 'display_name' or 'short_name'
        # We'll standardize it to 'player_name'
        if 'display_name' in players.columns:
            players = players.rename(columns={'display_name': 'player_name'})
        
        # Filter for active skill positions
        active = players[players['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        return sorted(active['player_name'].dropna().unique().tolist())
    except Exception as e:
        st.error(f"Registry Error: {e}")
        return ["Patrick Mahomes", "Lamar Jackson", "Tyreek Hill"] # Fallbacks

# --- 2. THE STATS LOADER (ON-DEMAND) ---
@st.cache_data(ttl=3600)
def get_player_deep_stats(player_name):
    try:
        curr_season = nfl.get_current_season()
        # seasons=[curr_season] ensures we are looking at the RIGHT now.
        # summary_level='week' is critical for game-by-game distribution
        df = nfl.load_player_stats(seasons=[curr_season], summary_level='week').to_pandas()
        
        # 2025 Column Mapping: nflreadpy now returns 'player_display_name'
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Match the player
        p_df = df[df['player_name'] == player_name].copy()
        
        # If current season is empty (e.g. rookie hasn't played yet), try previous
        if p_df.empty:
            df_prev = nfl.load_player_stats(seasons=[curr_season-1], summary_level='week').to_pandas()
            df_prev = df_prev.rename(columns={k: v for k, v in rename_map.items() if k in df_prev.columns})
            p_df = df_prev[df_prev['player_name'] == player_name].copy()
            
        return p_df
    except Exception as e:
        st.sidebar.warning(f"Stat Fetch Error: {e}")
        return pd.DataFrame()

# --- 3. THE ANALYTICS ENGINE ---
def run_simulation(data_series, market):
    iterations = 10000
    avg = data_series.mean()
    if avg <= 0: return np.zeros(iterations) # Guard for no data
    
    if "TD" in market:
        return np.random.poisson(avg, iterations)
    else:
        # Volatility check: if no variance, use a default 40% coefficient
        std = data_series.std() if data_series.std() > 0 else (avg * 0.4)
        sigma = np.sqrt(np.log(1 + (std**2 / (avg**2 + 1e-9))))
        mu = np.log(avg + 1e-9) - (sigma**2 / 2)
        return np.random.lognormal(mu, sigma, iterations)

# --- 4. UI FLOW ---
st.set_page_config(page_title="2026 NFL Sharp Predictor", layout="wide")
player_list = get_active_player_list()
selected_p = st.sidebar.selectbox("Select Player", player_list)

if selected_p:
    p_data = get_player_deep_stats(selected_p)
    
    if not p_data.empty:
        st.title(f"🚀 {selected_p} Predictor (2025-26 Season)")
        
        # Position Detection
        pos = p_data['position'].iloc[-1] if 'position' in p_data.columns else "WR"
        
        # Market Mapping
        options = {'QB': ['passing_yards', 'passing_tds'], 
                   'RB': ['rushing_yards', 'rushing_tds'],
                   'WR': ['receiving_yards', 'receiving_tds']}
        stat_col = st.sidebar.selectbox("Market", options.get(pos, ['receiving_yards']))
        line = st.sidebar.number_input("Sportsbook Line", value=1.5 if "td" in stat_col else 45.5)

        # Modeling
        sims = run_simulation(p_data[stat_col], stat_col)
        win_p = (np.sum(sims >= line) / 10000) * 100

        # Visuals
        c1, c2 = st.columns([2, 1])
        with c1:
            st.metric("Model Projection", f"{np.mean(sims):.1f} {stat_col.replace('_', ' ')}", f"Win Prob: {win_p:.1f}%")
            fig = go.Figure(go.Histogram(x=sims, marker_color='#00ff96'))
            fig.add_vline(x=line, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("**Last 5 Games Raw Data**")
            st.dataframe(p_data[['week', stat_col]].tail(5), hide_index=True)
    else:
        st.error(f"No 2025 or 2024 game stats found for {selected_p}. They may be inactive or a rookie with no games recorded.")
