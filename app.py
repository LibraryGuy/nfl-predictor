import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, time
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS ---
st.set_page_config(page_title="NFL Sharp: Intel", layout="wide")

# --- 2. THE FIX: ROBUST DATA LOADING ---
@st.cache_data(ttl=3600)
def load_and_standardize_nfl_data():
    try:
        # nflreadpy returns Polars; we MUST convert to Pandas immediately
        raw_polars = nfl.load_player_stats(seasons=[2024, 2025])
        df = raw_polars.to_pandas()
        
        if df.empty:
            return pd.DataFrame()

        # LOGIC: Aggressively look for name columns
        # nflverse often uses 'player_display_name' or 'player_name'
        possible_name_cols = ['player_display_name', 'player_name', 'player', 'name']
        found_name = next((c for c in possible_name_cols if c in df.columns), None)
        
        if found_name:
            df = df.rename(columns={found_name: 'player_name'})
        else:
            # If no name column is found, the data is likely corrupted or different schema
            return df 

        # Standardize other keys
        df = df.rename(columns={
            'recent_team': 'team', 
            'opponent_team': 'opponent',
            'position_group': 'position'
        })
        
        # Numeric Safety
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0
                
        return df
    except Exception as e:
        st.error(f"Critical Data Error: {e}")
        return pd.DataFrame()

# Initialize
data = load_and_standardize_nfl_data()
stadiums = NFLStadiums()

# --- 3. UI GUARD ---
# This prevents the sorted() crash by checking if the column actually exists
if not data.empty and 'player_name' in data.columns:
    with st.sidebar:
        st.title("🏈 Sharp Dashboard")
        
        # Safe access to unique names
        player_list = sorted(data['player_name'].dropna().unique())
        selected_p = st.selectbox("Select Player", player_list)
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if 'position' in p_df.columns else "WR"
        
        # Opponent Logic
        opp_col = 'opponent' if 'opponent' in data.columns else 'team'
        opponents = sorted(data[opp_col].unique())
        selected_opp = st.selectbox("Versus Defense", opponents)
        
        market = st.radio("Market", ["Yards", "Touchdowns"])
        is_td = market == "Touchdowns"
        line = st.number_input("Line", value=0.5 if is_td else 50.0)
        
        venue_name = st.selectbox("Venue", sorted(stadiums.get_list_of_stadium_names()))
        k_time = st.time_input("Kickoff", time(13, 0))

    # --- 4. CALCULATION ENGINE ---
    def run_sim(avg_v, avg_e, mult, td_mode):
        if td_mode:
            # Poisson for TDs (Drake Maye Fix)
            lam = max(0.01, avg_e * mult)
            return np.random.poisson(lam, 10000)
        else:
            # Lognormal for Yards
            vol = np.random.poisson(max(avg_v, 1), 10000)
            mu = np.log(max(avg_e * mult, 0.01)) - (0.4**2 / 2)
            eff = np.random.lognormal(mu, 0.4, 10000)
            return vol * eff

    # Stat prep
    if not is_td:
        stat = 'passing_yards' if p_pos == 'QB' else ('rushing_yards' if p_pos == 'RB' else 'receiving_yards')
        vol = 'attempts' if p_pos == 'QB' else ('carries' if p_pos == 'RB' else 'targets')
        v_val, e_val = p_df[vol].mean(), (p_df[stat] / p_df[vol].replace(0, np.nan)).mean()
    else:
        stat = 'passing_tds' if p_pos == 'QB' else ('rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        v_val, e_val = 1.0, p_df[stat].mean()

    # Weather/Matchup (Simulated for brevity)
    m_ratio = 1.05 if "Shutdown" not in selected_opp else 0.90
    sims = run_sim(v_val, e_val, m_ratio, is_td)
    win_p = (np.sum(sims >= line) / 10000) * 100

    # UI Rendering
    st.header(f"📊 {selected_p} Projections")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Mean", f"{round(np.mean(sims), 2)} {market}", f"Win Prob: {round(win_p, 1)}%")
        fig = go.Figure(go.Histogram(x=sims, marker_color='#22c55e'))
        fig.add_vline(x=line, line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Probability Ladder")
        ladder = []
        for t in ([0.5, 1.5, 2.5] if is_td else [25, 50, 75, 100]):
            p = (np.sum(sims >= t) / 10000) * 100
            ladder.append({"Target": f"{t}+", "Prob": f"{p:.1f}%"})
        st.table(pd.DataFrame(ladder))

else:
    st.error("### 🛑 Data Structure Mismatch")
    st.write("The player name column was not found. Here is what the data looks like:")
    st.write(data.head()) # This helps you debug which columns the API is actually sending
    st.write("Available Columns:", data.columns.tolist() if not data.empty else "No columns found")
