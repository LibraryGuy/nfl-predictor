import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Genius Pro", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA ENGINE (The "Shielded" Fix) ---
@st.cache_data(ttl=3600)
def load_and_fix_data():
    try:
        # Load data (nflreadpy handles 2024-2025)
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. FLATTEN HEADERS: Fixes MultiIndex columns that cause AttributeError
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        # B. RENAME VARIATIONS: Ensure 'player_name' always exists
        rename_map = {
            'player_display_name': 'player_name',
            'opponent_team': 'opponent',
            'recent_team': 'team',
            'rushing_tds': 'rush_td',
            'receiving_tds': 'rec_td'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # C. DEFENSE ENGINE: Yards allowed by opponent per position
        def_df = df.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return df.fillna(0), def_df
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Execute Data Load
data, def_data = load_and_fix_data()

# --- 3. THE ANALYTICS ---
def calc_td_prob(avg_td, modifier):
    lam = avg_td * modifier
    return (1 - exp(-lam)) * 100

# --- 4. THE INTERFACE ---
st.title("🏈 NFL Genius: Pro Builder")

# SIDEBAR: Always Rendered
with st.sidebar:
    st.header("🛒 Parlay Builder")
    if st.session_state.parlay_legs:
        for leg in st.session_state.parlay_legs:
            st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
        if st.button("Reset Parlay"):
            st.session_state.parlay_legs = []
            st.rerun()
    else:
        st.info("Add legs to build a parlay.")

# MAIN PAGE: Final check for 'player_name' column
if not data.empty and 'player_name' in data.columns:
    col1, col2 = st.columns([2, 1])

    with col1:
        # PLAYER SELECT: Safe column access
        players = sorted(data['player_name'].unique())
        p_name = st.selectbox("Select Player", players)
        p_data = data[data['player_name'] == p_name]
        p_pos = p_data['position'].iloc[-1]
        
        # DEFENSE SELECT
        opp_list = sorted(data['opponent'].unique())
        target_opp = st.selectbox("Opponent Defense", opp_list)
        
        # Matchup Logic
        d_match = def_data[(def_data['def_team'] == target_opp) & (def_data['position'] == p_pos)]
        mod = 1.15 if (not d_match.empty and d_match['receiving_yards'].iloc[0] > 55) else 0.90

    with col2:
        st.subheader("🎯 Projections")
        line = st.number_input("Market Line", value=50.5)
        
        # Stats
        avg_yds = p_data['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_data['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - line) / line) * 100
        
        # TD Prob
        avg_td = (p_data['rush_td'].mean() + p_data['rec_td'].mean())
        td_p = calc_td_prob(avg_td, mod)
        
        st.metric("Model Proj", f"{proj_yds:.1f} Yds", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD %", f"{td_p:.1f}%")

    st.divider()
    if st.button("➕ Add to Parlay"):
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{'Over' if edge > 0 else 'Under'} {line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Added {p_name}!")

    st.plotly_chart(px.line(p_data, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title=f"Season Performance: {p_name}", markers=True))
else:
    st.warning("⚠️ Data Syncing... Please refresh the page if the selector doesn't appear.")
