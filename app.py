import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.express as px
from math import exp

# --- 1. CONFIG & SESSION ---
st.set_page_config(page_title="NFL Genius: Pro Suite", layout="wide", page_icon="🏈")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

# --- 2. DATA ENGINE (The Fix is Here) ---
@st.cache_data(ttl=3600)
def load_and_secure_data():
    try:
        # Load 2024-2025 data
        raw = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. FLATTEN MULTI-INDEX HEADERS
        # This prevents the AttributeError by turning nested columns into simple strings
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ['_'.join(filter(None, map(str, col))).strip() for col in raw.columns.values]
        
        # B. FUZZY MAPPING (Standardizing for 2026 schema)
        mapping = {
            'player_name': ['player_display_name', 'player_name', 'player', 'displayName'],
            'team': ['recent_team', 'team_abbr', 'team', 'recent_team_abbr'],
            'opponent': ['opponent_team', 'opponent', 'opp'],
            'rush_td': ['rushing_tds', 'rush_td'],
            'rec_td': ['receiving_tds', 'rec_td']
        }
        
        for target, options in mapping.items():
            found = next((opt for opt in options if opt in raw.columns), None)
            if found:
                raw = raw.rename(columns={found: target})
            elif target not in raw.columns:
                # Shield: Create missing columns with defaults so UI doesn't crash
                raw[target] = "Unknown" if 'name' in target or 'team' in target else 0

        # C. DEFENSE ENGINE (Aggregating yards allowed per team/position)
        def_stats = raw.groupby(['opponent', 'position']).agg({
            'passing_yards': 'mean',
            'rushing_yards': 'mean',
            'receiving_yards': 'mean'
        }).reset_index().rename(columns={'opponent': 'def_team'})
        
        return raw.fillna(0), def_stats
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, def_data = load_and_secure_data()

# --- 3. THE ANALYTICS ---
def get_td_prob(avg_td, match_mod):
    """Calculates Anytime TD Probability using Poisson distribution"""
    lam = avg_td * match_mod
    return (1 - exp(-lam)) * 100

# --- 4. THE INTERFACE ---
st.title("🏈 NFL Genius: Pro Parlay Suite")

# Guard against empty data
if not data.empty and 'player_name' in data.columns:
    with st.sidebar:
        st.header("📋 Parlay Builder")
        if st.session_state.parlay_legs:
            for leg in st.session_state.parlay_legs:
                st.success(f"**{leg['player']}**: {leg['pick']} ({leg['edge']}% Edge)")
            if st.button("Reset Parlay"):
                st.session_state.parlay_legs = []
                st.rerun()
        else:
            st.info("No legs added yet.")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Player Selection (Attribute-safe access)
        p_list = sorted([p for p in data['player_name'].unique() if p != "Unknown"])
        p_name = st.selectbox("Select Player", p_list)
        p_sub = data[data['player_name'] == p_name]
        p_pos = p_sub['position'].iloc[-1] if not p_sub.empty else "WR"
        
        # Defense Selection (Restored Feature)
        opp_list = sorted([o for o in data['opponent'].unique() if o != "Unknown"])
        target_def = st.selectbox("Opponent Defense", opp_list)
        
        # Matchup Logic
        d_impact = def_data[(def_data['def_team'] == target_def) & (def_data['position'] == p_pos)]
        mod = 1.15 if (not d_impact.empty and d_impact['receiving_yards'].iloc[0] > 55) else 0.90

    with col2:
        st.subheader("📊 Projections")
        v_line = st.number_input("Market Line", value=50.5)
        
        # Stats Logic
        avg_yds = p_sub['receiving_yards'].mean() if p_pos in ['WR', 'TE'] else p_sub['rushing_yards'].mean()
        proj_yds = avg_yds * mod
        edge = ((proj_yds - v_line) / v_line) * 100
        
        # TD Probability Logic (Restored Feature)
        avg_td = (p_sub['rush_td'].mean() + p_sub['rec_td'].mean())
        td_p = get_td_prob(avg_td, mod)
        
        st.metric("Model Proj", f"{proj_yds:.1f} Yds", delta=f"{edge:.1f}% Edge")
        st.metric("Anytime TD %", f"{td_p:.1f}%")

    # --- PARLAY BUILDER ---
    st.divider()
    if st.button("➕ Add to Parlay", use_container_width=True):
        st.session_state.parlay_legs.append({
            "player": p_name,
            "pick": f"{'Over' if edge > 0 else 'Under'} {v_line} Yds",
            "edge": round(edge, 1)
        })
        st.toast(f"Locked in {p_name}!")

    # Performance Visual
    st.plotly_chart(px.line(p_sub, x='week', y=['rushing_yards', 'receiving_yards'], 
                            title=f"Season Momentum: {p_name}", markers=True), use_container_width=True)
else:
    st.warning("⚠️ Data Syncing... If this continues for more than 10 seconds, check your API/Internet.")
