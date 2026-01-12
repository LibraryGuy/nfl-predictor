import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

DEF_STATS_2025 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92}, 'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}, 
}

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        return df.dropna(subset=['player_name', 'opponent']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}"); return pd.DataFrame(), pd.DataFrame()

data, schedules = load_data_pro()
stadium_client = NFLStadiums()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        p_team = data[data['player_name'] == selected_p]['team'].iloc[-1]
        
        # --- AUTOMATED GAME SCRIPT ---
        game = schedules[((schedules['home_team'] == p_team) & (schedules['away_team'] == selected_opp)) | 
                         ((schedules['away_team'] == p_team) & (schedules['home_team'] == selected_opp))].iloc[-1:]
        
        if not game.empty:
            v_total = game['total_line'].values[0]
            v_spread = game['spread_line'].values[0] if game['home_team'].values[0] == p_team else -game['spread_line'].values[0]
            if v_total > 48: auto_script = "Shootout"
            elif v_total < 41: auto_script = "Defensive Struggle"
            elif v_spread < -7: auto_script = "Heavy Lead"
            else: auto_script = "Balanced"
        else: auto_script, v_total, v_spread = "Balanced", 44.0, 0.0

        st.success(f"🤖 **Auto-Script:** {auto_script}")
        
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds", value=-110, step=5)
        sharp_money = st.slider("% Sharp Handle", 0, 100, 45)

    # --- 3. PROJECTION LOGIC ---
    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'Pass Yds', 'pass_adj'), 'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'), 
                    'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'), 'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')}
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        hist_avg = p_df[stat_col].mean()
        sos_mult = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.88, "Balanced": 1.0, "Shootout": 1.18, "Heavy Lead": 0.94}[auto_script]
        model_proj = (hist_avg * 1.10) * script_boost * sos_mult

        # --- 4. THE RE-INTEGRATED PARLAY SECTION ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        
        col_m, col_p = st.columns([2, 1])
        
        with col_m:
            m1, m2, m3 = st.columns(3)
            m1.metric("Vegas Line", f"{v_total} O/U", delta=auto_script)
            m2.metric("Model Projection", f"{model_proj:.1f}")
            m3.metric("Edge", f"{((model_proj/market_line)-1)*100:.1f}%" if market_line > 0 else "0%")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=p_df[stat_col], name="History", line=dict(color='#00c8ff')))
            fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b")
            st.plotly_chart(fig, use_container_width=True)

        with col_p:
            st.subheader("🔥 Parlay Recommendations")
            
            # Leg 1: The Safe Prop
            safe_val = round(model_proj * 0.82)
            st.info(f"**Leg 1 (Safe):**\n{selected_p} {safe_val}+ {stat_label}")
            
            # Leg 2: Correlated Correlation
            if auto_script == "Shootout":
                st.warning(f"**Leg 2 (Correlation):**\nGame Total OVER {v_total - 1.5}")
            elif auto_script == "Heavy Lead":
                st.warning(f"**Leg 2 (Correlation):**\n{p_team} Moneyline")
            else:
                st.warning(f"**Leg 2 (Correlation):**\n{selected_opp} Team Total UNDER")
            
            st.divider()
            st.success(f"**Genius Power-Parlay:**\nCombine both for approx. +145 Odds")

            if sharp_money > 60:
                st.error("⚠️ SHARP WHALE ALERT: Heavy pro money on this game flow.")
