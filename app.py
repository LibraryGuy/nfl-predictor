import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_data_pro():
    df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
    rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']:
        if col not in df.columns: df[col] = 0
        df[col] = df[col].fillna(0)
    return df.dropna(subset=['player_name', 'opponent', 'position'])

data = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. THE INTELLIGENCE ENGINE ---
if not data.empty:
    # Sidebar Selections
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        sel_stad = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        market_line = st.number_input("Sportsbook Line", value=0.0, step=0.5)

    # Player Context
    p_df = data[data['player_name'] == selected_p].copy()
    p_pos = p_df['position'].iloc[-1]
    
    # Dynamic Stat Allocation
    stat_map = {
        'QB': ('passing_yards', 'Pass Yds'),
        'RB': ('rushing_yards', 'Rush Yds'),
        'WR': ('receiving_yards', 'Rec Yds'),
        'TE': ('receiving_yards', 'Rec Yds')
    }
    stat_col, stat_label = stat_map.get(p_pos, ('receiving_yards', 'Yds'))

    # Calculate Multi-Point Metrics
    hist_avg = p_df[stat_col].mean()
    last_3_avg = p_df[stat_col].tail(3).mean()
    
    # Mock Projection Logic (Integrating your previous Defense/Weather mods)
    # Note: In your real code, replace '1.10' with your actual calculated mods
    model_proj = hist_avg * 1.10 
    edge = ((model_proj - market_line) / market_line * 100) if market_line > 0 else 0

    # --- 3. THE DASHBOARD ---
    st.title(f"📊 {selected_p} Intelligence Hub")
    st.markdown(f"**Position:** {p_pos} | **Matchup:** vs {selected_opp} | **Venue:** {sel_stad}")

    # Top Row: The "Big Three" Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season Average", f"{hist_avg:.1f}")
    m2.metric("L3 Game Trend", f"{last_3_avg:.1f}", delta=f"{last_3_avg - hist_avg:+.1f}")
    m3.metric("Model Projection", f"{model_proj:.1f}")
    m4.metric("Market Edge", f"{edge:.1f}%", delta=f"{edge:.1f}%" if edge > 5 else None)

    st.divider()

    # Middle Row: Visual Intelligence
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Comparison Chart: Projection vs History
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_df['week'], y=p_df[stat_col], name="Actuals", line=dict(color='#00c8ff', width=3)))
        fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Model Projection")
        if market_line > 0:
            fig.add_hline(y=market_line, line_dash="dot", line_color="yellow", annotation_text="Market Line")
        
        fig.update_layout(title=f"{stat_label} Performance vs. Projections", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Consistency Radar / Stats
        st.subheader("🎯 Consistency Profile")
        hit_80 = (p_df[stat_col] >= (model_proj * 0.8)).mean() * 100
        st.write(f"**Reliability Score:** {hit_80:.1f}%")
        st.caption("Percentage of games where player achieved at least 80% of current projection.")
        
        st.progress(hit_80 / 100)
        
        # Genius Parlay Recommendation (Pinned here)
        st.success(f"**Genius Leg:**\n{selected_p} OVER {round(model_proj * 0.85)}+ {stat_label}")

    # Bottom Row: Advanced Table
    with st.expander("📂 Raw Matchup Data & Splits"):
        st.dataframe(p_df[['week', 'opponent', stat_col, 'receptions', 'team']].tail(10), use_container_width=True)
