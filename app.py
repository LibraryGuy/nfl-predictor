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
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 
                     'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].fillna(0)
            
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

data = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. THE INTELLIGENCE ENGINE ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        player_list = sorted(data['player_name'].unique())
        
        selected_p = st.selectbox("Select Player", player_list)
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        sel_stad = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        
        st.divider()
        st.subheader("💰 Market Details")
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        # NEW: American Odds Input
        market_odds = st.number_input("American Odds (e.g. -110)", value=-110, step=5)

    p_df = data[data['player_name'] == selected_p].copy()
    
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        
        stat_map = {
            'QB': ('passing_yards', 'Pass Yds'),
            'RB': ('rushing_yards', 'Rush Yds'),
            'WR': ('receiving_yards', 'Rec Yds'),
            'TE': ('receiving_yards', 'Rec Yds')
        }
        stat_col, stat_label = stat_map.get(p_pos, ('receiving_yards', 'Yds'))

        # Calculations
        hist_avg = p_df[stat_col].mean()
        last_3_avg = p_df[stat_col].tail(3).mean()
        
        # TD Logic
        if p_pos == 'QB':
            td_col = 'passing_tds'
        else:
            p_df['total_tds'] = p_df['rushing_tds'] + p_df['receiving_tds']
            td_col = 'total_tds'

        games_played = len(p_df)
        td_games = len(p_df[p_df[td_col] >= 1])
        td_prob = (td_games / games_played) * 100 if games_played > 0 else 0
        
        # Projection & Basic Edge
        model_proj = hist_avg * 1.10 
        yardage_edge = ((model_proj - market_line) / market_line * 100) if market_line > 0 else 0

        # --- GENIUS LOGIC: MARKET IMPLIED PROBABILITY & EV ---
        # Convert American Odds to Implied Prob %
        if market_odds < 0:
            implied_prob = (abs(market_odds) / (abs(market_odds) + 100)) * 100
        else:
            implied_prob = (100 / (market_odds + 100)) * 100
        
        # Determine Reliability (Hit Rate of the market line)
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        
        # Expected Value (EV) Edge
        # Formula: (Win Prob * Payout) - (Loss Prob * Stake)
        # Simplified as Percentage Edge: Your Prob - Market Implied Prob
        ev_edge = hit_rate - implied_prob

        # --- 3. THE DASHBOARD ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        st.markdown(f"**Position:** {p_pos} | **Matchup:** vs {selected_opp} | **Venue:** {sel_stad}")

        # Top Row: Metrics (Updated with EV Edge)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Average", f"{hist_avg:.1f}")
        m2.metric("L3 Game Trend", f"{last_3_avg:.1f}", delta=f"{last_3_avg - hist_avg:+.1f}")
        m3.metric("Model Projection", f"{model_proj:.1f}")
        # NEW Genius Metric: EV Edge
        m4.metric("Market Edge (EV)", f"{ev_edge:+.1f}%", 
                  delta="POS VALUE" if ev_edge > 0 else "BAD VALUE",
                  delta_color="normal" if ev_edge > 0 else "inverse")

        st.divider()

        # Middle Row: Visuals
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_df['week'], y=p_df[stat_col], name="Actuals", line=dict(color='#00c8ff', width=3)))
            fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Model Projection")
            if market_line > 0:
                fig.add_hline(y=market_line, line_dash="dot", line_color="yellow", annotation_text="Market Line")
            fig.update_layout(title=f"{stat_label} Performance vs. Projections", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🎯 Intelligence Gauges")
            
            # Gauge 1: Yardage Reliability (vs Market Line)
            st.write(f"**Market Hit Rate:** {hit_rate:.1f}%")
            st.progress(hit_rate / 100)
            st.caption(f"Historical frequency of hitting {market_line} yards.")
            
            # Gauge 2: TD Probability
            st.write(f"**TD Probability (Any):** {td_prob:.1f}%")
            st.progress(td_prob / 100)
            st.caption(f"Scored in {td_games} of {games_played} games.")
            
            st.divider()
            
            # Genius Parlay Recommendation
            st.success(f"**Genius Leg:**\n{selected_p} OVER {round(model_proj * 0.85)}+ {stat_label}")
            if ev_edge > 5:
                st.info(f"💎 **Math Edge:** Your probability ({hit_rate:.0f}%) is higher than the house ({implied_prob:.0f}%).")
            if td_prob > 60:
                st.warning(f"🔥 **High Value Add:** {selected_p} Anytime TD")

        with st.expander("📂 Raw Matchup Data & Splits"):
            display_cols = ['week', 'opponent', stat_col, td_col, 'team']
            st.dataframe(p_df[display_cols].tail(10), use_container_width=True)
else:
    st.error("Data Load Error: Please check connectivity.")
