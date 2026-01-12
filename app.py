import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

# SOS Data (Defensive Multipliers)
DEF_STATS_2025 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 
    'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 
    'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92},
    'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}, 
    'CIN': {'pass_adj': 1.11, 'rush_adj': 1.27}, 
}

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']
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
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        sel_stad = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        
        st.divider()
        st.subheader("💰 Market Details")
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds (e.g. -110)", value=-110, step=5)

        st.divider()
        st.subheader("📊 Market Sentiment (Tickets vs Money)")
        public_tickets = st.slider("% of Total Bets (Tickets)", 0, 100, 70)
        sharp_money = st.slider("% of Total Cash (Handle)", 0, 100, 45)

        st.divider()
        st.subheader("🎬 Game Script & Weather")
        game_script = st.select_slider("Expected Game Flow", options=["Defensive Struggle", "Balanced", "Shootout"], value="Balanced")
        precip = st.selectbox("Precipitation", ["None", "Rain", "Heavy Snow"])
        wind_speed = st.slider("Wind Speed (MPH)", 0, 40, 5)

    p_df = data[data['player_name'] == selected_p].copy()
    
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {
            'QB': ('passing_yards', 'Pass Yds', 'pass_adj'),
            'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'),
            'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'),
            'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')
        }
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        # Calculations
        hist_avg = p_df[stat_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        
        # Weather Logic
        weather_multiplier = 1.0
        if p_pos in ['QB', 'WR', 'TE']:
            if wind_speed > 20: weather_multiplier *= 0.85 
            elif wind_speed > 15: weather_multiplier *= 0.93
            if precip == "Rain": weather_multiplier *= 0.88 
            elif precip == "Heavy Snow": weather_multiplier *= 0.75 
        else: # RB
            if precip == "Heavy Snow": weather_multiplier *= 1.10
            if wind_speed > 20: weather_multiplier *= 1.05

        # FINAL PROJECTION
        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier * weather_multiplier
        
        # Market Math
        implied_prob = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        ev_edge = hit_rate - implied_prob

        # --- SHARP INDICATOR LOGIC ---
        money_delta = sharp_money - public_tickets
        is_sharp = money_delta > 10  # Money % significantly higher than ticket %
        is_public = public_tickets > 75 # "Public Hero" status

        # --- 3. THE DASHBOARD ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        
        # Multi-Alert Banner
        cols = st.columns(3)
        with cols[0]:
            if sos_multiplier < 1.0: st.error(f"📉 SOS: {selected_opp} -{abs(sos_multiplier-1)*100:.0f}%")
        with cols[1]:
            if weather_multiplier != 1.0: st.warning(f"🌧️ Weather: {abs(weather_multiplier-1)*100:.0f}% Impact")
        with cols[2]:
            if is_sharp: st.success("💎 SHARP SIGNAL: Big money backing this side.")
            elif is_public: st.info("📢 PUBLIC FAVORITE: Casuals are heavy on this.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Average", f"{hist_avg:.1f}")
        m2.metric("Money vs Tickets", f"{sharp_money}% / {public_tickets}%", delta=f"{money_delta:+.1f}% Spread")
        m3.metric("Model Projection", f"{model_proj:.1f}")
        m4.metric("Market Edge (EV)", f"{ev_edge:+.1f}%", delta="POS VALUE" if ev_edge > 0 else "BAD VALUE", delta_color="normal" if ev_edge > 0 else "inverse")

        st.divider()
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df[stat_col], name="Actuals", line=dict(color='#00c8ff', width=3)))
            fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Final Adj. Projection")
            if market_line > 0: fig.add_hline(y=market_line, line_dash="dot", line_color="yellow", annotation_text="Market Line")
            fig.update_layout(title=f"{stat_label} Performance Analysis", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🎯 Intelligence Gauges")
            st.write(f"**Historical Hit Rate:** {hit_rate:.1f}%")
            st.progress(hit_rate / 100)
            
            # Sentiment Gauge
            st.write(f"**Sharpness Index:** {sharp_money}% Cash")
            st.progress(sharp_money / 100)
            st.caption("Professionals (Sharp) usually represent a higher % of money than tickets.")
            
            st.divider()
            st.success(f"**Genius Leg:**\n{selected_p} OVER {round(model_proj * 0.85)}+ {stat_label}")
            if is_sharp and ev_edge > 0:
                st.warning("🔥 **SHARP/MODEL ALIGNMENT:** Model and Pro Money both love this.")

        with st.expander("📂 Raw Matchup Data & Splits"):
            st.dataframe(p_df[['week', 'opponent', stat_col, 'passing_tds' if p_pos=='QB' else 'rushing_tds']].tail(10), use_container_width=True)
else:
    st.error("Data Load Error: Please check connectivity.")
