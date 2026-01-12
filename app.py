import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

# --- 1. PRO-TIER SETTINGS ---
st.set_page_config(page_title="NFL Sharp: Professional Prediction Engine", layout="wide", page_icon="💰")

# Defensive Strength-of-Schedule (SOS) Data
DEF_STATS_2026 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92}, 'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}
}

@st.cache_data(ttl=3600)
def load_betting_data():
    try:
        # Load data and immediately convert to Pandas
        raw_df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # --- FIX FOR LINE 41: Flatten MultiIndex Columns ---
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in raw_df.columns.values]
        
        # Reset and Clean Column names
        raw_df.columns = [str(c).lower() for c in raw_df.columns]
        
        # Dynamic Mapping (handles both possible naming conventions)
        rename_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent'
        }
        df = raw_df.rename(columns=rename_map)
        
        # Ensure critical betting stats exist
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame()

data = load_betting_data()
stadium_client = NFLStadiums()

# --- 2. BETTING INTELLIGENCE ENGINE ---
if not data.empty:
    with st.sidebar:
        st.header("🔍 Prop Intelligence")
        selected_p = st.selectbox("Select Target Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Versus Defense", sorted(data['opponent'].unique()))
        sel_stad_name = st.selectbox("Venue (Auto-Weather)", sorted(stadium_client.get_list_of_stadium_names()))
        
        st.divider()
        st.subheader("📊 Market Sentiment")
        market_line = st.number_input("Sportsbook Line", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds (-110)", value=-110, step=5)
        sharp_money = st.slider("Sharp Money % (Handle)", 0, 100, 50)
        public_tickets = st.slider("Public Bet % (Tickets)", 0, 100, 50)

        st.divider()
        st.subheader("⚡ Volume Adjustments")
        injury_status = st.checkbox("Teammate (WR1/RB1) is OUT")
        game_script = st.select_slider("Game Script", options=["Heavy Lead", "Balanced", "Playing From Behind"])

    # --- AUTOMATED WEATHER ---
    stad_data = stadium_client.get_stadium_by_name(sel_stad_name)
    is_indoor = any(x in stad_data.get('roof_type', '').lower() for x in ['dome', 'indoor', 'retractable'])
    try:
        forecast = stadium_client.get_weather_forecast_for_stadium(sel_stad_name, datetime.now().strftime("%Y-%m-%d"))
        hourly = forecast.get('hourly', {})
        wind_speed = sum(hourly.get('wind_speed_10m', [0])[:8]) / 8 if not is_indoor else 0
        precip_prob = max(hourly.get('precipitation_probability', [0])[:8]) if not is_indoor else 0
    except:
        wind_speed, precip_prob = 0, 0

    # --- PROJECTION LOGIC ---
    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'Pass Yds', 'pass_adj'), 'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'), 
                    'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'), 'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')}
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        # Multipliers
        base_avg = p_df[stat_col].mean()
        sos_mod = DEF_STATS_2026.get(selected_opp, {}).get(adj_key, 1.0)
        injury_mod = 1.18 if injury_status else 1.0 
        
        weather_mod = 1.0
        if not is_indoor:
            if wind_speed > 18 and p_pos != 'RB': weather_mod *= 0.85
            if precip_prob > 60 and p_pos == 'RB': weather_mod *= 1.05

        script_mod = {"Heavy Lead": 0.85 if p_pos == 'QB' else 1.15, "Balanced": 1.0, "Playing From Behind": 1.25 if p_pos != 'RB' else 0.80}[game_script]

        final_proj = base_avg * sos_mod * injury_mod * weather_mod * script_mod
        
        # EV/Market Logic
        implied_p = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        ev_edge = hit_rate - implied_p

        # --- DASHBOARD ---
        st.title(f"🚀 {selected_p} Betting Intelligence")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharp Projection", f"{final_proj:.1f}")
        m2.metric("Market Edge (EV)", f"{ev_edge:+.1f}%")
        m3.metric("Money Spread", f"{sharp_money - public_tickets:+.0f}%")
        m4.metric("Conditions", "Perfect" if is_indoor else f"{wind_speed:.0f} MPH")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df[stat_col], name="Actual Yards", line=dict(color='#00ff88', width=3)))
        fig.add_hline(y=market_line, line_dash="dot", line_color="yellow", annotation_text="Bookie Line")
        fig.add_hline(y=final_proj, line_dash="dash", line_color="red", annotation_text="Sharp Model")
        fig.update_layout(template="plotly_dark", title=f"Trend vs. Model")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Data Hub Offline. Ensure nflreadpy is installed.")
