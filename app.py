import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

# --- 1. SETTINGS ---
st.set_page_config(page_title="NFL Sharp: Betting Engine", layout="wide", page_icon="💰")

# Defensive Stats
DEF_STATS_2026 = {
    'HOU': {'pass_adj': 0.87, 'rush_adj': 0.81}, 'DEN': {'pass_adj': 0.89, 'rush_adj': 0.79},
    'CLE': {'pass_adj': 0.80, 'rush_adj': 1.01}, 'DAL': {'pass_adj': 1.19, 'rush_adj': 1.09}, 
    'BAL': {'pass_adj': 1.18, 'rush_adj': 0.92}, 'JAX': {'pass_adj': 1.04, 'rush_adj': 0.74}
}

@st.cache_data(ttl=3600)
def load_betting_data():
    try:
        # 1. Raw Load
        raw_df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # 2. Flatten MultiIndex if it exists
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = ['_'.join(filter(None, map(str, col))).strip().lower() for col in raw_df.columns.values]
        else:
            raw_df.columns = [str(c).lower() for c in raw_df.columns]
        
        # 3. BRUTE FORCE RENAMING (The Fix for Line 46)
        # We look for the most common naming patterns in nflverse data
        col_map = {}
        for c in raw_df.columns:
            if 'player_display_name' in c or (c == 'player_name'): col_map[c] = 'player_name'
            if 'recent_team' in c or (c == 'team'): col_map[c] = 'team'
            if 'opponent_team' in c or (c == 'opponent'): col_map[c] = 'opponent'
        
        df = raw_df.rename(columns=col_map)
        
        # 4. Standardize Stats
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            # Find the best match if exact name isn't there
            actual_col = next((c for c in df.columns if col in c), None)
            if actual_col:
                df[col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0)
            else:
                df[col] = 0
            
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame()

data = load_betting_data()
stadium_client = NFLStadiums()

# --- 2. THE BETTING INTERFACE ---
if not data.empty:
    with st.sidebar:
        st.header("🔍 Prop Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Vs. Defense", sorted(data['opponent'].unique()))
        sel_stad_name = st.selectbox("Stadium", sorted(stadium_client.get_list_of_stadium_names()))
        
        st.divider()
        market_line = st.number_input("Vegas Line", value=0.0, step=0.5)
        market_odds = st.number_input("Odds (-110)", value=-110, step=5)
        sharp_money = st.slider("Sharp Money %", 0, 100, 55)
        public_tickets = st.slider("Public Tickets %", 0, 100, 45)
        injury_boost = st.checkbox("Teammate Out (+Volume)")

    # --- 3. SAFE WEATHER LOGIC ---
    stad_data = stadium_client.get_stadium_by_name(sel_stad_name)
    is_indoor = False
    if stad_data:
        is_indoor = any(x in stad_data.get('roof_type', '').lower() for x in ['dome', 'indoor', 'retractable'])
    
    wind_speed = 0
    if not is_indoor:
        try:
            forecast = stadium_client.get_weather_forecast_for_stadium(sel_stad_name, datetime.now().strftime("%Y-%m-%d"))
            wind_speed = sum(forecast.get('hourly', {}).get('wind_speed_10m', [0])[:8]) / 8
        except:
            wind_speed = 5

    # --- 4. CALCULATION ENGINE ---
    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': 'passing_yards', 'RB': 'rushing_yards', 'WR': 'receiving_yards', 'TE': 'receiving_yards'}
        stat_col = stat_map.get(p_pos, 'receiving_yards')
        
        # Math Multipliers
        sos = DEF_STATS_2026.get(selected_opp, {}).get('pass_adj' if p_pos != 'RB' else 'rush_adj', 1.0)
        vol = 1.15 if injury_boost else 1.0
        weather = 0.90 if wind_speed > 15 and p_pos != 'RB' else 1.0
        
        model_proj = p_df[stat_col].mean() * sos * vol * weather
        
        # EV Edge
        implied = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        ev_edge = hit_rate - implied

        # --- 5. DASHBOARD ---
        st.title(f"🏈 {selected_p} Intelligence")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Projection", f"{model_proj:.1f}")
        col2.metric("Market Edge", f"{ev_edge:+.1f}%", delta="VALUE" if ev_edge > 0 else None)
        col3.metric("Money Delta", f"{sharp_money - public_tickets:+.0f}%", help="Positive means Pros are on this.")

        st.divider()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p_df[stat_col], name="Last 20 Games", line=dict(color='#00ff88')))
        fig.add_hline(y=market_line, line_dash="dot", line_color="yellow", annotation_text="Vegas")
        fig.add_hline(y=model_proj, line_dash="dash", line_color="red", annotation_text="Sharp")
        fig.update_layout(template="plotly_dark", title="Trend vs Market")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Awaiting Data Sync...")
