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
    'CIN': {'pass_adj': 1.11, 'rush_adj': 1.27}, 
}

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        # Load Player Stats & Schedule (for lines)
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in stat_cols:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].fillna(0)
            
        return df.dropna(subset=['player_name', 'opponent', 'position']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

data, schedules = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. THE INTELLIGENCE ENGINE ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        # Determine Player's Team
        p_team = data[data['player_name'] == selected_p]['team'].iloc[-1]
        
        # --- AUTOMATED GAME SCRIPT LOGIC ---
        game = schedules[((schedules['home_team'] == p_team) & (schedules['away_team'] == selected_opp)) | 
                         ((schedules['away_team'] == p_team) & (schedules['home_team'] == selected_opp))].iloc[-1:]
        
        if not game.empty:
            v_total = game['total_line'].values[0]
            v_spread = game['spread_line'].values[0] if game['home_team'].values[0] == p_team else -game['spread_line'].values[0]
            
            if v_total > 48: auto_script = "Shootout"
            elif v_total < 41: auto_script = "Defensive Struggle"
            elif v_spread < -7: auto_script = "Heavy Lead"
            else: auto_script = "Balanced"
        else:
            auto_script, v_total, v_spread = "Balanced", 44.0, 0.0

        st.info(f"🤖 **Auto-Script:** {auto_script}\n(Total: {v_total} | Spread: {v_spread})")
        
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        stad_data = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type = stad_data.get('roof_type', 'Outdoor').lower() if stad_data else 'outdoor'
        is_indoor = any(x in roof_type for x in ['dome', 'indoor', 'retractable'])
        
        st.divider()
        st.subheader("💰 Market Details")
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds (e.g. -110)", value=-110, step=5)

        st.divider()
        st.subheader("📊 Market Sentiment")
        public_tickets = st.slider("% of Total Bets (Tickets)", 0, 100, 70)
        sharp_money = st.slider("% of Total Cash (Handle)", 0, 100, 45)

    # --- 3. WEATHER ENGINE ---
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        forecast = stadium_client.get_weather_forecast_for_stadium(sel_stad_name, today_str)
        if is_indoor or not forecast:
            wind_speed, precip_prob = 0, 0
        else:
            hourly = forecast.get('hourly', {})
            wind_speed = sum(hourly.get('wind_speed_10m', [0])[:8]) / 8
            precip_prob = max(hourly.get('precipitation_probability', [0])[:8])
    except:
        wind_speed, precip_prob = 5, 0

    p_df = data[data['player_name'] == selected_p].copy()
    
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'Pass Yds', 'pass_adj'), 'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'), 
                    'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'), 'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')}
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        # Calculations
        hist_avg = p_df[stat_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.88, "Balanced": 1.0, "Shootout": 1.18, "Heavy Lead": 0.94}[auto_script]
        
        weather_multiplier = 1.0
        if not is_indoor:
            if p_pos != 'RB' and wind_speed > 18: weather_multiplier *= 0.85
            if precip_prob > 50: weather_multiplier *= 0.92

        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier * weather_multiplier
        
        # Market Math
        implied_prob = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        ev_edge = hit_rate - implied_prob

        # --- 4. THE DASHBOARD ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Automated Script", auto_script)
        m2.metric("Total Multiplier", f"{sos_multiplier * script_boost * weather_multiplier:.2f}x")
        m3.metric("Model Projection", f"{model_proj:.1f}")
        m4.metric("Market Edge (EV)", f"{ev_edge:+.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=p_df[stat_col], name="Actuals", line=dict(color='#00c8ff', width=3)))
        fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Auto Projection")
        fig.update_layout(template="plotly_dark", title=f"Trend Analysis: {auto_script} Environment")
        st.plotly_chart(fig, use_container_width=True)
