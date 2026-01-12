import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

# --- 1. SETTINGS & DATA ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

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
        # AUTOMATED DATA LOADING: Pull stats and schedules simultaneously
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

# Unpack both dataframes
data, schedules = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. THE INTELLIGENCE ENGINE ---
if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        # LOGIC-DRIVEN SCRIPT ENGINE: Identify Team and Matchup Line
        p_team = data[data['player_name'] == selected_p]['team'].iloc[-1]
        
        # Find the game in the schedule to get Vegas lines
        matchup = schedules[((schedules['home_team'] == p_team) & (schedules['away_team'] == selected_opp)) | 
                           ((schedules['away_team'] == p_team) & (schedules['home_team'] == selected_opp))].iloc[-1:]
        
        # Default script values
        v_total = 44.5
        v_spread = 0.0
        auto_script_val = "Balanced"
        
        if not matchup.empty:
            v_total = matchup['total_line'].values[0]
            # Adjust spread to be relative to the selected player's team
            v_spread = matchup['spread_line'].values[0] if matchup['home_team'].values[0] == p_team else -matchup['spread_line'].values[0]
            
            # AUTOMATED LOGIC: Categorize based on Vegas
            if v_total > 49: auto_script_val = "Shootout"
            elif v_total < 41: auto_script_val = "Defensive Struggle"
            else: auto_script_val = "Balanced"

        st.info(f"🤖 **Vegas Pulse:** {v_total} O/U | Spread: {v_spread}")
        
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        stad_data = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type = stad_data.get('roof_type', 'Outdoor').lower()
        is_indoor = any(x in roof_type for x in ['dome', 'indoor', 'retractable'])
        
        st.divider()
        st.subheader("💰 Market Details")
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds (e.g. -110)", value=-110, step=5)

        st.divider()
        st.subheader("📊 Market Sentiment")
        public_tickets = st.slider("% of Total Bets (Tickets)", 0, 100, 70)
        sharp_money = st.slider("% of Total Cash (Handle)", 0, 100, 45)

        st.divider()
        st.subheader("🎬 Game Script Engine")
        # Pre-set the slider index based on the automated logic
        script_options = ["Defensive Struggle", "Balanced", "Shootout"]
        default_idx = script_options.index(auto_script_val)
        game_script = st.select_slider("Expected Flow", options=script_options, value=auto_script_val)

    # --- 3. WEATHER & PROJECTION ENGINE ---
    # (Weather logic remains identical to your provided code)
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        forecast = stadium_client.get_weather_forecast_for_stadium(sel_stad_name, today_str)
        if is_indoor or not forecast:
            wind_speed, precip_prob = 0, 0
        else:
            hourly = forecast.get('hourly', {})
            wind_list = hourly.get('wind_speed_10m', [0])
            precip_list = hourly.get('precipitation_probability', [0])
            wind_speed = sum(wind_list[:8]) / 8 if wind_list else 5
            precip_prob = max(precip_list[:8]) if precip_list else 0
    except:
        wind_speed, precip_prob = 5, 0

    p_df = data[data['player_name'] == selected_p].copy()
    
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'Pass Yds', 'pass_adj'), 'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'), 
                    'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'), 'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')}
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        hist_avg = p_df[stat_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        
        weather_multiplier = 1.0
        if not is_indoor:
            if p_pos in ['QB', 'WR', 'TE']:
                if wind_speed > 18: weather_multiplier *= 0.88
                if precip_prob > 50: weather_multiplier *= 0.90
            elif p_pos == 'RB' and precip_prob > 50: 
                weather_multiplier *= 1.05

        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier * weather_multiplier
        
        # --- 4. THE DASHBOARD ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        
        # Implied Math for value detection
        implied_prob = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        ev_edge = hit_rate - implied_prob
        money_delta = sharp_money - public_tickets

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Automated Script", game_script, delta=f"{v_total} Total")
        m2.metric("Total Multiplier", f"{sos_multiplier * script_boost * weather_multiplier:.2f}x")
        m3.metric("Model Projection", f"{model_proj:.1f}")
        m4.metric("Market Edge (EV)", f"{ev_edge:+.1f}%")

        # Visualizations
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=p_df[stat_col], name="History", line=dict(color='#00c8ff')))
            fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Sharp Target")
            fig.update_layout(template="plotly_dark", title=f"{stat_label} Trend Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🔥 Parlay Recommendations")
            safe_val = round(model_proj * 0.85)
            st.success(f"**Safe Leg:** {selected_p} {safe_val}+ {stat_label}")
            if game_script == "Shootout":
                st.warning(f"**Correlated Leg:** Over {v_total - 1.5} Total Pts")
            elif v_spread < -6:
                st.warning(f"**Correlated Leg:** {p_team} Moneyline")
            
            st.divider()
            st.write(f"**Sharpness Gauge:** {money_delta}% Handle Lead")
            st.progress(max(0, min(100, 50 + money_delta)) / 100)

else:
    st.error("Critical Failure: Data connection lost.")
