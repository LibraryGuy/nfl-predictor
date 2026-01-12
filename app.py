import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.graph_objects as go
from nfl_stadiums import NFLStadiums
from datetime import datetime

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
        
        # --- AUTOMATED VENUE & WEATHER LOGIC ---
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        stad_data = stadium_client.get_stadium_by_name(sel_stad_name)
        
        # Determine Roof Type
        roof_type = stad_data.get('roof_type', 'Outdoor').lower()
        is_indoor = any(x in roof_type for x in ['dome', 'indoor', 'retractable'])
        
        st.divider()
        st.subheader("💰 Market Details")
        market_line = st.number_input("Sportsbook Line (Yards)", value=0.0, step=0.5)
        market_odds = st.number_input("American Odds (e.g. -110)", value=-110, step=5)

        st.divider()
        st.subheader("📊 Market Sentiment")
        public_tickets = st.slider("% of Total Bets", 0, 100, 70)
        sharp_money = st.slider("% of Total Cash", 0, 100, 45)

        st.divider()
        st.subheader("🎬 Game Script")
        game_script = st.select_slider("Expected Flow", options=["Defensive Struggle", "Balanced", "Shootout"], value="Balanced")

    # --- FETCH LIVE WEATHER ---
    with st.spinner(f"Fetching forecast for {sel_stad_name}..."):
        # Uses Open-Meteo via nfl-stadiums for a 7-day forecast
        forecast = stadium_client.get_weather_forecast_for_stadium(sel_stad_name, datetime.now().strftime("%Y-%m-%d"))
        
        # Extract hourly average or current (Simplified logic for prompt)
        # Note: If indoor, we override these to neutral
        if is_indoor:
            wind_speed = 0
            precip_prob = 0
            weather_desc = "Controlled (Indoor/Dome)"
        else:
            # Safely navigate forecast JSON (handles API variations)
            hourly = forecast.get('hourly', {})
            wind_speed = sum(hourly.get('wind_speed_10m', [0])[:12]) / 12 # Avg next 12h
            precip_prob = max(hourly.get('precipitation_probability', [0])[:12])
            weather_desc = "Outdoor / Open"

    p_df = data[data['player_name'] == selected_p].copy()
    
    if not p_df.empty:
        p_pos = p_df['position'].iloc[-1]
        stat_map = {'QB': ('passing_yards', 'Pass Yds', 'pass_adj'), 'RB': ('rushing_yards', 'Rush Yds', 'rush_adj'), 
                    'WR': ('receiving_yards', 'Rec Yds', 'pass_adj'), 'TE': ('receiving_yards', 'Rec Yds', 'pass_adj')}
        stat_col, stat_label, adj_key = stat_map.get(p_pos, ('receiving_yards', 'Yds', 'pass_adj'))

        # Multipliers
        hist_avg = p_df[stat_col].mean()
        sos_multiplier = DEF_STATS_2025.get(selected_opp, {}).get(adj_key, 1.0)
        script_boost = {"Defensive Struggle": 0.90, "Balanced": 1.0, "Shootout": 1.15}[game_script]
        
        # Weather Multiplier (Automated)
        weather_multiplier = 1.0
        if not is_indoor:
            if p_pos in ['QB', 'WR', 'TE']:
                if wind_speed > 18: weather_multiplier *= 0.88
                if precip_prob > 50: weather_multiplier *= 0.90
            elif p_pos == 'RB':
                if precip_prob > 50: weather_multiplier *= 1.05 # RBs benefit slightly from rain/snow chaos

        # FINAL PROJECTION
        model_proj = (hist_avg * 1.10) * script_boost * sos_multiplier * weather_multiplier
        
        # TD & Market Math
        td_col = 'passing_tds' if p_pos == 'QB' else 'total_tds'
        if p_pos != 'QB': p_df['total_tds'] = p_df['rushing_tds'] + p_df['receiving_tds']
        td_prob = (len(p_df[p_df[td_col] >= (1.5 if p_pos == 'QB' else 0.5)]) / len(p_df)) * 100
        hit_rate = (p_df[stat_col] >= market_line).mean() * 100 if market_line > 0 else 0
        implied_prob = (abs(market_odds)/(abs(market_odds)+100))*100 if market_odds < 0 else (100/(market_odds+100))*100
        ev_edge = hit_rate - implied_prob

        # --- 3. THE DASHBOARD ---
        st.title(f"📊 {selected_p} Intelligence Hub")
        
        # Live Weather & SOS Alert Bar
        a1, a2, a3 = st.columns(3)
        with a1:
            st.info(f"🏟️ **Venue:** {roof_type.title()}")
        with a2:
            st.warning(f"🌬️ **Live Wind:** {wind_speed:.1f} MPH")
        with a3:
            st.error(f"🌧️ **Precip Prob:** {precip_prob}%")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Season Average", f"{hist_avg:.1f}")
        m2.metric("Weather Impact", f"{weather_multiplier:.2f}x", delta="DOME/CLEAN" if is_indoor else "LIVE CONDITIONS")
        m3.metric("Model Projection", f"{model_proj:.1f}")
        m4.metric("Market Edge (EV)", f"{ev_edge:+.1f}%", delta="SHARP" if (sharp_money-public_tickets)>10 else None)

        st.divider()
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_df.index, y=p_df[stat_col], name="Actuals", line=dict(color='#00c8ff', width=3)))
            fig.add_hline(y=model_proj, line_dash="dash", line_color="#ff4b4b", annotation_text="Auto-Weather Adj.")
            fig.update_layout(title=f"{stat_label} Performance vs Projections", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("🎯 Intelligence Gauges")
            st.write(f"**Market Hit Rate:** {hit_rate:.1f}%")
            st.progress(hit_rate / 100)
            st.write(f"**Money Split:** {sharp_money}% Handle")
            st.progress(sharp_money / 100)
            
            st.divider()
            st.success(f"**Genius Leg:**\n{selected_p} OVER {round(model_proj * 0.85)}+ {stat_label}")
            if ev_edge > 5 and (sharp_money - public_tickets) > 10:
                st.balloons()
                st.warning("💎 **MAX CONVICTION:** Model + Weather + Sharps all align!")

        with st.expander("📂 Raw Matchup Data"):
            st.dataframe(p_df[['week', 'opponent', stat_col, td_col]].tail(10), use_container_width=True)
else:
    st.error("Data Load Error: Please check connectivity.")
