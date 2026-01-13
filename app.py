import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

# --- WEATHER IMPACT LOGIC ---
def get_weather_multiplier(roof_type, wind, temp, precip, p_pos):
    if roof_type in ['Dome', 'Closed', 'Indoor']:
        return 1.0, "Dome (No Impact)"
    multiplier, impact_reasons = 1.0, []
    if wind >= 15:
        penalty = 0.05 if wind < 20 else 0.12
        if p_pos in ['QB', 'WR', 'TE']:
            multiplier -= penalty
            impact_reasons.append(f"High Wind (-{int(penalty*100)}%)")
        elif p_pos == 'RB':
            multiplier += 0.03
            impact_reasons.append("Wind Vol. Boost (+3%)")
    if precip in ['Rain', 'Snow']:
        multiplier -= 0.05
        impact_reasons.append(f"{precip} (-5%)")
    if temp <= 20:
        multiplier -= 0.03
        impact_reasons.append("Extreme Cold (-3%)")
    return round(multiplier, 2), (" + ".join(impact_reasons) if impact_reasons else "Fair Weather")

# --- CORE LOGIC ---
def get_dynamic_sos(data, stat_col):
    if data.empty: return {}
    league_avg = data[stat_col].mean()
    return (data.groupby('opponent')[stat_col].mean() / league_avg).to_dict()

def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td=False):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    offset = risk_map[risk_level]["offset"]
    
    # Poisson for TDs (Discrete), Normal for Yards (Continuous)
    if is_td:
        # For TDs, Conservative is usually 0.5+, Aggressive is often 1.5+
        primary_val = 0.5 if risk_level != "Aggressive (+200)" else 1.5
    else:
        primary_val = max(0, round(p_mean + (offset * p_std)))
    
    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)].sort_values(['season', 'week'], ascending=False)
    if not teammates.empty:
        latest_season = teammates['season'].max()
        if p_pos == 'QB':
            valid_targets = teammates[(teammates['season'] == latest_season) & (teammates['position'].isin(['WR', 'TE']))]
            if not valid_targets.empty:
                top_target = valid_targets.groupby('player_name')['receiving_yards'].sum().idxmax()
                leg_val = 40 if risk_level == "Conservative (-104)" else 60
                parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
        elif p_pos in ['WR', 'TE', 'RB']:
            current_qbs = teammates[(teammates['position'] == 'QB') & (teammates['season'] == latest_season)]
            if not current_qbs.empty:
                qb_name = current_qbs[current_qbs['week'] == current_qbs['week'].max()]['player_name'].iloc[0]
                leg_val = 195 if risk_level == "Conservative (-104)" else 240
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "QB Link"})
    return parlay_legs

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        sched = nfl.load_schedules(seasons=[2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'passing_tds', 'rushing_tds', 'receiving_tds']:
            df[col] = df.get(col, 0).fillna(0)
        return df.dropna(subset=['player_name', 'opponent', 'position']), sched
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. UI RENDERING ---
# FIXED: Safe unpacking of the cached data
raw_data, schedules = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

if not data.empty:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_team = p_df['team'].iloc[-1] if not p_df.empty else "N/A"
        p_pos = p_df['position'].iloc[-1] if not p_df.empty else "WR"

        # Market Selection
        stat_options = {
            'Yards': ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards'),
            'Touchdowns': ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        }
        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        stat_col = stat_options[selected_market]
        is_td_market = "tds" in stat_col

        market_line = st.number_input("Sportsbook Line", value=0.5 if is_td_market else 50.0, step=0.5)
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
        
        # Weather & Stadium logic
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type = stad_obj.get('roof_type', 'Outdoor') if stad_obj else 'Outdoor'
        
        if roof_type in ['Dome', 'Closed', 'Indoor']:
            w_wind, w_temp, w_precip = 0, 70, "None"
        else:
            w_wind = st.slider("Wind Speed (MPH)", 0, 40, 5)
            w_temp = st.slider("Temperature (F)", -10, 100, 55)
            w_precip = st.selectbox("Precipitation", ["None", "Rain", "Snow"])

    # --- Calculations ---
    p_mean = p_df[stat_col].mean()
    p_std = p_df[stat_col].std() if len(p_df) > 1 else (p_mean * 0.4) # Fallback std dev
    
    weather_mult, weather_reason = get_weather_multiplier(roof_type, w_wind, w_temp, w_precip, p_pos)
    sos_mult = get_dynamic_sos(data, stat_col).get(selected_opp, 1.0)
    model_proj = p_mean * sos_mult * weather_mult

    # Probability Calculation
    if is_td_market:
        # Poisson: 1 - P(X <= line - 1)
        win_prob = (1 - poisson.cdf(max(0, market_line - 0.5), model_proj)) * 100
    else:
        # Normal: 1 - P(X < line)
        win_prob = (1 - norm.cdf(market_line, model_proj, p_std)) * 100

    # --- Display ---
    st.title(f"📊 {selected_p} Intelligence Hub")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.metric("Model Projection", f"{round(model_proj, 1)} {selected_market}", f"{round(win_prob, 1)}% Prob > Line")
        
        # Hit Rate Chart
        last_5 = p_df.tail(5).copy()
        last_5['hit'] = last_5[stat_col] >= market_line
        fig = go.Figure(go.Bar(x=[f"Wk {w}" for w in last_5['week']], y=last_5[stat_col], 
                               marker_color=['#00ff96' if h else '#4a4a4a' for h in last_5['hit']]))
        fig.add_hline(y=market_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Line")
        fig.update_layout(title="Last 5 Games vs Current Line", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🚀 Smart Parlay Legs")
        parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj, p_std, selected_market, data, risk_pref, is_td_market)
        for leg in parlay_legs:
            st.info(f"🔹 **{leg['type']}**: {leg['label']}")
        
        st.write(f"**Weather Adjustment:** {weather_reason}")
