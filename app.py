import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, time
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & API CONFIG ---
st.set_page_config(page_title="NFL Sharp: Intelligence Hub", layout="wide", page_icon="🏈")

# --- INITIALIZE SESSION STATE ---
if "w_temp_val" not in st.session_state:
    st.session_state["w_temp_val"] = 70
if "w_wind_val" not in st.session_state:
    st.session_state["w_wind_val"] = 0
if "w_precip_val" not in st.session_state:
    st.session_state["w_precip_val"] = "None"
if "last_stadium_query" not in st.session_state:
    st.session_state["last_stadium_query"] = ""

FORCE_OUTDOOR = [
    "Gillette Stadium", "Lumen Field", "Hard Rock Stadium", 
    "Acrisure Stadium", "GEHA Field at Arrowhead Stadium",
    "Highmark Stadium", "Lambeau Field", "Soldier Field",
    "MetLife Stadium", "Lincoln Financial Field", "Paycor Stadium",
    "Levi's Stadium", "Empower Field at Mile High", "Bank of America Stadium"
]

# --- 2. INTELLIGENT MATCHUP LOGIC ---
def get_matchup_context(data, opponent, p_pos, stat_col):
    opp_def_stats = data[(data['opponent'] == opponent) & (data['position'] == p_pos)]
    if opp_def_stats.empty:
        return 1.0, "Neutral"
    
    league_avg = data[data['position'] == p_pos][stat_col].mean()
    opp_avg = opp_def_stats[stat_col].mean()
    
    m_ratio = opp_avg / league_avg if league_avg > 0 else 1.0
    status = "Neutral"
    if m_ratio < 0.85: status = "Shutdown"
    elif m_ratio > 1.15: status = "Vulnerable"
    
    return round(m_ratio, 2), status

# --- AUTOMATED WEATHER FETCH ---
def fetch_stadium_weather(lat, lon, game_time):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto", "forecast_days": 7
        }
        response = requests.get(url, params=params, timeout=5).json()
        hourly = response.get('hourly', {})
        times = hourly.get('time', [])
        target_str = datetime.now().strftime(f"%Y-%m-%dT{game_time.strftime('%H')}:00")
        idx = times.index(target_str) if target_str in times else 0
        temp = hourly.get('temperature_2m', [70])[idx]
        wind = hourly.get('wind_speed_10m', [0])[idx]
        precip_val = hourly.get('precipitation', [0])[idx]
        precip_type = "None"
        if precip_val > 0.1: precip_type = "Rain"
        if temp < 32 and precip_val > 0: precip_type = "Snow"
        return temp, wind, precip_type
    except Exception:
        return 70, 0, "None"

def get_weather_multiplier(roof_type, wind, temp, precip, p_pos):
    if any(x in roof_type.lower() for x in ['dome', 'closed', 'indoor']):
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

# --- 3. ADVANCED USAGE-BASED MONTE CARLO ENGINE ---
def run_usage_monte_carlo(avg_volume, avg_efficiency, efficiency_std, matchup_mult, is_td, iterations=10000):
    """
    Simulates outcomes by decoupling Volume (targets/carries) from Efficiency (yards per).
    """
    if avg_volume <= 0: return np.zeros(iterations)
    
    if is_td:
        # For TDs, we still use Poisson based on the expected mean
        return np.random.poisson(avg_volume * matchup_mult, iterations)
    else:
        # 1. Simulate Volume (Poisson distribution for discrete attempts/targets)
        sim_volume = np.random.poisson(avg_volume, iterations)
        
        # 2. Simulate Efficiency (Log-normal distribution for yards per attempt)
        # We apply the matchup multiplier to the efficiency side
        adj_eff = avg_efficiency * matchup_mult
        # Use a log-normal to ensure efficiency doesn't drop below 0 and has a long tail
        sigma = 0.4 # Variance in efficiency
        mu = np.log(adj_eff) - (sigma**2 / 2)
        sim_efficiency = np.random.lognormal(mu, sigma, iterations)
        
        return sim_volume * sim_efficiency

# --- PARLAY GENERATION ---
def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td, opponent):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    offset = risk_map[risk_level]["offset"]
    stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if not is_td else ('passing_tds' if p_pos == 'QB' else 'receiving_tds')
    m_ratio, m_status = get_matchup_context(data, opponent, p_pos, stat_col)
    
    if is_td:
        primary_val = 0.5 if risk_level != "Aggressive (+200)" else 1.5
    else:
        primary_val = max(0, round(p_mean + (offset * p_std)))
    
    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"], "color": "info"}]
    
    if is_td:
        prob_zero = poisson.pmf(0, p_mean)
        market_name = "Passing TD" if p_pos == "QB" else "Anytime TD"
        if prob_zero > 0.58 or m_status == "Shutdown":
            parlay_legs.append({"label": f"AVOID: {selected_p} {market_name}", "type": "Risk Alert", "color": "error"})
    
    return parlay_legs

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if 'player_name' not in df.columns and 'player' in df.columns:
            df = df.rename(columns={'player': 'player_name'})
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets', 'passing_tds', 'rushing_tds', 'receiving_tds']:
            df[col] = df.get(col, 0).fillna(0)
        return df.dropna(subset=['player_name'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

raw_data = load_data_pro()
data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
stadium_client = NFLStadiums()

# --- UI ---
if not data.empty and 'player_name' in data.columns:
    with st.sidebar:
        st.header("🎯 Target Selection")
        selected_p = st.selectbox("Select Player", sorted(data['player_name'].unique()))
        selected_opp = st.selectbox("Opponent Defense", sorted(data['opponent'].unique()))
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1]
        p_team = p_df['team'].iloc[-1]

        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        is_td_market = selected_market == "Touchdowns"
        market_line = st.number_input("Sportsbook Line", value=0.5 if is_td_market else 50.0, step=0.5)
        risk_pref = st.radio("Target Odds Profile", ["Conservative (-104)", "Standard (+105)", "Aggressive (+200)"], index=1)
        
        # Weather Logic
        sel_stad_name = st.selectbox("Game Venue", sorted(stadium_client.get_list_of_stadium_names()))
        game_time = st.time_input("Kickoff Time", time(13, 0))
        stad_obj = stadium_client.get_stadium_by_name(sel_stad_name)
        roof_type = str(stad_obj.get('roof_type', 'Outdoor'))
        is_forced_outdoor = any(s.lower() in sel_stad_name.lower() for s in FORCE_OUTDOOR)
        is_actually_indoor = any(x in roof_type.lower() for x in ['dome', 'closed', 'indoor']) and not is_forced_outdoor

        if not is_actually_indoor:
            lat, lon = (stad_obj.get('latitude'), stad_obj.get('longitude'))
            w_temp, w_wind, w_precip = fetch_stadium_weather(lat, lon, game_time)
        else:
            w_temp, w_wind, w_precip = 70, 0, "None"

    # --- ENHANCED CALCULATION LOGIC ---
    stat_col = ('passing_yards' if p_pos == 'QB' else 'rushing_yards' if p_pos == 'RB' else 'receiving_yards') if not is_td_market else ('passing_tds' if p_pos == 'QB' else 'rushing_tds' if p_pos == 'RB' else 'receiving_tds')
    volume_col = 'attempts' if p_pos == 'QB' else 'carries' if p_pos == 'RB' else 'targets'
    
    avg_vol = p_df[volume_col].mean()
    
    if not is_td_market:
        # Calculate Efficiency (Yards per target/carry/pass)
        # We handle division by zero by using a small epsilon
        efficiency_series = p_df[stat_col] / p_df[volume_col].replace(0, np.nan)
        avg_eff = efficiency_series.dropna().mean() if not efficiency_series.dropna().empty else 0
        eff_std = efficiency_series.dropna().std() if len(efficiency_series.dropna()) > 1 else 0.2
    else:
        avg_eff = 1.0 # Not used for TDs
        eff_std = 0.0

    w_mult, w_reason = get_weather_multiplier("Indoor" if is_actually_indoor else "Outdoor", w_wind, w_temp, w_precip, p_pos)
    m_ratio, m_status = get_matchup_context(data, selected_opp, p_pos, stat_col)
    
    # Run the Usage-Based Simulation
    # We apply weather to volume and matchup to efficiency
    sim_results = run_usage_monte_carlo(avg_vol * w_mult, avg_eff, eff_std, m_ratio, is_td_market)
    
    sim_mean = np.mean(sim_results)
    win_prob = (np.sum(sim_results >= market_line) / 10000) * 100

    # --- DASHBOARD ---
    st.title(f"📊 {selected_p} Intelligence Hub")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.metric("Model Projection", f"{round(sim_mean, 1)} {selected_market}", f"Win Prob: {round(win_prob, 1)}%")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=sim_results, nbinsx=50, marker_color='#00ff96', opacity=0.75))
        fig_dist.add_vline(x=market_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Line")
        fig_dist.update_layout(title="Usage-Based Outcome Distribution", template="plotly_dark", height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("🚀 Smart Parlay Legs")
        parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, sim_mean, np.std(sim_results), selected_market, data, risk_pref, is_td_market, selected_opp)
        for leg in parlay_legs:
            st.info(f"🔹 **{leg['type']}**: {leg['label']}")
        st.divider()
        st.write(f"**Weather**: {w_reason}")
        st.write(f"**Matchup**: {m_status} ({m_ratio}x)")
else:
    st.warning("⚠️ Data Initialization Error.")
