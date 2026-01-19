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

# --- 3. REFACTORED MONTE CARLO ENGINE ---
def run_usage_monte_carlo(avg_volume, avg_efficiency, efficiency_std, matchup_mult, is_td, iterations=10000):
    """
    FIXED: Distinguishes between Yardage (Volume * Efficiency) and TDs (Poisson Events).
    """
    if avg_volume <= 0 and not is_td: return np.zeros(iterations)
    
    if is_td:
        # For TDs, avg_efficiency is actually 'Average TDs per Game'
        # Matchup multiplier adjusts the expected frequency (lambda)
        td_lambda = max(0.01, avg_efficiency * matchup_mult)
        return np.random.poisson(td_lambda, iterations)
    else:
        # For Yards, simulate volume (Poisson) and efficiency (Lognormal)
        sim_volume = np.random.poisson(avg_volume, iterations)
        adj_eff = avg_efficiency * matchup_mult
        # Use lognormal to model 'Big Play' potential in yardage
        sigma = 0.4 
        mu = np.log(max(adj_eff, 0.01)) - (sigma**2 / 2)
        sim_efficiency = np.random.lognormal(mu, sigma, iterations)
        return sim_volume * sim_efficiency

# --- 4. ODDS & LADDER LOGIC ---
def generate_prob_ladder(sim_results, is_td):
    if is_td:
        thresholds = [0.5, 1.5, 2.5]
        unit = "TDs"
    else:
        mean_val = np.mean(sim_results)
        step = 25 if mean_val < 150 else 50
        start = max(0, (mean_val // step) * step - step)
        thresholds = [start + (i * step) for i in range(6)]
        unit = "Yards"

    ladder_data = []
    for t in thresholds:
        prob = (np.sum(sim_results >= t) / len(sim_results)) * 100
        if 0 < prob < 100:
            safe_prob = max(min(prob, 99.9), 0.1)
            if safe_prob <= 50:
                odds = int(100 / (safe_prob / 100) - 100)
                odds_str = f"+{odds}"
            else:
                odds = int(-(safe_prob / (1 - safe_prob / 100)))
                odds_str = f"{odds}"
        else:
            odds_str = "N/A"
            
        label = f"Anytime" if is_td and t == 0.5 else f"{int(t)}+" if not is_td else f"{int(t+0.5)}+"
        ladder_data.append({
            f"Threshold ({unit})": label,
            "Hit Probability": f"{prob:.1f}%",
            "Implied Odds": odds_str
        })
    return pd.DataFrame(ladder_data)

# --- 5. DATA LOADING ---
@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        # Standardization of column names for different nflreadpy versions
        df = df.rename(columns={'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'})
        required = ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for col in required:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df.dropna(subset=['player_name'])
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

raw_data = load_data_pro()
stadium_client = NFLStadiums()

# --- 6. UI & DASHBOARD ---
if not raw_data.empty:
    with st.sidebar:
        st.title("🏈 NFL Sharp: Intel")
        selected_p = st.selectbox("Select Player", sorted(raw_data['player_name'].unique()))
        
        p_df = raw_data[raw_data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if 'position' in p_df.columns else "WR"
        p_team = p_df['team'].iloc[-1] if 'team' in p_df.columns else "N/A"
        
        opponents = sorted(raw_data['opponent'].unique())
        selected_opp = st.selectbox("Opponent Defense", opponents)

        selected_market = st.radio("Market Type", ["Yards", "Touchdowns"])
        is_td_market = selected_market == "Touchdowns"
        
        default_line = 0.5 if is_td_market else 50.0
        market_line = st.number_input("Sportsbook Line", value=default_line, step=0.5)
        
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

    # --- UPDATED CALCULATION LOGIC ---
    if not is_td_market:
        stat_col = 'passing_yards' if p_pos == 'QB' else ('rushing_yards' if p_pos == 'RB' else 'receiving_yards')
        vol_col = 'attempts' if p_pos == 'QB' else ('carries' if p_pos == 'RB' else 'targets')
        
        avg_vol = p_df[vol_col].mean()
        efficiency_series = p_df[stat_col] / p_df[vol_col].replace(0, np.nan)
        avg_eff = efficiency_series.dropna().mean()
        eff_std = efficiency_series.dropna().std() or 0.2
    else:
        # FIXED: TD Logic uses average TDs per game as the 'efficiency' and 1.0 as volume
        stat_col = 'passing_tds' if p_pos == 'QB' else ('rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        avg_vol = 1.0
        avg_eff = p_df[stat_col].mean()
        eff_std = 0

    w_mult, w_reason = get_weather_multiplier("Indoor" if is_actually_indoor else "Outdoor", w_wind, w_temp, w_precip, p_pos)
    m_ratio, m_status = get_matchup_context(raw_data, selected_opp, p_pos, stat_col)
    
    # Run Simulation
    sim_results = run_usage_monte_carlo(avg_vol * w_mult, avg_eff, eff_std, m_ratio, is_td_market)
    sim_mean = np.mean(sim_results)
    win_prob = (np.sum(sim_results >= market_line) / 10000) * 100

    # --- DASHBOARD RENDERING ---
    st.title(f"📊 {selected_p} Intelligence Hub ({p_pos})")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.metric("Model Projection", f"{round(sim_mean, 2)} {selected_market}", f"Win Prob: {round(win_prob, 1)}%")
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=sim_results, nbinsx=10 if is_td_market else 50, marker_color='#00ff96', opacity=0.75))
        fig_dist.add_vline(x=market_line, line_dash="dash", line_color="#ff4b4b", annotation_text="Line")
        fig_dist.update_layout(title=f"Usage-Based {selected_market} Distribution", template="plotly_dark", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("📈 Probability Ladder")
        ladder_df = generate_prob_ladder(sim_results, is_td_market)
        st.table(ladder_df) 

        st.divider()
        st.write(f"🌡️ **Weather**: {w_reason} ({w_temp}°F, {w_wind}mph)")
        st.write(f"🛡️ **Matchup**: {m_status} vs {selected_opp} ({m_ratio}x)")
        
        # Risk Insight
        if is_td_market and win_prob < 30:
            st.error("🚨 High Risk: Model shows low TD conversion probability.")
        elif win_prob > 65:
            st.success("✅ Strong Value: Model projects a high hit rate.")

else:
    st.warning("⚠️ Data Initialization Error. Please check data source connection.")
