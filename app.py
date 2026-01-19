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

# Initialize weather defaults if not present
for key, val in {"w_temp_val": 70, "w_wind_val": 0, "w_precip_val": "None"}.items():
    if key not in st.session_state:
        st.session_state[key] = val

FORCE_OUTDOOR = [
    "Gillette Stadium", "Lumen Field", "Hard Rock Stadium", 
    "Acrisure Stadium", "GEHA Field at Arrowhead Stadium",
    "Highmark Stadium", "Lambeau Field", "Soldier Field",
    "MetLife Stadium", "Lincoln Financial Field", "Paycor Stadium",
    "Levi's Stadium", "Empower Field at Mile High", "Bank of America Stadium"
]

# --- 2. LOGIC FUNCTIONS ---
def get_matchup_context(df, opponent, p_pos, stat_col):
    opp_def_stats = df[(df['opponent'] == opponent) & (df['position'] == p_pos)]
    if opp_def_stats.empty:
        return 1.0, "Neutral"
    
    league_avg = df[df['position'] == p_pos][stat_col].mean()
    opp_avg = opp_def_stats[stat_col].mean()
    
    m_ratio = opp_avg / league_avg if league_avg > 0 else 1.0
    status = "Shutdown" if m_ratio < 0.85 else "Vulnerable" if m_ratio > 1.15 else "Neutral"
    return round(m_ratio, 2), status

def fetch_stadium_weather(lat, lon, kickoff_time):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto", "forecast_days": 7
        }
        res = requests.get(url, params=params, timeout=5).json()
        target_str = datetime.now().strftime(f"%Y-%m-%dT{kickoff_time.strftime('%H')}:00")
        times = res['hourly']['time']
        idx = times.index(target_str) if target_str in times else 0
        temp = res['hourly']['temperature_2m'][idx]
        wind = res['hourly']['wind_speed_10m'][idx]
        precip = "Rain" if res['hourly']['precipitation'][idx] > 0.1 else "None"
        return temp, wind, precip
    except:
        return 70, 0, "None"

def get_weather_multiplier(roof, wind, temp, precip, pos):
    if any(x in roof.lower() for x in ['dome', 'closed', 'indoor']):
        return 1.0, "Dome (Neutral)"
    mult, reasons = 1.0, []
    if wind >= 15:
        p = 0.05 if wind < 20 else 0.12
        if pos in ['QB', 'WR', 'TE']: 
            mult -= p
            reasons.append(f"Wind (-{int(p*100)}%)")
    if precip != "None":
        mult -= 0.05
        reasons.append(f"{precip} (-5%)")
    return round(mult, 2), (" + ".join(reasons) if reasons else "Clear")

def run_usage_monte_carlo(avg_vol, avg_eff, eff_std, matchup_mult, is_td, iterations=10000):
    if is_td:
        # Corrected TD Logic: Discrete Poisson events
        td_lambda = max(0.01, avg_eff * matchup_mult)
        return np.random.poisson(td_lambda, iterations)
    else:
        # Yardage: Continuous Lognormal distribution
        sim_vol = np.random.poisson(max(avg_vol, 1), iterations)
        adj_eff = avg_eff * matchup_mult
        mu = np.log(max(adj_eff, 0.01)) - (0.4**2 / 2)
        sim_eff = np.random.lognormal(mu, 0.4, iterations)
        return sim_vol * sim_eff

def generate_prob_ladder(sims, is_td):
    thresholds = [0.5, 1.5, 2.5] if is_td else [(np.mean(sims) // 25 * 25) + (i * 25) for i in range(-1, 5)]
    ladder = []
    for t in thresholds:
        if t < 0: continue
        prob = (np.sum(sims >= t) / len(sims)) * 100
        odds = "N/A"
        if 0.1 < prob < 99.9:
            odds = f"+{int(100/(prob/100)-100)}" if prob <= 50 else f"{int(-(prob/(1-prob/100)))}"
        label = "Anytime TD" if is_td and t == 0.5 else f"{int(t)}+"
        ladder.append({"Threshold": label, "Probability": f"{prob:.1f}%", "Odds": odds})
    return pd.DataFrame(ladder)

# --- 3. DATA LOADING & ROBUST COLUMN FIXING ---
@st.cache_data(ttl=3600)
def load_and_fix_data():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Robust renaming map for consistency
        rename_map = {
            'player_display_name': 'player_name',
            'player': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'player_name' not in df.columns:
            st.error("Could not locate player names. Check data source.")
            return pd.DataFrame()

        # Clean numeric stats
        stats = ['passing_yards', 'rushing_yards', 'receiving_yards', 'attempts', 'carries', 'targets', 
                 'passing_tds', 'rushing_tds', 'receiving_tds']
        for s in stats:
            if s in df.columns:
                df[s] = pd.to_numeric(df[s], errors='coerce').fillna(0)
        
        return df.dropna(subset=['player_name'])
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

data = load_and_fix_data()
stadiums = NFLStadiums()

# --- 4. DASHBOARD UI ---
if not data.empty:
    with st.sidebar:
        st.title("🏈 NFL Sharp: Intel")
        # Fixing the selectbox crash with robust name column access
        player_list = sorted(data['player_name'].unique())
        selected_p = st.selectbox("Select Player", player_list)
        
        p_df = data[data['player_name'] == selected_p].copy()
        p_pos = p_df['position'].iloc[-1] if 'position' in p_df.columns else "WR"
        
        opponents = sorted(data['opponent'].unique())
        selected_opp = st.selectbox("Opponent Defense", opponents)
        
        market = st.radio("Market", ["Yards", "Touchdowns"])
        is_td = market == "Touchdowns"
        line = st.number_input("Sportsbook Line", value=0.5 if is_td else 50.0, step=0.5)
        
        venue = st.selectbox("Venue", sorted(stadiums.get_list_of_stadium_names()))
        k_time = st.time_input("Kickoff Time", time(13, 0))
        
        # Weather Logic
        s_obj = stadiums.get_stadium_by_name(venue)
        roof = str(s_obj.get('roof_type', 'Outdoor'))
        if any(x in roof.lower() for x in ['dome', 'closed', 'indoor']) and venue not in FORCE_OUTDOOR:
            w_temp, w_wind, w_prec = 70, 0, "None"
        else:
            w_temp, w_wind, w_prec = fetch_stadium_weather(s_obj.get('latitude'), s_obj.get('longitude'), k_time)

    # --- STAT CALCULATIONS ---
    if not is_td:
        stat_col = 'passing_yards' if p_pos == 'QB' else ('rushing_yards' if p_pos == 'RB' else 'receiving_yards')
        vol_col = 'attempts' if p_pos == 'QB' else ('carries' if p_pos == 'RB' else 'targets')
        avg_v, avg_e = p_df[vol_col].mean(), (p_df[stat_col] / p_df[vol_col].replace(0, np.nan)).mean()
    else:
        stat_col = 'passing_tds' if p_pos == 'QB' else ('rushing_tds' if p_pos == 'RB' else 'receiving_tds')
        avg_v, avg_e = 1.0, p_df[stat_col].mean()

    w_mult, w_text = get_weather_multiplier(roof, w_wind, w_temp, w_prec, p_pos)
    m_ratio, m_status = get_matchup_context(data, selected_opp, p_pos, stat_col)
    
    sims = run_usage_monte_carlo(avg_v * w_mult, avg_e, 0.4, m_ratio, is_td)
    win_p = (np.sum(sims >= line) / 10000) * 100

    # UI Render
    st.title(f"📊 {selected_p} ({p_pos})")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("Projection", f"{round(np.mean(sims), 2)} {market}", f"Win Prob: {round(win_p, 1)}%")
        fig = go.Figure(go.Histogram(x=sims, nbinsx=15 if is_td else 40, marker_color='#00ff96'))
        fig.add_vline(x=line, line_dash="dash", line_color="red")
        fig.update_layout(title="Probability Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("📈 Ladder")
        st.table(generate_prob_ladder(sims, is_td))
        st.write(f"🌡️ {w_text} | 🛡️ {m_status} ({m_ratio}x)")
else:
    st.error("No data available. Check your connection to nflverse.")
