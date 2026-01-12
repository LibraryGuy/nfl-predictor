import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import requests
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & DATA LOAD ---
st.set_page_config(page_title="NFL Sharp: Ultimate Genius", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_data_safe():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df['total_yds'] = df.get('rushing_yards', 0).fillna(0) + df.get('receiving_yards', 0).fillna(0)
        
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

data = load_data_safe()
stadium_client = NFLStadiums()

# --- 2. DEFENSE ENGINE ---
def get_defense_metrics(df):
    benchmarks = df.groupby('position')['total_yds'].mean().to_dict()
    def_stats = df.groupby(['opponent', 'position'])['total_yds'].mean().reset_index()
    return def_stats, benchmarks

# --- 3. UI LAYOUT ---
if not data.empty:
    def_data, league_benchmarks = get_defense_metrics(data)
    
    st.title("🏈 NFL Genius: Matchup & Weather Pro")
    
    # Sidebar: Player, Defense, and Stadium Selectors
    st.sidebar.header("Matchup Configuration")
    players = sorted(data['player_name'].unique())
    selected_p = st.sidebar.selectbox("1. Select Player", players)
    
    opponents = sorted(data['opponent'].unique())
    selected_opp = st.sidebar.selectbox("2. Select Opponent Defense", opponents)
    
    # Stadium/Weather Selector (Back in the sidebar)
    st.sidebar.divider()
    st.sidebar.header("🏟️ Game Environment")
    all_stads = sorted(stadium_client.get_list_of_stadium_names())
    sel_stad = st.sidebar.selectbox("Game Venue", all_stads)
    
    # Fetch Weather Data
    stad_info = stadium_client.get_stadium_by_name(sel_stad)
    lat, lon = stad_info.get('Latitude', 40.0), stad_info.get('Longitude', -75.0)
    try:
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        w_res = requests.get(w_url).json()
        c_temp = (w_res['current']['temperature_2m'] * 1.8) + 32
        c_wind = w_res['current']['wind_speed_10m'] * 0.621
    except:
        c_temp, c_wind = 50.0, 5.0 # Fallback
    
    st.sidebar.info(f"📍 {sel_stad}\n🌡️ {c_temp:.1f}°F | 💨 {c_wind:.1f} MPH")

    # --- 4. CALCULATION LOGIC ---
    p_df = data[data['player_name'] == selected_p]
    p_pos = p_df['position'].iloc[-1]
    p_avg = p_df['total_yds'].mean()
    
    # Defense Multiplier
    opp_match = def_data[(def_data['opponent'] == selected_opp) & (def_data['position'] == p_pos)]
    bench = league_benchmarks.get(p_pos, 1.0)
    def_mod = (opp_match['total_yds'].iloc[0] / bench) if not opp_match.empty else 1.0
    
    # Weather Multiplier (Simple Logic: High winds hurt passing/receiving)
    weather_mod = 1.0
    if c_wind > 15 and p_pos in ['QB', 'WR', 'TE']:
        weather_mod = 0.92  # 8% reduction for high wind
    
    # Final Projection
    proj_yds = p_avg * def_mod * weather_mod

    # --- 5. DASHBOARD ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player Avg", f"{p_avg:.1f} Yds")
    c2.metric("Defense Mod", f"{def_mod:.2f}x")
    c3.metric("Weather Mod", f"{weather_mod:.2f}x")
    c4.metric("Final Projection", f"{proj_yds:.1f} Yds", delta=f"{proj_yds - p_avg:+.1f}")

    st.plotly_chart(px.line(p_df, x='week', y='total_yds', title=f"{selected_p} Career Momentum"), use_container_width=True)

else:
    st.error("Check data source - column mapping failed.")
