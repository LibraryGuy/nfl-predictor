import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
import requests
from nfl_stadiums import NFLStadiums

# --- 1. SETTINGS & DATA LOAD ---
st.set_page_config(page_title="NFL Sharp: Position Pro", layout="wide", page_icon="🏈")

@st.cache_data(ttl=3600)
def load_data_pro():
    try:
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, col))).strip() for col in df.columns.values]
        
        rename_map = {'player_display_name': 'player_name', 'recent_team': 'team', 'opponent_team': 'opponent'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        # Fill missing stats with 0 so math doesn't break
        cols_to_fix = ['passing_yards', 'rushing_yards', 'receiving_yards']
        for col in cols_to_fix:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].fillna(0)
            
        return df.dropna(subset=['player_name', 'opponent', 'position'])
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

data = load_data_pro()
stadium_client = NFLStadiums()

# --- 2. DYNAMIC DEFENSE ENGINE ---
def get_dvp_metrics(df, yard_type):
    """Calculates Defense vs Position for a specific stat type (passing/rushing/etc)"""
    benchmarks = df.groupby('position')[yard_type].mean().to_dict()
    def_stats = df.groupby(['opponent', 'position'])[yard_type].mean().reset_index()
    return def_stats, benchmarks

# --- 3. UI LOGIC ---
if not data.empty:
    st.title("🏈 NFL Genius: Position-Specific Predictor")
    
    # Selection Sidebar
    st.sidebar.header("Matchup Setup")
    selected_p = st.sidebar.selectbox("1. Select Player", sorted(data['player_name'].unique()))
    selected_opp = st.sidebar.selectbox("2. Select Opponent", sorted(data['opponent'].unique()))
    sel_stad = st.sidebar.selectbox("3. Venue", sorted(stadium_client.get_list_of_stadium_names()))

    # --- 4. POSITION-SPECIFIC LOGIC ---
    p_df = data[data['player_name'] == selected_p]
    p_pos = p_df['position'].iloc[-1]
    
    # Define which stats to track based on position
    if p_pos == 'QB':
        primary_stat = 'passing_yards'
        secondary_stat = 'rushing_yards'
        label = "Passing Yards"
    elif p_pos == 'RB':
        primary_stat = 'rushing_yards'
        secondary_stat = 'receiving_yards'
        label = "Rushing Yards"
    else: # WR or TE
        primary_stat = 'receiving_yards'
        secondary_stat = 'rushing_yards' # Jet sweeps etc
        label = "Receiving Yards"

    # Calculate Totals
    p_df['combined_yds'] = p_df[primary_stat] + p_df[secondary_stat]
    
    # Defensive Context for Primary Stat
    def_data, bench_data = get_dvp_metrics(data, primary_stat)
    opp_match = def_data[(def_data['opponent'] == selected_opp) & (def_data['position'] == p_pos)]
    bench = bench_data.get(p_pos, 1.0)
    def_mod = (opp_match[primary_stat].iloc[0] / bench) if not opp_match.empty else 1.0

    # --- 5. WEATHER LOGIC ---
    stad_info = stadium_client.get_stadium_by_name(sel_stad)
    try:
        w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={stad_info['Latitude']}&longitude={stad_info['Longitude']}&current=wind_speed_10m").json()
        wind = w_res['current']['wind_speed_10m'] * 0.621
    except: wind = 5.0
    
    weather_mod = 0.90 if (wind > 15 and p_pos in ['QB', 'WR', 'TE']) else 1.0

    # Final Projections
    proj_primary = p_df[primary_stat].mean() * def_mod * weather_mod
    proj_total = p_df['combined_yds'].mean() * def_mod * weather_mod

    # --- 6. DISPLAY DASHBOARD ---
    st.subheader(f"Analysis for {selected_p} ({p_pos})")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Projected {label}", f"{proj_primary:.1f}")
    m2.metric("Projected Total Yards", f"{proj_total:.1f}")
    m3.metric("Matchup Difficulty", f"{def_mod:.2f}x", delta="Favorable" if def_mod > 1 else "Tough")

    # Trend Chart
    fig = px.area(p_df, x='week', y=[primary_stat, 'combined_yds'], 
                  title=f"Yardage Trend: {selected_p}",
                  labels={'value': 'Yards', 'variable': 'Stat Type'},
                  color_discrete_sequence=['#00c8ff', '#0078ff'])
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Data Load Error")
