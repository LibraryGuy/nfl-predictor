import streamlit as st
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBRegressor
import plotly.express as px
import numpy as np

# Set page layout
st.set_page_config(page_title="NFL Predictor: Ultimate Strategist", layout="wide")
st.title("🏈 NFL Ultimate Predictive Dashboard")

@st.cache_data
def load_nfl_data_pro():
    years = [2024, 2025]
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    sched = nfl.load_schedules(seasons=years).to_pandas()
    
    # Standardize column names for the 2025-2026 season shift
    if 'team' in weekly.columns: weekly = weekly.rename(columns={'team': 'recent_team'})
    weekly = weekly.dropna(subset=['player_name'])
    
    # Metric Initialization
    metrics = ['passing_yards', 'rushing_yards', 'receiving_yards']
    for m in metrics: weekly[m] = weekly[m].fillna(0)
    weekly['total_scrimmage_yards'] = weekly['rushing_yards'] + weekly['receiving_yards']
    
    # Load Injuries with Fallback
    try:
        injuries = nfl.load_injuries(seasons=[2025]).to_pandas()
    except Exception:
        injuries = nfl.load_injuries(seasons=[2024]).to_pandas()
    if 'full_name' in injuries.columns: injuries = injuries.rename(columns={'full_name': 'player_name'})
    
    # Merge Schedule
    df = weekly.merge(
        sched[['season', 'week', 'home_team', 'temp', 'surface']], 
        left_on=['season', 'week', 'recent_team'], 
        right_on=['season', 'week', 'home_team'], 
        how='left'
    )
    df['temp'] = df['temp'].fillna(70)
    df['is_grass'] = df['surface'].apply(lambda x: 1 if str(x).lower() == 'grass' else 0)
    
    return df, injuries

data, injury_report = load_nfl_data_pro()

# --- SIDEBAR: GLOBAL FILTERS ---
st.sidebar.header("Global Simulation Settings")
curr_temp = st.sidebar.slider("Game Temperature", 0, 100, 65)
curr_surface = st.sidebar.radio("Field Type", ["Grass", "Turf"])
is_grass_val = 1 if curr_surface == "Grass" else 0

# --- DEFENSE MATCHUP FILTER ---
st.sidebar.divider()
st.sidebar.header("Value Scanner Filters")
# Calculate Defense Tiers (based on Total Yards allowed)
def_ranking = data.groupby('opponent_team')['total_scrimmage_yards'].mean().sort_values(ascending=False)
def_options = ["All Defenses", "Bottom 10 Defenses (Easiest)", "Top 10 Defenses (Hardest)"]
def_filter = st.sidebar.selectbox("Filter Scanner by Defense Strength", def_options)

def get_filtered_defenses(filter_choice):
    if filter_choice == "Bottom 10 Defenses (Easiest)":
        return def_ranking.head(10).index.tolist()
    elif filter_choice == "Top 10 Defenses (Hardest)":
        return def_ranking.tail(10).index.tolist()
    return data['opponent_team'].unique().tolist()

# --- VALUE SCANNER ENGINE ---
def get_best_value_matchups(df, temp, is_grass, allowed_defs):
    value_list = []
    # Only scan players with enough sample size
    active_players = df.groupby('player_name').filter(lambda x: len(x) >= 5)['player_name'].unique()
    
    for p_name in active_players[:60]: # Scan subset for performance
        p_df = df[df['player_name'] == p_name]
        pos = p_df['position'].iloc[0]
        metric = 'passing_yards' if pos == 'QB' else 'total_scrimmage_yards'
        
        # Mini-model for quick scanning
        X = p_df[['temp', 'is_grass']].fillna(0)
        y = p_df[metric]
        m = XGBRegressor(n_estimators=20).fit(X, y)
        
        pred = m.predict(pd.DataFrame([[temp, is_grass]], columns=['temp', 'is_grass']))[0]
        avg = p_df[metric].mean()
        
        value_list.append({'Player': p_name, 'Pos': pos, 'Pred': pred, 'Avg': avg, 'Edge': pred - avg})
    
    return pd.DataFrame(value_list).sort_values(by='Edge', ascending=False)

# --- DISPLAY TOP VALUES ---
st.subheader(f"🚀 Top Value Matchups ({def_filter})")
allowed_list = get_filtered_defenses(def_filter)
# Note: In a real app, you'd match the player's upcoming opponent here.
# For this scanner, we show players whose general 'Edge' is highest under these conditions.
best_values = get_best_value_matchups(data, curr_temp, is_grass_val, allowed_list)
val_cols = st.columns(3)
for i, row in enumerate(best_values.head(3).itertuples()):
    with val_cols[i]:
        st.success(f"**{row.Player}** ({row.Pos})")
        st.write(f"Model Pred: **{row.Pred:.1f}**")
        st.write(f"Edge vs Avg: **+{row.Edge:.1f}**")

# --- INDIVIDUAL PLAYER ANALYSIS (EXISTING FEATURES) ---
st.divider()
player_list = sorted(data['player_name'].dropna().unique())
selected_player = st.selectbox("Detailed Analysis & Custom Opponent Prediction", player_list, 
                             index=player_list.index("P.Mahomes") if "P.Mahomes" in player_list else 0)

p_info = data[data['player_name'] == selected_player].iloc[0]
p_pos = p_info['position']
target_stat = 'passing_yards' if p_pos == 'QB' else 'total_scrimmage_yards'

# Custom Opponent Prediction Section
st.write(f"### Custom Simulation for {selected_player}")
opp_list = sorted(data['opponent_team'].unique())
selected_opp = st.selectbox("Select Specific Opponent for Prediction", opp_list)

# Calculate Prediction with specific Opponent Difficulty
def_avg = data.groupby('opponent_team')[target_stat].mean().reset_index()
def_avg.columns = ['opponent_team', 'def_diff']
opp_diff_val = def_avg[def_avg['opponent_team'] == selected_opp]['def_diff'].iloc[0]

p_df = data[data['player_name'] == selected_player].copy()
p_df = p_df.merge(def_avg, on='opponent_team', how='left')

X_p = p_df[['temp', 'is_grass', 'def_diff']].fillna(0)
y_p = p_df[target_stat]
final_model = XGBRegressor(n_estimators=50).fit(X_p, y_p)
final_pred = final_model.predict(pd.DataFrame([[curr_temp, is_grass_val, opp_diff_val]], 
                                            columns=['temp', 'is_grass', 'def_diff']))[0]

# Display Results
res_c1, res_c2 = st.columns(2)
with res_c1:
    st.metric(f"Predicted {target_stat.replace('_',' ').title()}", f"{final_pred:.1f}")
    st.info(f"🎯 **Betting Leg:** Over {int(final_pred * 0.88)}.5")
with res_c2:
    # Injury Check
    p_injury = injury_report[injury_report['player_name'] == selected_player]
    if not p_injury.empty:
        latest = p_injury.iloc[0]
        st.warning(f"⚠️ {latest['report_status']} ({latest['practice_primary_injury']})")
    else:
        st.success("✅ No Injuries Reported")

st.plotly_chart(px.line(p_df, x='week', y=target_stat, title=f"Historical {target_stat.replace('_',' ').title()}"))