import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson
from nfl_stadiums import NFLStadiums

# ... [Keep your get_weather_multiplier and get_dynamic_sos functions here] ...

def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td=False):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    offset = risk_map[risk_level]["offset"]
    
    # LOGIC: Poisson for TDs, Normal for Yards
    if is_td:
        # Most "Anytime TD" lines are 0.5. Aggressive targets 2+ (1.5 line)
        primary_val = 0.5 if risk_level != "Aggressive (+200)" else 1.5
    else:
        primary_val = max(0, round(p_mean + (offset * p_std)))
    
    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    # Teammate Selection Logic (Unchanged)
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)].sort_values(['season', 'week'], ascending=False)
    if not teammates.empty:
        latest_season = teammates['season'].max()
        if p_pos == 'QB':
            seasonal_targets = teammates[(teammates['season'] == latest_season) & (teammates['position'].isin(['WR', 'TE']))]
            if not seasonal_targets.empty:
                top_target = seasonal_targets.groupby('player_name')['receiving_yards'].sum().idxmax()
                leg_val = 40 if risk_level == "Conservative (-104)" else 60
                parlay_legs.append({"label": f"{top_target}: {leg_val}+ Rec Yds", "type": "Teammate Stack"})
        elif p_pos in ['WR', 'TE', 'RB']:
            current_qbs = teammates[(teammates['position'] == 'QB') & (teammates['season'] == latest_season)]
            if not current_qbs.empty:
                last_week = current_qbs['week'].max()
                qb_name = current_qbs[current_qbs['week'] == last_week]['player_name'].iloc[0]
                leg_val = 195 if risk_level == "Conservative (-104)" else 240 if p_pos == 'RB' else 215
                parlay_legs.append({"label": f"{qb_name}: {leg_val}+ Pass Yds", "type": "QB Link"})
    return parlay_legs

# --- DATA LOADING ---
data, schedules = load_data_pro()

# FIX: Explicit check to ensure data is a DataFrame
if isinstance(data, pd.DataFrame) and not data.empty:
    # ... [Rest of your UI Rendering] ...
    
    # When calculating win probability:
    is_td_market = "tds" in stat_col
    if market_line > 0:
        if is_td_market:
            # Poisson Prob: 1 - P(X < market_line)
            prob = (1 - poisson.cdf(market_line - 1, model_proj)) * 100
        else:
            # Normal Prob
            prob = (1 - norm.cdf(market_line, model_proj, p_std)) * 100
        st.metric("Model Win Probability", f"{round(prob, 1)}%")
