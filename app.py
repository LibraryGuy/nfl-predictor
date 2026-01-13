import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, poisson  # Added poisson
from nfl_stadiums import NFLStadiums

# --- CORE LOGIC UPDATES ---

def get_poisson_probability(projected_mean, target_value):
    """
    Calculates the probability of hitting AT LEAST target_value.
    Professional models use 1 - CDF(target - 1) for 'Over' bets.
    """
    if projected_mean <= 0: return 0.0
    # Probability of getting target_value or more
    prob_over = 1 - poisson.cdf(target_value - 1, projected_mean)
    return round(prob_over * 100, 2)

def generate_risk_parlay(selected_p, p_pos, p_team, p_mean, p_std, stat_label, data, risk_level, is_td=False):
    risk_map = {
        "Conservative (-104)": {"offset": -0.6, "label": "Floor"},
        "Standard (+105)": {"offset": 0.0, "label": "Mean"},
        "Aggressive (+200)": {"offset": 0.6, "label": "Ceiling"}
    }
    
    offset = risk_map[risk_level]["offset"]
    
    # Logic Switch: Use Poisson for TDs, Normal for Yards
    if is_td:
        # For TDs, we use the mean to find a likely integer target (0.5, 1.5, etc)
        # Professional standard: TDs are almost always 0.5 or 1.5 lines
        primary_val = 0.5 if risk_level != "Aggressive (+200)" else 1.5
    else:
        primary_val = max(0, round(p_mean + (offset * p_std)))

    parlay_legs = [{"label": f"{selected_p}: {primary_val}+ {stat_label}", "type": risk_map[risk_level]["label"]}]
    
    # --- TEAMMATE SELECTION (KEEPING YOUR CURRENT LOGIC) ---
    teammates = data[(data['team'] == p_team) & (data['player_name'] != selected_p)].sort_values(['season', 'week'], ascending=False)
    
    if not teammates.empty:
        latest_season = teammates['season'].max()
        if p_pos == 'QB':
            seasonal_teammates = teammates[teammates['season'] == latest_season]
            valid_targets = seasonal_teammates[seasonal_teammates['position'].isin(['WR', 'TE'])]
            if not valid_targets.empty:
                top_target = valid_targets.groupby('player_name')['receiving_yards'].sum().idxmax()
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

# --- UI LOGIC MODIFICATION ---
# (Inside your UI rendering where you calculate targets)

# ... [keep your data loading and sidebar code same] ...

if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
    p_df = data[data['player_name'] == selected_p].copy()
    if not p_df.empty:
        # ... [keep stat_map and weather_multiplier code same] ...
        
        # Calculate Base Projections
        p_mean = p_df[stat_col].mean()
        p_std = p_df[stat_col].std() if len(p_df) > 1 else 1.0
        td_mean = p_df[td_col].mean() # New TD mean for Poisson
        
        # Apply multipliers to both
        model_proj_yards = p_mean * script_boost * sos_multiplier * weather_multiplier
        model_proj_tds = td_mean * script_boost * sos_multiplier # Weather has less impact on TD efficiency than volume
        
        # Calculate Probability for UI
        if market_line > 0:
            if "td" in stat_col:
                prob = get_poisson_probability(model_proj_tds, market_line)
            else:
                prob = round((1 - norm.cdf(market_line, model_proj_yards, p_std)) * 100, 2)
            st.metric("Win Probability", f"{prob}%")

        # Update the Parlay Generation Call
        # If the user is looking at a TD stat, tell the function to use Poisson
        is_td_stat = "td" in stat_col
        parlay_legs = generate_risk_parlay(selected_p, p_pos, p_team, model_proj_yards, p_std, stat_label, data, risk_pref, is_td=is_td_stat)

