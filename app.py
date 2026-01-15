import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
import nflreadpy as nfl  # Modern 2026 NFL data standard

# --- 1. DATA LOADING & COLUMN MAPPING ---
@st.cache_data(ttl=3600)
def load_nfl_data():
    try:
        # Load stats for 2024-2025
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # Mapping NFL standard columns to your existing logic
        column_map = {
            'player_display_name': 'player_name',
            'recent_team': 'team',
            'opponent_team': 'opponent',
            'passing_yards': 'points', # Mapping yards to the 'points' slot in your logic
            'passing_tds': 'tds'
        }
        df = df.rename(columns=column_map)
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# --- 2. CORE PROJECTION LOGIC ---
def get_refined_projection(p_df, stat_cat, weight, usage_boost, dvp):
    if p_df.empty: return 0.0, 0.0
    
    # Using your weighted average logic (Recent vs Season)
    season_avg = p_df[stat_cat].mean()
    last3_avg = p_df.tail(3)[stat_cat].mean()
    
    weighted_base = (last3_avg * weight) + (season_avg * (1 - weight))
    st_lambda = weighted_base * usage_boost * dvp
    
    return round(st_lambda, 2), round(season_avg, 2)

# --- 3. VISUALIZATION ---
def plot_poisson_chart(mu, line, cat):
    x = np.arange(0, max(mu * 2.5, line + 5))
    y = poisson.pmf(x, mu)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker_color='#636EFA', opacity=0.6))
    fig.add_vline(x=line - 0.5, line_dash="dash", line_color="#FF4B4B", annotation_text="Market Line")
    fig.update_layout(title=f"Distribution: {cat.upper()}", template="plotly_dark", height=300)
    return fig

# --- 4. MAIN INTERFACE ---
st.set_page_config(page_title="NFL Sharp Pro", layout="wide")
data = load_nfl_data()

if not data.empty:
    with st.sidebar:
        st.title("🏈 NFL Sharp Pro")
        
        # --- FIX FOR THE 'player_name' ERROR ---
        # We check if our renamed column exists, otherwise fall back to original
        name_col = 'player_name' if 'player_name' in data.columns else 'player_display_name'
        player_list = sorted(data[name_col].unique())
        
        selected_player = st.selectbox("Select Player", player_list)
        stat_choice = st.selectbox("Category", ["points", "rushing_yards", "receiving_yards"])
        
        recency_weight = st.slider("Recency Bias", 0.0, 1.0, 0.3)
        manual_boost = st.slider("Usage Boost", 1.0, 1.5, 1.0)

    # Filter data for selected player
    p_df = data[data[name_col] == selected_player]
    
    if not p_df.empty:
        # Calculate Projections
        proj, avg = get_refined_projection(p_df, stat_choice, recency_weight, manual_boost, 1.0)
        
        # UI Display
        st.title(f"Analysis: {selected_player}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Projection", proj)
        col2.metric("Season Average", avg)
        col3.metric("Last 3 Games", round(p_df[stat_choice].tail(3).mean(), 1))
        
        line = st.number_input("Enter Market Line", value=float(proj))
        win_prob = (1 - poisson.cdf(line - 0.5, proj)) * 100
        
        st.subheader(f"Win Probability: {round(win_prob, 1)}%")
        st.plotly_chart(plot_poisson_chart(proj, line, stat_choice), use_container_width=True)
    else:
        st.warning("No historical data found for this player in the 2024-2025 range.")
else:
    st.error("The dataset failed to load. Please check your `requirements.txt` for `nflreadpy`.")
