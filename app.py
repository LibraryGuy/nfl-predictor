import streamlit as st
import nflreadpy as nfl
import pandas as pd
import plotly.express as px
from math import exp

# --- 1. CONFIG ---
st.set_page_config(page_title="NFL Genius Fix", layout="wide")

st.title("🏈 NFL Genius: Pro Builder")

if "parlay_legs" not in st.session_state:
    st.session_state.parlay_legs = []

with st.sidebar:
    st.header("📋 Parlay Builder")
    for leg in st.session_state.parlay_legs:
        st.success(f"{leg['player']}: {leg['pick']}")
    if st.button("Reset Parlay"):
        st.session_state.parlay_legs = []
        st.rerun()

# --- 2. INPUT ---
p_input = st.text_input("Enter Player Name (e.g., Jefferson)", "Jefferson")

# --- 3. THE UPDATED DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_and_standardize_data():
    try:
        # Load the stats (Returns a Polars df, converted to Pandas)
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        
        # A. FLATTEN HEADERS (Prevents AttributeError)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
        
        # B. THE KEYERROR FIX: TRANSLATION LAYER
        # We map the raw nflverse names to the names your UI uses
        rename_map = {
            'player_display_name': 'player_name',
            'opponent_team': 'opponent',   # This fixes your KeyError!
            'recent_team': 'team',
            'rushing_tds': 'rush_td',
            'receiving_tds': 'rec_td'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        return df.fillna(0)
    except Exception as e:
        st.error(f"Sync Failure: {e}")
        return pd.DataFrame()

data = load_and_standardize_data()

# --- 4. UI LOGIC ---
if not data.empty and p_input:
    # Fuzzy search for the player
    matches = data[data['player_name'].str.contains(p_input, case=False, na=False)]
    
    if not matches.empty:
        selected_p = st.selectbox("Confirm Player", matches['player_name'].unique())
        p_df = data[data['player_name'] == selected_p]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # This line now works because 'opponent' exists thanks to the rename!
            opp_list = sorted(data['opponent'].unique()) 
            target_opp = st.selectbox("Select Opponent", opp_list)
            
            # Stats Display
            avg_yds = p_df['receiving_yards'].mean()
            st.metric(f"{selected_p} Avg Yards", f"{avg_yds:.1f}")
            
            if st.button("Add to Parlay"):
                st.session_state.parlay_legs.append({"player": selected_p, "pick": "Over Model"})
                st.rerun()
        
        with col2:
            st.plotly_chart(px.line(p_df, x='week', y='receiving_yards', title="Weekly Trend"))
    else:
        st.info("Searching for player data...")
