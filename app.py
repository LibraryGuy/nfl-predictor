import streamlit as st
import pandas as pd
import nflreadpy as nfl

st.set_page_config(page_title="NFL Predictor", layout="wide")

@st.cache_data(ttl=3600)
def load_data():
    try:
        # Load data for the most recent available seasons
        df = nfl.load_player_stats(seasons=[2024, 2025]).to_pandas()
        return df
    except Exception as e:
        st.error(f"Failed to fetch NFL data: {e}")
        return pd.DataFrame()

# 1. Load the data
data = load_data()

# 2. Check if data is empty first
if data.empty:
    st.warning("⚠️ No data found. Please check your internet connection or the nflverse API status.")
    st.stop() # Stops the script here so it doesn't hit line 63

# 3. Identify the correct Name Column dynamically
# NFL data usually uses 'player_display_name', but we'll check for both
if 'player_display_name' in data.columns:
    name_col = 'player_display_name'
elif 'player_name' in data.columns:
    name_col = 'player_name'
else:
    st.error(f"Critical Error: Could not find a player name column. Found columns: {list(data.columns)}")
    st.stop()

# 4. Now safely create the player list
player_list = sorted(data[name_col].dropna().unique())

# --- REST OF YOUR APP LOGIC ---
st.title("🏈 NFL Sharp Predictor")
selected_player = st.selectbox("Select Player", player_list)

if selected_player:
    player_stats = data[data[name_col] == selected_player]
    st.write(f"Displaying stats for **{selected_player}**")
    st.dataframe(player_stats.head())
