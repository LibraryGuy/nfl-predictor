import nfl_data_py as nfl
import pandas as pd

# 1. Pull Data
weekly_stats = nfl.import_weekly_data([2023])
schedule = nfl.import_schedules([2023])

# 2. Fix the "Opponent" logic
# In weekly_stats, the 'opponent_team' is who they played.
# In schedule, we need to check both home and away to find the weather/field.
# For simplicity, we'll merge on the team and week.
data = weekly_stats.merge(
    schedule[['season', 'week', 'home_team', 'away_team', 'temp', 'surface', 'roof']], 
    left_on=['season', 'week', 'recent_team'], 
    right_on=['season', 'week', 'home_team'], 
    how='left'
)

# 3. See the results!
print("Successfully merged! Here is a sample:")
print(data[['player_name', 'recent_team', 'temp', 'surface']].head())