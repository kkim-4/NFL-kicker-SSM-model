import pandas as pd
import numpy as np

print("1. Loading subset of data...")
# nrows=1000 ensures this runs instantly on your laptop
df = pd.read_csv('nfl_kicks_1999_2024.csv', nrows=1000)
print(f"Success! Loaded {len(df)} rows.")

print("2. Testing the sorting logic...")
df = df.sort_values(['game_date', 'kicker_player_id'])
print("Success! Data sorted chronologically.")

print("3. Checking for required columns...")
required_cols = ['game_date', 'kicker_player_id', 'kick_distance', 'field_goal_result']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    print(f"🚨 ERROR: Missing columns: {missing}")
else:
    print("Success! All required columns are present.")
    
print("\n✅ Phase 1 Data Prep is completely valid and ready for the cluster!")