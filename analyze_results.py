import torch
import pandas as pd
import numpy as np

# 1. Load the samples on your Mac
print("Loading model results...")
samples = torch.load('full_era_samples.pt', map_location='cpu')

# 2. Reconstruct the full timeline of Skill (z)
print("Restitching timeline (550 weeks)...")
z_keys = sorted([k for k in samples.keys() if k.startswith('z_')], 
                key=lambda x: int(x.split('_')[1]))

# Create matrix: [Samples, Kickers, Weeks]
z_tensor = torch.stack([samples[k] for k in z_keys], dim=2)
z_samples = z_tensor.numpy()

# 3. Load Kicker Names from your CSV
df_raw = pd.read_csv('nfl_kicks_1999_2024.csv')
id_to_name = df_raw.set_index('kicker_player_id')['kicker_player_name'].to_dict()

# 4. Calculate Career Statistics
z_mean = np.mean(z_samples, axis=(0, 2))  
z_std = np.std(z_samples, axis=(0, 2))

# 5. The "Certified Score" (Risk-Adjusted Skill)
# We use 1.65 to capture the 95% lower bound of their skill.
penalty_factor = 1.65 
certified_scores = z_mean - (penalty_factor * z_std)

# 6. Build the Leaderboard
unique_ids = sorted(df_raw['kicker_player_id'].unique())

leaderboard = pd.DataFrame({
    'Kicker': [id_to_name.get(kid, kid) for kid in unique_ids],
    'Latent Skill (z)': z_mean,
    'Confidence (σ)': z_std,
    'Certified Score': certified_scores
})

# --- ADDED: HUMAN-READABLE POWER RATING ---
# Map the Certified Score to a 0-100 scale
min_cs = leaderboard['Certified Score'].min()
max_cs = leaderboard['Certified Score'].max()
leaderboard['Power Rating'] = ((leaderboard['Certified Score'] - min_cs) / (max_cs - min_cs)) * 100

# 7. Print the Final Certified Leaderboard
# We sort by Certified Score (Highest/Least Negative first)
certified_leaderboard = leaderboard.sort_values('Certified Score', ascending=False)

print("\n🏆 THE FGOE PRO CERTIFIED LEADERBOARD (1999-2024) 🏆")
print("Ranking by: Reliability-Adjusted Latent Skill")
print("-" * 85)
# Showing the top 25
print(certified_leaderboard[['Kicker', 'Latent Skill (z)', 'Confidence (σ)', 'Certified Score', 'Power Rating']].head(25).to_string(index=False))

# 8. Save the full ranked list to CSV
certified_leaderboard.to_csv('fgoe_final_leaderboard.csv', index=False)
print("\n✅ Ranked leaderboard saved to: fgoe_final_leaderboard.csv")