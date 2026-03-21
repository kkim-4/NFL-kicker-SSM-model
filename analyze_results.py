import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD DATA
print("📂 Loading data and CSV...")
samples = torch.load('full_era_samples.pt', map_location='cpu')
df_raw = pd.read_csv('nfl_kicks_1999_2024.csv')

# Standardize IDs and Names
kicker_ids = sorted(df_raw['kicker_player_id'].unique())
id_to_name = df_raw.set_index('kicker_player_id')['kicker_player_name'].to_dict()

# 2. DEFINE ACTIVE FILTERS
print("🔍 Identifying active kickers per era...")
# Who kicked in 2024?
active_2024 = set(df_raw[df_raw['season'] == 2024]['kicker_player_id'].unique())
# Who kicked between 2020 and 2024?
active_past_5 = set(df_raw[df_raw['season'] >= 2020]['kicker_player_id'].unique())

# 3. RECONSTRUCT THE TIMELINE
print("🧩 Reconstructing 550 weeks...")
z_keys = sorted([k for k in samples.keys() if k.startswith('z_')], 
                key=lambda x: int(x.split('_')[1]))
z_list = [samples[k].numpy() if isinstance(samples[k], torch.Tensor) else samples[k] for k in z_keys]
z_timeline = np.stack(z_list, axis=2) 

# 4. UPDATED GENERATOR WITH FILTERING
def generate_and_save(data_subset, filename, label, active_filter=None):
    mean_z = np.mean(data_subset, axis=(0, 2))
    std_z = np.std(data_subset, axis=(0, 2))
    cert = mean_z - (1.5 * std_z)
    
    lb = pd.DataFrame({
        'kicker_id': kicker_ids,
        'Kicker': [id_to_name.get(kid, kid) for kid in kicker_ids],
        'Latent Skill (z)': mean_z,
        'Confidence (σ)': std_z,
        'Certified Score': cert
    })
    
    # APPLY THE FILTER
    if active_filter:
        lb = lb[lb['kicker_id'].isin(active_filter)].copy()
    
    # Power Rating 0-100 (Re-calculated based only on active players)
    s_min, s_max = lb['Certified Score'].min(), lb['Certified Score'].max()
    lb['Power Rating'] = 100 * (lb['Certified Score'] - s_min) / (s_max - s_min)
    
    lb.sort_values('Certified Score', ascending=False).drop(columns=['kicker_id']).to_csv(filename, index=False)
    print(f"💾 Saved {label} to {filename} ({len(lb)} kickers)")

# --- FILE 1: CAREER ALL TIME (No filter - The Hall of Fame) ---
generate_and_save(z_timeline, 'career_all_time.csv', 'All-Time Career')

# --- FILE 2: PAST 5 YEARS (Active 2020-2024 only) ---
# We use the last ~105 weeks (approx 5 seasons)
generate_and_save(z_timeline[:, :, -105:], 'past_5_years_active.csv', 'Past 5 Years (Active)', active_filter=active_past_5)

# --- FILE 3: 2024 FINAL STATE (Active 2024 only) ---
generate_and_save(z_timeline[:, :, -1:], 'final_2024_rankings_active.csv', '2024 Final State (Active)', active_filter=active_2024)

# 5. THE COMPARISON PLOT
def plot_elite_trio(names):
    plt.figure(figsize=(14, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    name_to_id = {v: k for k, v in id_to_name.items()}
    
    for name in names:
        if name not in name_to_id: continue
        kid = name_to_id[name]
        k_idx = kicker_ids.index(kid)
        traj = np.mean(z_timeline[:, k_idx, :], axis=0)
        smoothed = pd.Series(traj).rolling(window=15, min_periods=1).mean()
        plt.plot(smoothed, label=name, linewidth=3)

    plt.title("Bayesian Evolution: The Reliable, The Volatile, and The GOAT", color='white', size=16, pad=20)
    plt.legend(facecolor='#222222', labelcolor='white', fontsize=12)
    plt.grid(color='#444444', linestyle='--', alpha=0.3)
    plt.tick_params(colors='white')
    plt.savefig('kicker_comparison.png', facecolor='#121212')
    plt.show()

plot_elite_trio(['A.Vinatieri', 'B.Walsh', 'J.Tucker'])