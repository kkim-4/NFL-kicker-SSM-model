import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("Loading data and CSV...")
samples = torch.load('full_era_samples_deep.pt', map_location='cpu')
df_raw = pd.read_csv('nfl_kicks_1999_2024.csv')

# Standardize IDs and Names
kicker_ids = sorted(df_raw['kicker_player_id'].unique())
id_to_name = df_raw.set_index('kicker_player_id')['kicker_player_name'].to_dict()

# --- 2. DEFINE ACTIVE FILTERS & QUALIFYING THRESHOLDS ---
print("Filtering kickers by sample size...")

def get_qualified_kickers(df_slice, min_kicks):
    """Calculates total kicks in a timeframe and filters out small sample sizes."""
    counts = df_slice.groupby('kicker_player_id').size()
    return set(counts[counts >= min_kicks].index)

# All-time minimum: 50 kicks to be ranked
qualified_all_time = get_qualified_kickers(df_raw, min_kicks=50)

# Past 5 years minimum: 30 kicks in that specific window
df_past_5 = df_raw[df_raw['season'] >= 2020]
qualified_past_5 = get_qualified_kickers(df_past_5, min_kicks=30)

# 2024 minimum: 10 kicks in the current season
df_2024 = df_raw[df_raw['season'] == 2024]
qualified_2024 = get_qualified_kickers(df_2024, min_kicks=10)

# --- 3. RECONSTRUCT THE TIMELINE & BUILD GAME-DAY MASKS ---
print("Reconstructing 550 weeks and building Game-Day masks...")
z_keys = sorted([k for k in samples.keys() if k.startswith('z_')], 
                key=lambda x: int(x.split('_')[1]))
z_list = [samples[k].numpy() if isinstance(samples[k], torch.Tensor) else samples[k] for k in z_keys]
z_timeline = np.stack(z_list, axis=2) 

# Create the timeline index so we know exactly when kicks happened
timeline = df_raw[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
timeline['t_idx'] = range(len(timeline))
df_raw = df_raw.merge(timeline, on=['season', 'week'], how='left')

# THE FIX: Only mark '1' for the EXACT weeks they recorded a kick (Ignores Off-Seasons/Byes/Retirement)
gameday_mask = np.zeros((len(kicker_ids), len(timeline)))
for kid in kicker_ids:
    k_idx = kicker_ids.index(kid)
    k_data = df_raw[df_raw['kicker_player_id'] == kid]
    for t_idx in k_data['t_idx']:
        gameday_mask[k_idx, t_idx] = 1.0

# --- 4. THE GENERATOR (MASKED & FILTERED) ---
def generate_and_save(data_subset, mask_subset, filename, label, active_filter=None):
    
    # Step 1: Sum the latent skill ONLY for the weeks the kicker was active
    sum_z = np.sum(data_subset * mask_subset, axis=2) 
    
    # Step 2: Count exactly how many weeks they actually played in this subset
    active_weeks = np.sum(mask_subset, axis=1) 
    active_weeks[active_weeks == 0] = 1 # Prevent division by zero for completely inactive players
    
    # Step 3: Get the TRUE average skill, immune to 0-padding and off-season decay
    samples_across_time = sum_z / active_weeks 
    
    # Step 4: Isolate the pure Bayesian Uncertainty across the 200 samples
    mean_z = np.mean(samples_across_time, axis=0)
    std_z = np.std(samples_across_time, axis=0) 
    
    # THE FIX: Crank the penalty to 3 Sigma (99.7% confidence bound)
    cert = mean_z - (3.0 * std_z)
    
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
    
    # Power Rating 0-100 (Re-calculated based only on qualified players)
    s_min, s_max = lb['Certified Score'].min(), lb['Certified Score'].max()
    lb['Power Rating'] = 100 * (lb['Certified Score'] - s_min) / (s_max - s_min)
    
    lb.sort_values('Certified Score', ascending=False).drop(columns=['kicker_id']).to_csv(filename, index=False)
    print(f"Saved {label} to {filename} ({len(lb)} qualified kickers)")

# --- GENERATE CSV FILES ---
first_week_2024 = timeline[timeline['season'] == 2024]['t_idx'].min()

generate_and_save(z_timeline, gameday_mask, 'career_all_time.csv', 'All-Time Career', active_filter=qualified_all_time)
generate_and_save(z_timeline[:, :, -105:], gameday_mask[:, -105:], 'past_5_years_active.csv', 'Past 5 Years (Active)', active_filter=qualified_past_5)
generate_and_save(z_timeline[:, :, first_week_2024:], gameday_mask[:, first_week_2024:], 'final_2024_rankings_active.csv', '2024 Season (Active)', active_filter=qualified_2024)

# --- 5. ELITE TRIO PLOT ---
def plot_elite_trio(names):
    plt.figure(figsize=(14, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    name_to_id = {v: k for k, v in id_to_name.items()}
    
    for name in names:
        if name not in name_to_id: continue
        kid = name_to_id[name]
        k_idx = kicker_ids.index(kid)
        
        # Pull their trajectory and apply the mask to hide padded zeros
        mask_k = gameday_mask[k_idx, :]
        traj = np.mean(z_timeline[:, k_idx, :], axis=0)
        traj[mask_k == 0] = np.nan  # NaN prevents plotting the fake 0s and decay
        
        smoothed = pd.Series(traj).rolling(window=15, min_periods=1).mean()
        plt.plot(smoothed, label=name, linewidth=3)

    plt.title("Select Kicker Career Trajectories", color='white', size=16, pad=20)
    plt.legend(facecolor='#222222', labelcolor='white', fontsize=12)
    plt.grid(color='#444444', linestyle='--', alpha=0.3)
    plt.tick_params(colors='white')
    plt.savefig('kicker_comparison.png', facecolor='#121212')
    plt.show()

# --- 6. 2024 EXTREMES PLOT ---
def plot_2024_extremes():
    print("Ranking qualified 2024 kickers to find Top 3 and Bottom 3...")
    
    idx_2024 = timeline[timeline['season'] == 2024]['t_idx'].values
    weeks_2024 = timeline[timeline['season'] == 2024]['week'].values
    
    if len(weeks_2024) == 0:
        print("No data found for 2024.")
        return

    kicker_season_stats = []
    
    # We use the 'qualified_2024' set here to ensure rookies with 1 kick don't ruin the chart
    for kid in qualified_2024:
        if kid not in kicker_ids: 
            continue
            
        k_idx = kicker_ids.index(kid)
        
        # Extract the trajectory and mask for 2024
        traj_2024 = np.mean(z_timeline[:, k_idx, idx_2024], axis=0)
        mask_2024 = gameday_mask[k_idx, idx_2024]
        
        # Calculate season average strictly on active game weeks
        active_weeks = np.sum(mask_2024)
        if active_weeks == 0:
            continue
            
        season_avg = np.sum(traj_2024 * mask_2024) / active_weeks
        
        # Clean the trajectory line for plotting (hide byes/missed weeks)
        plot_traj = traj_2024.copy()
        plot_traj[mask_2024 == 0] = np.nan
        
        kicker_season_stats.append({
            'name': id_to_name.get(kid, kid),
            'season_avg': season_avg,
            'trajectory': plot_traj
        })
        
    kicker_season_stats.sort(key=lambda x: x['season_avg'], reverse=True)
    
    top_3 = kicker_season_stats[:3]
    bottom_3 = kicker_season_stats[-3:]
    
    plt.figure(figsize=(14, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    
    for item in top_3:
        plt.plot(weeks_2024, item['trajectory'], marker='o', linewidth=2.5, markersize=7, 
                 label=f"TOP: {item['name']} ({item['season_avg']:.4f})")
                 
    for item in bottom_3:
        plt.plot(weeks_2024, item['trajectory'], marker='X', linewidth=2, markersize=7, linestyle='--',
                 label=f"WORST: {item['name']} ({item['season_avg']:.4f})")

    plt.title("2024 Week-to-Week: Top 3 vs Worst 3 Qualified Kickers", color='white', size=16, pad=20)
    plt.xlabel("Week of 2024 Season", color='white', size=12)
    plt.ylabel("Latent Skill Score (z)", color='white', size=12)
    
    plt.xticks(weeks_2024, color='white')
    plt.yticks(color='white')
    
    plt.legend(facecolor='#222222', labelcolor='white', fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(color='#444444', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('kicker_2024_extremes_filtered.png', facecolor='#121212', bbox_inches='tight')
    plt.show()

# Uncomment to run plots:
# plot_elite_trio(['A.Vinatieri', 'B.Walsh', 'J.Tucker'])
plot_2024_extremes()