import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def load_latent_stats(path, csv_path='nfl_kicks_1999_2024.csv'):
    print(f"⌛ Loading {path}...")
    samples = torch.load(path, map_location='cpu')
    
    # 1. Manually reconstruct the Kicker ID list from the CSV
    # This MUST match the 'sorted()' logic used in the training script
    df_temp = pd.read_csv(csv_path)
    ids = sorted(df_temp['kicker_player_id'].astype(str).unique())
    
    # 2. Extract z keys (Latent Skill) and stack them
    z_keys = sorted([k for k in samples.keys() if k.startswith('z_')], 
                    key=lambda x: int(x.split('_')[1]))
    
    # Shape: [Samples, Kickers, Time]
    z_grid = np.stack([samples[k].numpy() for k in z_keys], axis=2)
    
    # Sanity Check: Ensure the number of kickers in the tensor matches the CSV
    if z_grid.shape[1] != len(ids):
        raise ValueError(f"Mismatch! Tensor has {z_grid.shape[1]} kickers, but CSV has {len(ids)}.")
    
    # 3. Compute Mean and Variance
    mean_z = np.mean(z_grid, axis=0) # [Kickers, Time]
    var_z = np.var(z_grid, axis=0)
    
    return {
        'mean_z': mean_z,
        'var_z': var_z,
        'ids': ids,
        'global_mean': np.mean(mean_z, axis=1) # Mean skill across the era
    }

# 1. Load both models
v1 = load_latent_stats('full_era_samples.pt')
v2 = load_latent_stats('full_era_samples_deep.pt')

# 2. Align Kickers
common_ids = sorted(list(set(v1['ids']) & set(v2['ids'])))
v1_idx = [v1['ids'].index(i) for i in common_ids]
v2_idx = [v2['ids'].index(i) for i in common_ids]

v1_scores = v1['global_mean'][v1_idx]
v2_scores = v2['global_mean'][v2_idx]

# 3. Compute Spearman Rank Correlation
corr, _ = spearmanr(v1_scores, v2_scores)

# 4. Find the "Disruptors" (Kickers who moved the most)
diff = v2_scores - v1_scores
rank_diff = pd.DataFrame({
    'id': common_ids,
    'v1_skill': v1_scores,
    'v2_skill': v2_scores,
    'delta': diff
}).sort_values('delta', ascending=False)

print("\n" + "="*45)
print("🔍 LATENT FEATURE COMPARISON")
print("-" * 45)
print(f"Rank Correlation (Spearman): {corr:.4f}")
print(f"Mean Skill Shift:           {np.mean(diff):.4f}")
print(f"Mean Uncertainty Reduction: {np.mean(v1['var_z']) - np.mean(v2['var_z']):.4f}")
print("="*45)

print("\n🚀 Top 5 'Rising Stars' in V2 (Model discovered more talent):")
print(rank_diff.head(5))

print("\n📉 Top 5 'Falling Stars' in V2 (Model realized it was overestimating):")
print(rank_diff.tail(5))

# 5. Visualization: Skill Correlation Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x=v1_scores, y=v2_scores, alpha=0.6)
plt.plot([min(v1_scores), max(v1_scores)], [min(v1_scores), max(v1_scores)], 'r--')
plt.title("Latent Skill Alignment: V1 vs V2")
plt.xlabel("V1 Average Skill (z)")
plt.ylabel("V2 Average Skill (z)")
plt.savefig('skill_comparison_scatter.png')

print("\n✅ Comparison complete. Check 'skill_comparison_scatter.png'.")