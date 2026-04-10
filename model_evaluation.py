import torch
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

# --- SETTINGS ---
SAMPLE_PATH = 'full_era_samples_deep.pt'
DATA_PATH = 'nfl_kicks_1999_2024.csv'

def load_and_sync():
    print("📂 Loading Posterior Samples and CSV...")
    samples = torch.load(SAMPLE_PATH, map_location='cpu')
    df = pd.read_csv(DATA_PATH)
    
    # Ensure IDs are strings and Dates are chronological
    df['kicker_player_id'] = df['kicker_player_id'].astype(str)
    df = df.sort_values(['season', 'week', 'game_date'])

    return samples, df

def build_indices(df):
    print("🔗 Building Spatio-Temporal Maps...")
    # Timeline Map (Season, Week) -> Index
    time_groups = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    week_to_idx = {(int(s), int(w)): i for i, (s, w) in enumerate(zip(time_groups['season'], time_groups['week']))}

    # Kicker Map ID -> Index
    k_ids = sorted(df['kicker_player_id'].unique())
    id_to_idx = {kid: i for i, kid in enumerate(k_ids)}
    
    return week_to_idx, id_to_idx, k_ids

def run_evaluation(samples, df, week_to_idx, id_to_idx):
    print("🧠 Extracting Latent Probabilities...")
    # Extract 'obs_t' keys and compute mean across samples (Variational Mean)
    obs_keys = sorted([k for k in samples.keys() if k.startswith('obs_')], 
                      key=lambda x: int(x.split('_')[1]))
    
    # Grid Shape: [Kickers, Time]
    prob_grid = np.stack([np.mean(samples[k].numpy(), axis=0) for k in obs_keys], axis=1)
    
    y_prob, y_true = [], []
    
    # Data Alignment Loop
    for s, w, kid, made in zip(df['season'], df['week'], df['kicker_player_id'], df['made']):
        k_idx = id_to_idx.get(kid)
        t_idx = week_to_idx.get((int(s), int(w)))
        
        if k_idx is not None and t_idx is not None:
            y_prob.append(prob_grid[k_idx, t_idx])
            y_true.append(made)

    return np.array(y_true), np.array(y_prob)

def extract_latent_leaderboard(samples, k_ids):
    print("🏆 Recovering 'Hidden Talent' Feature ($z$)...")
    # Extract 'z_t' keys (The Latent Skill)
    z_keys = sorted([k for k in samples.keys() if k.startswith('z_')], 
                    key=lambda x: int(x.split('_')[1]))
    
    # Latent Skill Grid: [Samples, Kickers, Time]
    z_grid = np.stack([samples[k].numpy() for k in z_keys], axis=2)
    
    # Compute mean skill over time for each kicker
    # We take the mean across samples and then the peak skill reached in their career
    mean_skill_over_samples = np.mean(z_grid, axis=0) 
    peak_latent_skill = np.max(mean_skill_over_samples, axis=1)
    
    leaderboard = pd.DataFrame({
        'kicker_id': k_ids,
        'peak_latent_skill': peak_latent_skill
    }).sort_values('peak_latent_skill', ascending=False)
    
    return leaderboard

if __name__ == "__main__":
    samples, df = load_and_sync()
    week_map, id_map, k_ids = build_indices(df)
    
    # 1. Metric Validation
    y_true, y_prob = run_evaluation(samples, df, week_map, id_map)
    
    brier = brier_score_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob > 0.5))

    print("\n" + "="*45)
    print(f"📊 MODEL CALIBRATION (N={len(y_true)})")
    print("-" * 45)
    print(f"Brier Score:  {brier:.4f} (Closer to 0 is better)")
    print(f"AUC-ROC:      {auc:.4f} (Closer to 1 is better)")
    print(f"Accuracy:     {acc:.2%}")
    print("="*45)