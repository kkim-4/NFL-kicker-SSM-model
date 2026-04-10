import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from sklearn.preprocessing import StandardScaler

# --- 1. DATA PREPARATION ---
def prepare_amortized_tensors(file_path):
    print("Loading and filtering 1999-2024 data...")
    df = pd.read_csv(file_path)
    # Filter for the full 25-year era
    df = df[(df['season'] >= 1999) & (df['season'] <= 2024)]
    df = df.sort_values(['game_date', 'game_id'])
    
    # Track career kick count for volume features
    df['career_kicks'] = df.groupby('kicker_player_id').cumcount()
    
    feat_cols = ['kick_distance', 'quarter_seconds_remaining', 'score_differential', 'career_kicks']
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    
    # Create the global timeline (Week 1 1999 to Week 18 2024)
    timeline = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    timeline['t_idx'] = range(len(timeline))
    df = df.merge(timeline, on=['season', 'week'])
    
    k_ids = sorted(df['kicker_player_id'].unique())
    max_t = len(timeline)
    
    print(f"Processing {len(k_ids)} unique kickers across {max_t} weeks.")
    
    # Initialize Tensors (Batch x Time x Features)
    X = torch.zeros((len(k_ids), max_t, len(feat_cols)))
    y = torch.zeros((len(k_ids), max_t))
    mask = torch.zeros((len(k_ids), max_t))
    cs_mask = torch.zeros((len(k_ids), max_t), dtype=torch.bool)
    
    k_map = {kid: i for i, kid in enumerate(k_ids)}
    for _, row in df.iterrows():
        i, t = k_map[row['kicker_player_id']], int(row['t_idx'])
        X[i, t] = torch.from_numpy(row[feat_cols].values.astype(np.float32))
        y[i, t] = float(row['made'])
        mask[i, t] = 1.0
        
    # Set the 'Active Career' mask
    for i in range(len(k_ids)):
        first_idx = torch.where(mask[i] > 0)[0]
        if len(first_idx) > 0:
            cs_mask[i, first_idx[0]:] = True
            
    return X, y, mask, cs_mask, k_ids, df

# --- 2. MODEL DEFINITIONS ---
class AmortizedFGOE_Encoder(nn.Module):
    def __init__(self, in_features, decay_rate=0.85):
        super().__init__()
        self.decay_rate = decay_rate
        self.net = nn.Sequential(
            nn.Linear(in_features * 2 + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_weighted, x_current, volume):
        inputs = torch.cat([x_weighted, x_current, volume], dim=-1)
        params = self.net(inputs)
        return params[:, 0], torch.exp(params[:, 1].clamp(min=-3.0, max=0.5))

class FGOE_AmortizedSplineSSM(PyroModule):
    def __init__(self, in_features):
        super().__init__()
        # Situational baseline (Difficulty of the kick)
        self.baseline = nn.Sequential(nn.Linear(in_features, 8), nn.Dropout(0.3), nn.Linear(8, 1))
        # Amortized Encoder (The Scout)
        self.encoder = AmortizedFGOE_Encoder(in_features)

    def model(self, X, y, mask, cs_mask, kl_weight=1.0):
        pyro.module("fgoe", self)
        batch_size, max_t, _ = X.shape
        z_prev = torch.zeros(batch_size).to(X.device)
        
        with pyro.plate("kickers", batch_size):
            for t in range(max_t):
                # Apply KL Annealing factor to the prior
                with pyro.poutine.scale(scale=kl_weight):
                    z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_prev * 0.98, 0.05))
                
                # Only active if the kicker has debuted
                z_t = torch.where(cs_mask[:, t], z_t, torch.zeros_like(z_t))
                
                # Combine situational difficulty + Talent
                logits = (self.baseline(X[:, t, :]).squeeze(-1) + (z_t * 15.0)) / 0.3
                pyro.sample(f"obs_{t+1}", dist.Bernoulli(logits=logits).mask(mask[:, t] > 0), obs=y[:, t])
                z_prev = z_t

    def guide(self, X, y, mask, cs_mask, kl_weight=1.0):
        pyro.module("encoder", self.encoder)
        batch_size, max_t, feat_dim = X.shape
        weighted_history = torch.zeros(batch_size, feat_dim).to(X.device)
        
        with pyro.plate("kickers", batch_size):
            for t in range(max_t):
                if t > 0:
                    active = mask[:, t-1].unsqueeze(-1)
                    weighted_history = (active * X[:, t-1, :]) + (weighted_history * self.encoder.decay_rate)
                
                mu_q, sigma_q = self.encoder(weighted_history, X[:, t, :], X[:, t, -1:])
                
                # Apply the same KL Annealing factor to the posterior
                with pyro.poutine.scale(scale=kl_weight):
                    pyro.sample(f"z_{t+1}", dist.Normal(mu_q, sigma_q))

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # Detect Apple Silicon (MPS), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Ensure this matches your data file name
    file_path = 'nfl_kicks_1999_2024.csv' 
    X, y, mask, cs_mask, k_ids, full_df = prepare_amortized_tensors(file_path)
    X, y, mask, cs_mask = X.to(device), y.to(device), mask.to(device), cs_mask.to(device)

    model = FGOE_AmortizedSplineSSM(in_features=X.shape[2]).to(device)
    
    # Exponential Learning Rate Scheduler
    optimizer_args = {'optimizer': torch.optim.Adam, 
                      'optim_args': {'lr': 0.005, 'weight_decay': 1e-5},
                      'gamma': 0.999}
    optimizer = pyro.optim.ExponentialLR(optimizer_args)
    
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    pyro.clear_param_store()
    print("Starting training with KL Annealing...")
    
    n_steps = 2501
    warmup_steps = 1000 # Take 1000 steps to reach full KL penalty
    
    for step in range(n_steps):
        # Linear KL Annealing schedule
        kl_weight = min(1.0, 0.01 + (step / warmup_steps))
        
        loss = svi.step(X, y, mask, cs_mask, kl_weight=kl_weight)
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss:,.2f} | KL Weight: {kl_weight:.3f}")

    print("Finalizing and saving results...")
    # Generate predictions (forcing KL weight to 1.0 for valid inference)
    predictive = Predictive(model.model, guide=model.guide, num_samples=50)
    samples = predictive(X, y, mask, cs_mask, kl_weight=1.0)
    
    # Store the actual kicker IDs so evaluation scripts never misalign them
    samples['kicker_ids'] = k_ids 
    
    torch.save(samples, 'full_era_samples.pt')
    torch.save(model.state_dict(), 'fgoe_full_era_model.pth')
    print("Training complete. 1999-2024 results saved.")