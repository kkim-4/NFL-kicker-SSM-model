import os
import time
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
    df = df[(df['season'] >= 1999) & (df['season'] <= 2024)]
    
    # Chronological sort: season -> week -> date -> time remaining
    df = df.sort_values(
        ['season', 'week', 'game_date', 'quarter_seconds_remaining'], 
        ascending=[True, True, True, False]
    )
    
    df['career_kicks'] = df.groupby('kicker_player_id').cumcount()
    
    feat_cols = ['kick_distance', 'quarter_seconds_remaining', 'score_differential', 'career_kicks']
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    
    timeline = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    timeline['t_idx'] = range(len(timeline))
    df = df.merge(timeline, on=['season', 'week'])
    
    k_ids = sorted(df['kicker_player_id'].unique())
    max_t = len(timeline)
    
    print(f"Processing {len(k_ids)} unique kickers across {max_t} weeks.")
    
    X = torch.zeros((len(k_ids), max_t, len(feat_cols)))
    y = torch.zeros((len(k_ids), max_t))
    mask = torch.zeros((len(k_ids), max_t))
    cs_mask = torch.zeros((len(k_ids), max_t), dtype=torch.bool)
    
    k_map = {kid: i for i, kid in enumerate(k_ids)}
    
    i_idx = df['kicker_player_id'].map(k_map).values
    t_idx = df['t_idx'].values
    
    # FIXED: Added .copy() to BOTH tensors to silence PyTorch non-writable memory warnings
    X[i_idx, t_idx] = torch.from_numpy(df[feat_cols].to_numpy().copy()).float()
    y[i_idx, t_idx] = torch.from_numpy(df['made'].to_numpy().copy()).float()
    mask[i_idx, t_idx] = 1.0
    
    for i in range(len(k_ids)):
        first_idx = torch.where(mask[i] > 0)[0]
        if len(first_idx) > 0:
            cs_mask[i, first_idx[0]:] = True
            
    return X, y, mask, cs_mask, k_ids, df

# --- 2. VECTORIZED MODEL DEFINITIONS ---
class AmortizedRNN_Encoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=16, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(16 + in_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, X):
        gru_out, _ = self.gru(X) 
        history = torch.cat([torch.zeros_like(gru_out[:, 0:1, :]), gru_out[:, :-1, :]], dim=1)
        inputs = torch.cat([history, X], dim=-1)
        params = self.net(inputs)
        return params[:, :, 0], torch.exp(params[:, :, 1].clamp(min=-3.0, max=0.5))

class FGOE_AmortizedSplineSSM(PyroModule):
    def __init__(self, in_features):
        super().__init__()
        self.baseline = nn.Sequential(nn.Linear(in_features, 8), nn.Dropout(0.3), nn.Linear(8, 1))
        self.encoder = AmortizedRNN_Encoder(in_features)

    def model(self, X, y, mask, cs_mask, kl_weight=1.0):
        pyro.module("fgoe", self)
        batch_size, max_t, _ = X.shape
        baseline_logits = self.baseline(X).squeeze(-1) 

        with pyro.plate("kickers", batch_size, dim=-1):
            z_prev = torch.zeros(batch_size).to(X.device)
            for t in pyro.markov(range(max_t)):
                with pyro.poutine.scale(scale=kl_weight):
                    loc = (z_prev * 0.98) * cs_mask[:, t].float()
                    scale = 0.05 * cs_mask[:, t].float() + 1e-5
                    z_t = pyro.sample(f"z_{t+1}", dist.Normal(loc, scale))
                
                logits = (baseline_logits[:, t] + (z_t * 15.0)) / 0.3
                obs_mask = mask[:, t] > 0
                pyro.sample(f"obs_{t+1}", dist.Bernoulli(logits=logits).mask(obs_mask), obs=y[:, t])
                z_prev = z_t

    def guide(self, X, y, mask, cs_mask, kl_weight=1.0):
        pyro.module("encoder", self.encoder)
        batch_size, max_t, _ = X.shape
        mu_q, sigma_q = self.encoder(X) 
        
        mu_q = mu_q * cs_mask.float()
        sigma_q = torch.where(cs_mask, sigma_q, torch.ones_like(sigma_q) * 1e-5)

        with pyro.plate("kickers", batch_size, dim=-1):
            for t in pyro.markov(range(max_t)):
                with pyro.poutine.scale(scale=kl_weight):
                    pyro.sample(f"z_{t+1}", dist.Normal(mu_q[:, t], sigma_q[:, t]))

# --- 3. EVALUATION HELPER ---
def fast_evaluate_accuracy(model, X, y, mask, cs_mask):
    """Calculates deterministic accuracy using the mean of the latents."""
    model.eval() 
    with torch.no_grad():
        mu_q, _ = model.encoder(X)
        mu_q = mu_q * cs_mask.float()
        baseline_logits = model.baseline(X).squeeze(-1)
        
        logits = (baseline_logits + (mu_q * 15.0)) / 0.3
        probs = torch.sigmoid(logits)
        
        valid_kicks = mask > 0
        valid_probs = probs[valid_kicks]
        valid_y = y[valid_kicks]
        
        predictions = (valid_probs >= 0.5).float()
        correct = (predictions == valid_y).float().sum()
        accuracy = correct / len(valid_y)
        
    model.train() 
    return accuracy.item()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    file_path = 'nfl_kicks_1999_2024.csv' 
    X, y, mask, cs_mask, k_ids, full_df = prepare_amortized_tensors(file_path)
    X, y, mask, cs_mask = X.to(device), y.to(device), mask.to(device), cs_mask.to(device)

    model = FGOE_AmortizedSplineSSM(in_features=X.shape[2]).to(device)
    
    n_steps = 30000        
    warmup_steps = 5000    
    
    optimizer_config = {
        'optimizer': torch.optim.AdamW, 
        'optim_args': {
            'lr': 0.01, 
            'weight_decay': 1e-4
        },
        'T_max': n_steps,      
        'eta_min': 1e-5        
    }
    
    pyro_scheduler = pyro.optim.CosineAnnealingLR(optimizer_config)
    
    svi = SVI(model.model, model.guide, pyro_scheduler, loss=Trace_ELBO(retain_graph=True))

    pyro.clear_param_store()
    print(f"📉 Starting deep high-speed training ({n_steps} steps)...")
    
    start_time = time.time()
    history_steps, history_loss, history_acc = [], [], []
    
    for step in range(n_steps):
        kl_weight = min(1.0, 0.001 + (step / warmup_steps))
        
        loss = svi.step(X, y, mask, cs_mask, kl_weight=kl_weight)
        pyro_scheduler.step()
        
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            
            # FIXED: Safe Learning Rate Retrieval for Pyro
            try:
                if pyro_scheduler.optim_objs:
                    first_sched = list(pyro_scheduler.optim_objs.values())[0]
                    current_lr = first_sched.optimizer.param_groups[0]['lr']
                else:
                    current_lr = 0.01  
            except Exception:
                current_lr = 0.01

            current_acc = fast_evaluate_accuracy(model, X, y, mask, cs_mask)
            
            history_steps.append(step)
            history_loss.append(loss)
            history_acc.append(current_acc)
            
            print(f"Step {step:05d} | Loss: {loss:,.0f} | Acc: {current_acc*100:.2f}% | KL: {kl_weight:.3f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

    print("Finalizing and saving results...")
    
    training_history = pd.DataFrame({'step': history_steps, 'elbo_loss': history_loss, 'accuracy': history_acc})
    training_history.to_csv('training_history.csv', index=False)
    
    print("Generating predictive samples (N=200)...")
    predictive = Predictive(model.model, guide=model.guide, num_samples=200)
    samples = predictive(X, y, mask, cs_mask, kl_weight=1.0)
    
    samples['kicker_ids'] = k_ids 
    torch.save(samples, 'full_era_samples_deep.pt')
    torch.save(model.state_dict(), 'fgoe_full_era_model_deep.pth')
    print(f"Training complete in {(time.time() - start_time)/60:.1f} minutes.")