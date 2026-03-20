#!/bin/bash
#SBATCH --job-name=fgoe_pro_25yr       
#SBATCH --output=fgoe_train_%j.log     
#SBATCH --partition=dsci-gpu                # (If this fails, try removing this line, some servers don't require it)
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:1                   
#SBATCH --mem=32G                      
#SBATCH --time=12:00:00                

# 1. Navigate to your project directory
cd ~/fgoe_project

# 2. Activate the fresh cluster environment you just built
source cluster_env/bin/activate

# 3. Print out GPU info for debugging/logging
echo "Starting FGOE Pro Job on GPU:"
nvidia-smi

# 4. Execute the Bayesian Amortized Engine
echo "Running SSM_Full.py..."
python3 SSM_Full.py

echo "Job completed."