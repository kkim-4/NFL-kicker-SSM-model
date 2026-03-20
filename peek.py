import torch

# Load your results
samples = torch.load('full_era_samples.pt', map_location='cpu')

# Print the keys (the variable names)
print("--- Available Variables in your Model ---")
for key in samples.keys():
    # Print the name and the shape of the data
    if isinstance(samples[key], torch.Tensor):
        print(f"Variable: {key} | Shape: {list(samples[key].shape)}")
    else:
        print(f"Variable: {key}")