import re
import matplotlib.pyplot as plt

# 1. Point this to the log file you downloaded from the cluster
LOG_FILE_PATH = "v3_train_35728.log"

steps = []
accuracies = []

# 2. The Regex pattern to perfectly match your specific log output
# Looks for: Step 01000 | ... | Acc: 82.13%
log_pattern = re.compile(r"Step\s+(\d+)\s+\|.*?Acc:\s+([\d\.]+)%")

# 3. Parse the file
try:
    with open(LOG_FILE_PATH, "r") as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                step = int(match.group(1))
                acc = float(match.group(2))
                
                steps.append(step)
                accuracies.append(acc)
                
    print(f"Successfully extracted {len(steps)} data points!")
except FileNotFoundError:
    print(f"Error: Could not find '{LOG_FILE_PATH}'. Make sure it is in the same folder as this script!")

# 4. Build the Graph
if steps:
    plt.figure(figsize=(12, 6))
    
    # Plot the main accuracy line
    plt.plot(steps, accuracies, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=5, label='Training Accuracy')
    
    # Add some visual annotations for your specific milestones
    plt.axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='KL Penalty Hits 1.0 (Regularization begins)')
    
    # Check if Step 16000 exists in the extracted data to annotate the "wobble"
#    if 16000 in steps:
#        wobble_idx = steps.index(16000)
#        plt.annotate('The Step 16k "Wobble"\n(Escaping Local Minimum)', 
#                     xy=(16000, accuracies[wobble_idx]), 
#                     xytext=(16000, accuracies[wobble_idx] - 2),
#                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
#                     horizontalalignment='center')

    # Formatting to make it presentation-ready
    plt.title('FGOE State-Space Model: Accuracy over 30,000 Steps', fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(min(accuracies) - 2, max(accuracies) + 2) # Give the y-axis some breathing room
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Clean up the layout and show
    plt.tight_layout()
    plt.show()