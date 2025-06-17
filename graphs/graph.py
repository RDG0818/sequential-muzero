import matplotlib.pyplot as plt
import re
import numpy as np

def load_smoothed_returns(file_path, window=3):
    episodes = []
    avg_returns = []

    pattern = r"Episodes:\s*(\d+)\s*\|\s*Avg Return:\s*(-?\d+\.?\d*)"

    with open(file_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                episodes.append(int(match.group(1)))
                avg_returns.append(float(match.group(2)))

    # Apply simple moving average smoothing
    smoothed_returns = np.convolve(avg_returns, np.ones(window)/window, mode='valid')
    smoothed_episodes = episodes[(window-1)//2: -(window//2) or None]  # Align with smoothing

    return smoothed_episodes, smoothed_returns

# Load both datasets
joint_eps, joint_returns = load_smoothed_returns("results/joint.txt", window=3)
indep_eps, indep_returns = load_smoothed_returns("results/independent.txt", window=3)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(joint_eps, joint_returns, label="Joint", color='blue')
plt.plot(indep_eps, indep_returns, label="Independent", color='red')

plt.title("MPE Simple Spread")
plt.xlabel("Episodes")
plt.ylabel("Avg Return")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("joint_vs_independent.png")
plt.close()