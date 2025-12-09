import wandb
import matplotlib.pyplot as plt
import pandas as pd

# Initialize wandb API
api = wandb.Api()

# Get the run
run = api.run("chad_qian_tamu/ManiSkill-PPO/4vgjj5j1")

# Get the history with train/return and its min/max if available
# Wandb typically logs these with __MIN and __MAX suffixes
history = run.history(keys=["train/return"])

# Check what columns are available
print("Available columns in history:")
print(history.columns.tolist())

# Remove NaN values
history = history.dropna(subset=['train/return'])

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the main line (mean)
plt.plot(history['_step'], history['train/return'], linewidth=1.5, color='#1f77b4')

plt.xlabel('Step', fontsize=12)
plt.ylabel('Train return', fontsize=12)
# plt.title(f'Training return - {run.name}', fontsize=14)
plt.grid(True, alpha=0.3)
# plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('train_return_plot.png', dpi=300, bbox_inches='tight')
print(f"Plot saved as 'train_return_plot.png'")

# Also save the data to CSV
history.to_csv('train_return_data.csv', index=False)
print(f"Data saved as 'train_return_data.csv'")

# Show the plot
plt.show()

# Print some statistics
print(f"\nStatistics:")
print(f"Max return: {history['train/return'].max():.4f}")
print(f"Min return: {history['train/return'].min():.4f}")
print(f"Mean return: {history['train/return'].mean():.4f}")
print(f"Final return: {history['train/return'].iloc[-1]:.4f}")