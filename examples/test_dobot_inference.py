"""    
Test inference script for Xtrainer robot.    
Save this as: examples/test_xtrainer_inference.py or test_inference.py    
"""    
    
import numpy as np    
import matplotlib.pyplot as plt  
from openpi.training import config as _config    
from openpi.policies import policy_config    
    
# Load your training config    
config = _config.get_config("pi0_xtrainer")    
    
# Point to your checkpoint (adjust step number as needed)    
checkpoint_dir = "checkpoints/pi0_xtrainer/dobot_pick_place/9000"    
    
print("Loading policy from checkpoint...")    
policy = policy_config.create_trained_policy(config, checkpoint_dir)    
    
print("Creating dummy observation...")    
observation = {    
    "images": {    
        "top": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),    
        "left_wrist": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),    
        "right_wrist": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),    
    },    
    "state": np.random.rand(14).astype(np.float32),    
    "prompt": "pick and place"    
}    
    
print("Running inference...")    
output = policy.infer(observation)    
    
print(f"\n=== Inference Results ===")    
print(f"Action chunk shape: {output['actions'].shape}")    
print(f"Expected shape: (action_horizon, 14)")    
print(f"\nFirst action: {output['actions'][0]}")    
print(f"Policy timing: {output['policy_timing']}")    
print("\n✓ Inference test successful!")  
  
# Plot the action chunk  
actions = output['actions']  # Shape: (action_horizon, 14)  
action_horizon, action_dim = actions.shape  
  
# Create subplots for each action dimension  
fig, axes = plt.subplots(4, 4, figsize=(16, 12))  
fig.suptitle('Predicted Action Trajectories', fontsize=16)  
  
for dim in range(action_dim):  
    row = dim // 4  
    col = dim % 4  
    ax = axes[row, col]  
      
    ax.plot(actions[:, dim], marker='o', linewidth=2)  
    ax.set_xlabel('Time Step')  
    ax.set_ylabel('Action Value')  
    ax.set_title(f'Dimension {dim}')  
    ax.grid(True, alpha=0.3)  
  
# Hide unused subplots  
for idx in range(action_dim, 16):  
    row = idx // 4  
    col = idx % 4  
    axes[row, col].axis('off')  
  
plt.tight_layout()  
plt.savefig('action_trajectories.png', dpi=150, bbox_inches='tight')  
print("\n✓ Saved action trajectories plot to 'action_trajectories.png'")  
plt.show()  
  
# Also create a heatmap view  
fig, ax = plt.subplots(figsize=(12, 6))  
im = ax.imshow(actions.T, aspect='auto', cmap='viridis')  
ax.set_xlabel('Time Step', fontsize=12)  
ax.set_ylabel('Action Dimension', fontsize=12)  
ax.set_title('Action Chunk Heatmap', fontsize=14)  
ax.set_yticks(range(action_dim))  
plt.colorbar(im, ax=ax, label='Action Value')  
plt.tight_layout()  
plt.savefig('action_heatmap.png', dpi=150, bbox_inches='tight')  
print("✓ Saved action heatmap to 'action_heatmap.png'")  
plt.show()