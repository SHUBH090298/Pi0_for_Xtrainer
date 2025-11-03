"""  
Test inference script for Xtrainer robot.  
Save this as: examples/test_xtrainer_inference.py or test_inference.py  
"""  
  
import numpy as np  
from openpi.training import config as _config  
from openpi.policies import policy_config  
  
# Load your training config  
config = _config.get_config("pi0_xtrainer")  
  
# Point to your checkpoint (adjust step number as needed)  
checkpoint_dir = "checkpoints/pi0_xtrainer/dobot_pick_place/9000"  
  
print("Loading policy from checkpoint...")  
policy = policy_config.create_trained_policy(config, checkpoint_dir)  
  
print("Creating dummy observation...")  
# Create observation matching the format AFTER RepackTransform  
# Your RepackTransform maps:  
# "observation.images.top" -> "images.top"  
# "observation.images.left_wrist" -> "images.left_wrist"  
# "observation.images.right_wrist" -> "images.right_wrist"  
# "observation.state" -> "state"  
  
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