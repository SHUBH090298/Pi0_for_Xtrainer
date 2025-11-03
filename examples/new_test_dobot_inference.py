import numpy as np  
from openpi.training import config  
from openpi.policies import policy_config  
import matplotlib.pyplot as plt  
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  
  
if __name__ == '__main__':  
    # Load your training config  
    train_config = config.get_config("pi0_xtrainer")  
      
    # Get dataset metadata to retrieve the correct FPS  
    dataset_meta = LeRobotDatasetMetadata(train_config.data.repo_id)  
      
    # Load the raw LeRobot dataset with correct FPS  
    dataset = LeRobotDataset(  
        train_config.data.repo_id,  
        delta_timestamps={  
            "action": [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]  
        }  
    )  
      
    # Load trained policy  
    checkpoint_dir = "checkpoints/pi0_xtrainer/dobot_pick_place/9000"  
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)  
      
    # Collect predictions and ground truth  
    predictions = []  
    ground_truths = []  
      
    # Process 10 batches worth of data  
    num_samples = min(10 * train_config.batch_size, len(dataset))  
      
    for i in range(num_samples):  
        raw_data = dataset[i]  
          
        # Construct observation dict in the format expected by policy.infer()  
        # This matches the RepackTransform mapping in XtrainerDataConfig  
        obs_dict = {  
            "images": {  
                "top": raw_data["observation.images.top"],  
                "left_wrist": raw_data["observation.images.left_wrist"],  
                "right_wrist": raw_data["observation.images.right_wrist"],  
            },  
            "state": raw_data["observation.state"],  
        }  
          
        # Get model predictions  
        result = policy.infer(obs_dict)  
        actions_pred = result["actions"]  
          
        # Get ground truth actions  
        actions_gt = raw_data["action"]  
          
        predictions.append(actions_pred)  
        ground_truths.append(actions_gt)  
      
    # Compute metrics  
    predictions = np.stack(predictions, axis=0)  
    ground_truths = np.stack(ground_truths, axis=0)  
      
    mse = np.mean((predictions - ground_truths) ** 2)  
    mae = np.mean(np.abs(predictions - ground_truths))  
      
    print(f"MSE: {mse:.4f}")  
    print(f"MAE: {mae:.4f}")  
    print(f"Per-dimension MAE: {np.mean(np.abs(predictions - ground_truths), axis=(0,1))}")  
      
    per_dim_mae = np.mean(np.abs(predictions - ground_truths), axis=(0,1))  
    plt.bar(range(14), per_dim_mae)  
    plt.xlabel("Action Dimension")  
    plt.ylabel("MAE")  
    plt.title("Per-Dimension Mean Absolute Error") 