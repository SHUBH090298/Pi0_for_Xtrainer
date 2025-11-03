import numpy as np    
from openpi.training import data_loader, config    
from openpi.policies import policy_config    
import matplotlib.pyplot as plt    
  
if __name__ == '__main__':  
    # Load your training config    
    train_config = config.get_config("pi0_xtrainer")    
        
    # Create data loader (without shuffling for reproducibility)    
    loader = data_loader.create_data_loader(    
        train_config,    
        shuffle=False,    
        num_batches=10,  # Validate on 10 batches    
        framework="jax"    
    )    
        
    # Load trained policy    
    checkpoint_dir = "checkpoints/pi0_xtrainer/dobot_pick_place/9000"    
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)    
        
    # Collect predictions and ground truth    
    predictions = []    
    ground_truths = []    
        
    for observation, actions_gt in loader:    
        # Get model predictions    
        actions_pred = policy.infer(observation)    
            
        predictions.append(actions_pred)    
        ground_truths.append(actions_gt)    
            
    # Compute metrics    
    predictions = np.concatenate(predictions, axis=0)    
    ground_truths = np.concatenate(ground_truths, axis=0)    
        
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
    plt.show()