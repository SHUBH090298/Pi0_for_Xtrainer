import dataclasses  
from typing import ClassVar  
import numpy as np  
from openpi import transforms  
  
@dataclasses.dataclass(frozen=True)  
class XtrainerInputs(transforms.DataTransformFn):  
    """Inputs for your custom robot policy.  
      
    Expected inputs:  
    - images: dict[name, img] where img is [channel, height, width]  
    - state: [N] where N is your state dimension  
    - actions: [action_horizon, N] where N is your action dimension  
    """  
      
    # Define your expected camera names (after RepackTransform)  
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (  
        "top",  
        "left_wrist",  
        "right_wrist",  
    )  
      
    def __call__(self, data: dict) -> dict:  
        in_images = data["images"]  
          
        # Map dataset cameras to model input format  
        images = {}  
        image_masks = {}  
          
        # Top camera (maps to base)  
        if "top" in in_images:  
            images["base_0_rgb"] = in_images["top"]  
            image_masks["base_0_rgb"] = np.True_  
          
        # Left wrist camera  
        if "left_wrist" in in_images:  
            images["left_wrist_0_rgb"] = in_images["left_wrist"]  
            image_masks["left_wrist_0_rgb"] = np.True_  
          
        # Right wrist camera  
        if "right_wrist" in in_images:  
            images["right_wrist_0_rgb"] = in_images["right_wrist"]  
            image_masks["right_wrist_0_rgb"] = np.True_  
          
        inputs = {  
            "image": images,  
            "image_mask": image_masks,  
            "state": data["state"],  
        }  
          
        if "actions" in data:  
    # Pad 14-dim actions to 32-dim for the model  
            actions = np.asarray(data["actions"])  
            padded_actions = np.pad(actions, ((0, 0), (0, 32 - 14)), mode='constant')  
            inputs["actions"] = padded_actions 
          
        if "prompt" in data:  
            inputs["prompt"] = data["prompt"]  
          
        return inputs  
  
  
@dataclasses.dataclass(frozen=True)  
class XtrainerOutputs(transforms.DataTransformFn):  
    """Outputs for your custom robot policy."""  
      
    def __call__(self, data: dict) -> dict:  
        # Extract only the action dimensions you need  
        actions = np.asarray(data["actions"][:, :14])  # Adjust dimension as needed  
        return {"actions": actions}