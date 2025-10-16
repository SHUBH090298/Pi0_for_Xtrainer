import dataclasses  
from typing import ClassVar  
import einops  
import numpy as np  
from openpi import transforms  
from openpi.models import model as _model  
  
  
def _parse_image(image) -> np.ndarray:  
    """Convert image to uint8 numpy array in (H, W, C) format."""  
    image = np.asarray(image)  # Convert tensor to numpy  
    if np.issubdtype(image.dtype, np.floating):  
        image = (255 * image).astype(np.uint8)  
    if image.shape[0] == 3:  # If channels-first (C, H, W)  
        image = einops.rearrange(image, "c h w -> h w c")  
    return image  
  
  
@dataclasses.dataclass(frozen=True)  
class XtrainerInputs(transforms.DataTransformFn):  
    """Inputs for your custom robot policy.  
      
    Expected inputs:  
    - images: dict[name, img] where img is [channel, height, width]  
    - state: [N] where N is your state dimension  
    - actions: [action_horizon, N] where N is your action dimension  
    """  
      
    # Determines which model will be used  
    model_type: _model.ModelType = _model.ModelType.PI0  
      
    # Define your expected camera names (after RepackTransform)  
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (  
        "top",  
        "left_wrist",  
        "right_wrist",  
    )  
      
    def __call__(self, data: dict) -> dict:  
        in_images = data["images"]  
          
        # Parse and convert images from tensors to numpy arrays  
        images = {}  
        image_masks = {}  
          
        # Top camera (maps to base)  
        if "top" in in_images:  
            images["base_0_rgb"] = _parse_image(in_images["top"])  
            image_masks["base_0_rgb"] = np.True_  
          
        # Left wrist camera  
        if "left_wrist" in in_images:  
            images["left_wrist_0_rgb"] = _parse_image(in_images["left_wrist"])  
            image_masks["left_wrist_0_rgb"] = np.True_  
          
        # Right wrist camera  
        if "right_wrist" in in_images:  
            images["right_wrist_0_rgb"] = _parse_image(in_images["right_wrist"])  
            image_masks["right_wrist_0_rgb"] = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.True_  
          
        inputs = {  
            "image": images,  
            "image_mask": image_masks,  
            "state": np.asarray(data["state"]),  
        }  
          
        # Pass through actions during training  
        if "actions" in data:  
            inputs["actions"] = np.asarray(data["actions"])  
          
        # Pass through prompt  
        if "prompt" in data:  
            inputs["prompt"] = data["prompt"]  
          
        return inputs  
  
  
@dataclasses.dataclass(frozen=True)  
class XtrainerOutputs(transforms.DataTransformFn):  
    """Outputs for your custom robot policy."""  
      
    def __call__(self, data: dict) -> dict:  
        # Extract only the first 14 action dimensions  
        return {"actions": np.asarray(data["actions"][:, :14])}