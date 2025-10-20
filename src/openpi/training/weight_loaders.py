import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    skip_regex: str | None = None 

    def load(self, params: at.Params) -> at.Params:  
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)  
          
        # If skip_regex is provided, filter out matching keys  
        if self.skip_regex:  
            import re  
            pattern = re.compile(self.skip_regex)  
            flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")  
            flat_loaded = {k: v for k, v in flat_loaded.items() if not pattern.fullmatch(k)}  
            loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")  
          
        return _merge_params(loaded_params, params, missing_regex=".*lora.*|.*action_(in|out)_proj.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

def load(self, params: at.Params) -> at.Params:    
    loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)    
      
    # Don't filter here - let _merge_params handle it  
    # Just pass the full loaded_params and update missing_regex  
    return _merge_params(loaded_params, params, missing_regex=".*lora.*|.*action_(in|out)_proj.*|.*state_proj.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
