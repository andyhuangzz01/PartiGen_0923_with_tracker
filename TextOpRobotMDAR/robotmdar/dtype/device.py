

import numpy as np
import torch
from typing import Union, Dict, Tuple, Any

to_torch = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def tree_to_torch(x: Union[np.ndarray, torch.Tensor, Dict[str, Any], Tuple, list]) -> Union[torch.Tensor, Dict[str, Any], Tuple, list]:
    if isinstance(x, (list, tuple)):
        return type(x)(tree_to_torch(i) for i in x)
    elif isinstance(x, dict):
        return {k: tree_to_torch(v) for k, v in x.items()}
    else:
        return to_torch(x)
    
def tree_to_numpy(x: Union[torch.Tensor, Dict[str, Any], Tuple, list]) -> Union[np.ndarray, Dict[str, Any], Tuple, list]:
    if isinstance(x, (list, tuple)):
        return type(x)(tree_to_numpy(i) for i in x)
    elif isinstance(x, dict):
        return {k: tree_to_numpy(v) for k, v in x.items()}
    else:
        return to_numpy(x)
    
def wrap_torch_to_numpy(func):
    """
    Decorator to wrap a function that takes torch tensors and returns torch tensor.
    Make it a numpy function.
    """

    
    def wrapper(*args, **kwargs):
        torch_args = tree_to_torch(args)
        torch_kwargs:Dict[str, Any] = tree_to_torch(kwargs) # type: ignore
        result = func(*torch_args, **torch_kwargs)
        return  tree_to_numpy(result)
    return wrapper