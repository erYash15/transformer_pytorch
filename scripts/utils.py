"""
Contains utility functions
"""
import torch
import torch.nn as nn

def count_parameters(model):
    """
    Count and print the number of trainable parameters in a model.

    This function iterates over all parameters in the given model,
    filters those that require gradients (trainable parameters), 
    and prints each parameter's count. Finally, it prints the total
    count of trainable parameters.

    Args:
        model (torch.nn.Module): The model for which to count parameters.

    Example:
        model = YourModel()
        count_parameters(model)
    """
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')


