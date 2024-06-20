# LayerNormalization

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    """
    Layer Normalization Class
    """
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))        # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(features))         # Learnable bias parameter

    def forward(self, x):
        """
        Forward pass for the layer normalization.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized input tensor with learnable scaling and bias.
        """

        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        # Apply normalization, scaling, and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

# Example usage
if __name__ == "__main__":

    batch = 16
    seq_len = 500
    d_model = 512
    # Input tensor of shape (batch_size, sequence_length, d_model)
    x = torch.randn(batch, seq_len, d_model)
    # Instantiate the LayerNormalization class
    layer_norm = LayerNormalization(d_model)
    # Apply layer normalization
    output = layer_norm(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)