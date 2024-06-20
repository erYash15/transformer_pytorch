# ResidualConnection

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from LayerNormalization import LayerNormalization

class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization.

    This module adds a residual connection around any given sublayer 
    with layer normalization and dropout applied before the residual 
    connection is added.

    Args:
        features (int): Number of input features
        dropout (float): Dropout rate to be applied after the sublayer.
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): A sublayer function or module.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection and dropout.
        """
        # Norm and Add
        return x + self.dropout(sublayer(self.norm(x)))

# Example usage
if __name__ == "__main__":

    batch_size = 16
    seq_len = 500
    d_model = 512
    dropout_rate = 0.1

    # Create a random tensor with shape (Batch, Seq_Len, d_model)
    x = torch.rand(batch_size, seq_len, d_model)

    # Define a simple sublayer function for demonstration
    class SimpleSublayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x):
            return F.relu(self.linear(x))

    sublayer = SimpleSublayer(d_model)

    # Create a residual connection block
    residual_connection = ResidualConnection(d_model, dropout_rate)

    # Apply the residual connection block with the sublayer
    output = residual_connection(x, sublayer)
    print("Output shape:", output.shape)