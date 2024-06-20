# ProjectionLayer

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ProjectionLayer(nn.Module):
    """
    Projection layer for a Transformer model.

    This layer projects the hidden states from the model's d_model dimensionality
    to the vocabulary size dimensionality and applies a log softmax function.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the ProjectionLayer with a linear projection layer.

        Args:
            d_model (int): Dimensionality of the model's hidden states.
            vocab_size (int): Size of the vocabulary (number of target classes).
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear layer to project to vocabulary size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) with log softmax applied.
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return F.log_softmax(self.proj(x), dim=-1)  # Apply linear projection and log softmax


# Example Usage
if __name__ == "__main__":

    d_model = 512  # Dimensionality of the model's hidden states
    vocab_size = 10000  # Size of the vocabulary
    seq_len = 10  # Sequence length
    batch_size = 32  # Batch size

    # Create an instance of the ProjectionLayer
    projection_layer = ProjectionLayer(d_model, vocab_size)

    # Dummy input tensor of shape (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass through the ProjectionLayer
    output = projection_layer(x)

    # Output tensor should have shape (batch_size, seq_len, vocab_size)
    print("Output shape:", output.shape)