# FeedForwardBlock

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FeedForwardBlock(nn.Module):
    """
    FeedForward Block Class

    This class defines a feedforward neural network block used in the 
    transformer model. It consists of two linear transformations with 
    a ReLU activation in between, and dropout for regularization.

    Args:
        d_model (int): The dimensionality of the input and output features.
        d_ff (int): The dimensionality of the intermediate (hidden) features.
        dropout (float): The dropout rate for regularization.

    Methods:
        forward(x): Applies the feedforward block to the input tensor.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # First linear layer with input size d_model and output size d_ff
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Second linear layer with input size d_ff and output size d_model
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        """
        Forward pass for the feedforward block.

        Args:
            x (torch.Tensor): Input tensor with shape (Batch, Seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with shape (Batch, Seq_len, d_model).
        """
        # Apply the first linear layer, ReLU activation, dropout, and then the second linear layer
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Example usage
if __name__ == "__main__":

    batch_size = 16
    seq_len = 500
    d_model = 512

    # Input tensor of shape (batch_size, sequence_length, d_model)
    x = torch.randn(batch_size, seq_len, d_model)  # Example input tensor

    # Instantiate the FeedForwardBlock class
    d_ff = 2048  # Dimension of the feedforward layer
    dropout = 0.1  # Dropout rate
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    # Apply the feedforward block
    output = feed_forward_block(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)