# Encoder

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from LayerNormalization import LayerNormalization
from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from EncoderBlock import EncoderBlock

class Encoder(nn.Module):
    """
    Encoder block for a Transformer model.

    This block consists of a stack of layers (e.g., multi-head self-attention and feed-forward networks)
    followed by layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initializes the Encoder with a list of layers and a layer normalization module.

        Args:
            features (int): Number of input features
            layers (nn.ModuleList): List of layers to be included in the encoder.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)  # Initialize layer normalization

    def forward(self, x, mask):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor after applying all layers and layer normalization.
        """
        for layer in self.layers:
            x = layer(x, mask)  # Apply each layer in the list to the input
        return self.norm(x)  # Apply layer normalization to the final output
    

# Example Usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    h = 4
    d_ff = 32
    dropout_rate = 0.1

    # Create random tensors for input and mask
    x = torch.rand(batch_size, seq_len, d_model)
    # Create a mask tensor to avoid attending to certain positions
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, 2:, :] = 0  # Masking out the last three positions

    # Initialize self-attention block and feed-forward block
    self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_rate)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)

    # Initialize the encoder block
    encoder_blocks = nn.ModuleList()
    for i in range(10):
        encoder_blocks.append(EncoderBlock(d_model, self_attention_block, feed_forward_block, dropout_rate))

    encoder = Encoder(d_model, encoder_blocks)

    # Apply the encoder block
    output = encoder(x, mask)
    print("Output shape:", output.shape)