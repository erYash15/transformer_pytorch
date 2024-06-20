# EncoderBlock

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from ResidualConnection import ResidualConnection

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Encoder Block for a Transformer model.

        This block consists of a self-attention mechanism followed by a feed-forward network, 
        with residual connections and layer normalization applied to each sublayer.

        Args:
            features (int): Number of input Features
            self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention mechanism.
            feed_forward_block (FeedForwardBlock): Feed-forward network.
            dropout (float): Dropout rate to be applied after each sublayer.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Output tensor after applying self-attention, feed-forward network, and residual connections.
        """
        # Apply the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # try without lambda

        # Apply the second residual connection with the feed-forward block
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x
    

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
    encoder_block = EncoderBlock(d_model, self_attention_block, feed_forward_block, dropout_rate)

    # Apply the encoder block
    output = encoder_block(x, mask)
    print("Output shape:", output.shape)