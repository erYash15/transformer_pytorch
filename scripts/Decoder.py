# Decoder

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from LayerNormalization import LayerNormalization
from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    """
    Decoder for a Transformer model.

    This decoder consists of a stack of decoder layers, each containing self-attention, 
    cross-attention, and feed-forward networks with residual connections and layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initializes the Decoder with a list of layers and a layer normalization module.

        Args:
            features (int): Number of input feautres for layewr normalization
            layers (nn.ModuleList): List of decoder layers to be included in the decoder.
        """
        super().__init__()
        self.layers = layers  # List of decoder layers
        self.norm = LayerNormalization(features)  # Initialize layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor after applying all decoder layers and layer normalization.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # Apply each decoder layer to the input

        return self.norm(x)  # Apply layer normalization to the final output
    

# Example Usage
if __name__ == "__main__":
    d_model = 512  # Dimension of the model
    num_heads = 8  # Number of attention heads
    d_ff = 2048  # Dimension of the feed-forward layer
    dropout = 0.1  # Dropout rate
    seq_len = 10  # Sequence length
    batch_size = 32  # Batch size

    # Create instances of the attention and feed-forward blocks
    self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
    cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

    # Create a list of decoder blocks
    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout)
        for _ in range(6)  # Number of decoder layers
    ])

    # Create an instance of the Decoder
    decoder = Decoder(d_model, decoder_blocks)

    # Dummy input tensors
    x = torch.randn(batch_size, seq_len, d_model)  # Input tensor
    encoder_output = torch.randn(batch_size, seq_len, d_model)  # Encoder output tensor
    src_mask = torch.ones(batch_size, 1, seq_len, seq_len)  # Source mask tensor
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len)  # Target mask tensor

    # Forward pass through the Decoder
    output = decoder(x, encoder_output, src_mask, tgt_mask)

    print("Output shape:", output.shape)
