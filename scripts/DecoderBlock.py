# DecoderBlock
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from ResidualConnection import ResidualConnection

class DecoderBlock(nn.Module):
    """
    Decoder block for a Transformer model.

    This block consists of a self-attention mechanism, a cross-attention mechanism (attending to encoder output),
    and a feed-forward network. Each sublayer is followed by a residual connection and layer normalization.
    """
    def __init__(
            self, 
            features: int,
            self_attention_block: MultiHeadAttentionBlock, 
            cross_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: float = 0.1
        ) -> None:
        """
        Initializes the DecoderBlock with self-attention, cross-attention, feed-forward blocks,
        and residual connections.

        Args:
            self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention mechanism.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention mechanism to attend to encoder output.
            feed_forward_block (FeedForwardBlock): Feed-forward network.
            dropout (float): Dropout rate to be applied after each sublayer.
        """
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention mechanism
        self.cross_attention_block = cross_attention_block  # Cross-attention mechanism
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])  # Residual connections with dropout

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor after applying self-attention, cross-attention, feed-forward network, and residual connections.
        """
        # Apply the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Apply the second residual connection with the cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Apply the third residual connection with the feed-forward block
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))

        return x  # Return the final output tensor

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

    # Create an instance of the DecoderBlock
    decoder_block = DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout)

    # Dummy input tensors
    x = torch.randn(batch_size, seq_len, d_model)  # Input tensor
    encoder_output = torch.randn(batch_size, seq_len, d_model)  # Encoder output tensor
    src_mask = torch.ones(batch_size, 1, seq_len, seq_len)  # Source mask tensor
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len)  # Target mask tensor

    # Forward pass through the DecoderBlock
    output = decoder_block(x, encoder_output, src_mask, tgt_mask)

    print("Output shape:", output.shape)