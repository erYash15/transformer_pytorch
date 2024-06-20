# PositionalEncoding

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Class

    This class adds positional encoding to the input embeddings to provide 
    the model with information about the relative or absolute position of 
    tokens in the sequence. The implementation includes a dropout layer 
    for regularization.

    Args:
        d_model (int): The dimensionality of the embeddings.
        seq_len (int): The maximum length of the input sequences.
        dropout (float): The dropout rate for regularization.
        div_term_implementation (str): Specifies the method to compute the 
            division term. Options are 'original' or 'modified'.
    
    Methods:
        forward(x): Adds positional encoding to the input tensor.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float, div_term_implementation: str = 'modified') -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) for positional encodings
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1) for position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term based on the specified implementation
        if div_term_implementation == 'original':
            div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch size
        pe = pe.unsqueeze(0)

        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for the positional encoding.

        Args:
            x (torch.Tensor): Input tensor containing embeddings.

        Returns:
            torch.Tensor: Input tensor with added positional encoding.
        """
        # Add positional encoding to the input tensor and disable gradient computation
        x = x + (self.pe[:, :x.shape[1],]).requires_grad_(False)
        return self.dropout(x)

# Example usage
if __name__ == "__main__":

    d_model = 512  # Dimension of the model
    max_len = 5000  # Maximum length of the sequence
    pos_encoder = PositionalEncoding(d_model, max_len, dropout=0.2, div_term_implementation = 'originial')

    # Dummy input tensor with shape (sequence length, batch size, d_model)
    x = torch.zeros(100, 32, d_model)

    # Apply positional encoding
    x = pos_encoder(x)
    print(x.shape)  # Should output: torch.Size([100, 32, 512])