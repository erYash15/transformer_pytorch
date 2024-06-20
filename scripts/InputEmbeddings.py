# InputEmbeddings

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    """
    Input Embeddings Class

    This class defines the input embedding layer used in the transformer model.
    It converts input tokens to their corresponding embeddings and scales them
    by the square root of the model's dimensionality (d_model).

    Args:
        d_model (int): The dimensionality of the embeddings.
        vocab_size (int): The size of the vocabulary (i.e., the number of unique tokens).

    Methods:
        forward(x): Computes the embeddings for the input tokens and scales them.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for the input embeddings.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Scaled embeddings for the input tokens.
        """
        # Compute embeddings and scale them by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# Example usage
if __name__ == "__main__":

    d_model = 512  # Dimension of the model
    vocab_size = 25000  # Maximum length of the sequence
    input_tensor = torch.randint(low = 0, high = vocab_size-1, size = (10,50), dtype=torch.int)
    input_embeddings = InputEmbeddings(d_model, vocab_size)
    print(input_embeddings(input_tensor).shape)