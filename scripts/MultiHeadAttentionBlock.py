# MultiHeadAttentionBlock

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block Class

    This class implements the multi-head attention mechanism as described 
    in the "Attention Is All You Need" paper. It allows the model to jointly 
    attend to information from different representation subspaces.

    Args:
        d_model (int): The dimensionality of the input and output features.
        h (int): The number of attention heads.
        dropout (float): The dropout rate for regularization.

    Methods:
        forward(q, k, v, mask): Applies multi-head attention to the input tensors.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h == 0, "d_model not divisible by h"

        self.d_k = d_model//h
        
        self.w_q = nn.Linear(d_model, d_model) #wq
        self.w_k = nn.Linear(d_model, d_model) #wk
        self.w_v = nn.Linear(d_model, d_model) #wv

        self.w_o = nn.Linear(d_model, d_model) #wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod 
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute the attention scores and apply the attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape (Batch, h, Seq_Len, d_k).
            key (torch.Tensor): Key tensor of shape (Batch, h, Seq_Len, d_k).
            value (torch.Tensor): Value tensor of shape (Batch, h, Seq_Len, d_k).
            mask (torch.Tensor): Mask tensor to avoid attending to certain positions.
            dropout (nn.Dropout): Dropout layer for regularization.

        Returns:
            torch.Tensor: The output tensor after applying attention.
            torch.Tensor: The attention scores.
        """
        d_k = query.shape[-1]
        
        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # very low value (indicateing -inf) to positions where mask == 0
            attention_scores.masked_fill(mask==0,-1e9)

        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch. h. seq_len, seq_len) -> (batch, h, seq_len, d_k)
        # also resturn attention scores used for visualization
        return (attention_scores @ value), attention_scores        

    def forward(self, q, k, v, mask):
        """
        Forward pass for the multi-head attention block.

        Args:
            q (torch.Tensor): Query tensor of shape (Batch, Seq_Len, d_model).
            k (torch.Tensor): Key tensor of shape (Batch, Seq_Len, d_model).
            v (torch.Tensor): Value tensor of shape (Batch, Seq_Len, d_model).
            mask (torch.Tensor): Mask tensor to avoid attending to certain positions.

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention.
        """

        # Apply linear layers to get query, key, and value tensors
        query = self.w_q(q) # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        key = self.w_k(k) # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        value = self.w_v(v) # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, h, d_k) -> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        # Apply the attention mechanism
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Reshape and transpose back to the original shape
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # Apply the final linear layer (Batch, seq_len, d_model)
        return self.w_o(x)


# Example Usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    h = 4
    dropout = 0.1

    # Create a random tensor with shape (Batch, Seq_Len, d_model)
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
        
    # Create a mask tensor to avoid attending to certain positions
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, 2:, :] = 0  # Masking out the last three positions

    # Create a multi-head attention block
    mha_block = MultiHeadAttentionBlock(d_model, h, dropout)

    # Apply the multi-head attention block
    output = mha_block(q, k, v, mask)
    print("Output shape:", output.shape)
