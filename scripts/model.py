import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# InputEmbeddings

class InputEmbeddings(nn.Module):
    """
    Input Embeddings Class
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Compute embeddings and scale them by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# PositionalEncoding

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Class
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
        # Add positional encoding to the input tensor and disable gradient computation
        x = x + (self.pe[:, :x.shape[1],]).requires_grad_(False)
        return self.dropout(x)



# LayerNormalization

class LayerNormalization(nn.Module):
    """
    Layer Normalization Class
    """
    def __init__(self, features: int, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))        # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(features))         # Learnable bias parameter

    def forward(self, x):
        """
        Forward pass for the layer normalization.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized input tensor with learnable scaling and bias.
        """

        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        # Apply normalization, scaling, and bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# FeedForwardBlock

class FeedForwardBlock(nn.Module):
    """
    FeedForward Block Class
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
        # Apply the first linear layer, ReLU activation, dropout, and then the second linear layer
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



# MultiHeadAttentionBlock

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block Class
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

    

    
# ResidualConnection

class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization.
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # Norm and Add
        return x + self.dropout(sublayer(self.norm(x)))



# EncoderBlock

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Encoder Block for a Transformer model.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Apply the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # try without lambda

        # Apply the second residual connection with the feed-forward block
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x

# Encoder

class Encoder(nn.Module):
    """
    Encoder block for a Transformer model.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)  # Initialize layer normalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # Apply each layer in the list to the input
        return self.norm(x)  # Apply layer normalization to the final output

# DecoderBlock

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
         """
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention mechanism
        self.cross_attention_block = cross_attention_block  # Cross-attention mechanism
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])  # Residual connections with dropout

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply the first residual connection with the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Apply the second residual connection with the cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Apply the third residual connection with the feed-forward block
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))

        return x  # Return the final output tensor

# Decoder

class Decoder(nn.Module):
    """
    Decoder for a Transformer model.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of decoder layers
        self.norm = LayerNormalization(features)  # Initialize layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # Apply each decoder layer to the input

        return self.norm(x)  # Apply layer normalization to the final output


# Projection Layer

class ProjectionLayer(nn.Module):
    """
    Projection layer for a Transformer model.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear layer to project to vocabulary size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return F.log_softmax(self.proj(x), dim=-1)  # Apply linear projection and log softmax


# Transformer

class Transformer(nn.Module):

    def __init__(
            self, 
            encoder: Encoder, 
            decoder: Decoder, 
            src_embed: InputEmbeddings, 
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
        ) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.src_tgt = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.src_embed(tgt)
        tgt = self.src_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


# build_transformer

def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int, 
        d_model: int=512, 
        N: int=6, 
        h: int=8, 
        dropout: float=0.1, 
        d_ff: int=2048
    ) -> Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameter
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer
    