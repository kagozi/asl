# ============================================================================
# models/baseline_transformer.py
# ============================================================================
"""Baseline transformer model for text-gloss translation."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaselineTransformer(nn.Module):
    """
    Baseline transformer model following Vaswani et al. (2017).
    Standard encoder-decoder architecture with multi-head attention.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0
    ):
        """
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            src_pad_idx: Padding index for source
            tgt_pad_idx: Padding index for target
        """
        super().__init__()
        
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False  # Post-LN (original transformer)
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Create padding mask."""
        return seq == pad_idx
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask  
            memory_mask: Memory attention mask
        
        Returns:
            Output logits (batch_size, tgt_seq_len-1, tgt_vocab_size)
        """
        # Remove last token from target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_seq_len = tgt_input.size(1)
        
        # Create padding masks
        src_padding_mask = self.create_padding_mask(src, self.src_pad_idx)
        tgt_padding_mask = self.create_padding_mask(tgt_input, self.tgt_pad_idx)
        
        # Generate causal mask for decoder
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)
        
        src_embedded = self.positional_encoding(src_embedded)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Pass through transformer
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        src_padding_mask = self.create_padding_mask(src, self.src_pad_idx)
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        memory = self.transformer.encoder(
            src_embedded,
            src_key_padding_mask=src_padding_mask
        )
        
        return memory
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        start_token_idx: int = 2,
        end_token_idx: int = 3
    ) -> torch.Tensor:
        """
        Generate translation using greedy decoding.
        
        Args:
            src: Source sequence (1, src_seq_len)
            max_len: Maximum generation length
            start_token_idx: Start token index
            end_token_idx: End token index
        
        Returns:
            Generated sequence (1, tgt_seq_len)
        """
        self.eval()
        device = src.device
        
        # Encode source
        memory = self.encode(src)
        src_padding_mask = self.create_padding_mask(src, self.src_pad_idx)
        
        # Initialize target with start token
        ys = torch.tensor([[start_token_idx]], device=device)
        
        for _ in range(max_len):
            tgt_seq_len = ys.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
            
            # Embed target
            tgt_embedded = self.tgt_embedding(ys) * math.sqrt(self.d_model)
            tgt_embedded = self.positional_encoding(tgt_embedded)
            
            # Decode
            output = self.transformer.decoder(
                tgt_embedded,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            # Get next token
            logits = self.fc_out(output[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Stop if end token
            if next_token.item() == end_token_idx:
                break
            
            # Append to sequence
            ys = torch.cat([ys, next_token], dim=1)
        
        return ys


def create_baseline_model(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> BaselineTransformer:
    """Create baseline transformer from config."""
    model_config = config['model']
    
    return BaselineTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=model_config['embedding_dim'],
        nhead=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dim_feedforward=model_config['ffn_dim'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        src_pad_idx=0,  # Assuming <pad> is index 0
        tgt_pad_idx=0
    )