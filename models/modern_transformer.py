# ============================================================================
# models/modern_transformer.py
# ============================================================================
"""
Modern transformer with state-of-the-art techniques:
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)
- RMSNorm
- SwiGLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================================
# COMPONENT 1: Rotary Position Embeddings (RoPE)
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding from RoFormer."""
    
    def __init__(self, dim: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if sequence length changes."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys."""
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)
        
        # Apply rotation
        q_embed = (q * self._cos_cached) + (self.rotate_half(q) * self._sin_cached)
        k_embed = (k * self._cos_cached) + (self.rotate_half(k) * self._sin_cached)
        
        return q_embed, k_embed


# ============================================================================
# COMPONENT 2: RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ============================================================================
# COMPONENT 3: SwiGLU Activation
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation function (gated linear unit with Swish)."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# COMPONENT 4: Grouped-Query Attention
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) for efficient inference."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: Optional[int] = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.num_groups = num_heads // self.num_kv_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.rope.apply_rotary_pos_emb(q, k)
        
        # Repeat K, V for grouped attention
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out


# ============================================================================
# COMPONENT 5: Encoder and Decoder Layers
# ============================================================================

class ModernTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with modern improvements."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: Optional[int] = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-LN architecture
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(dim)
        # SwiGLU with increased FFN dimension (2.7x for SwiGLU vs 4x for standard FFN)
        self.ffn = SwiGLU(dim, int(dim * 2.7))
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN attention block
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask, key_padding_mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Pre-LN FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


class ModernTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with modern improvements."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: Optional[int] = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = RMSNorm(dim)
        self.self_attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention
        self.norm2 = RMSNorm(dim)
        self.cross_attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * 2.7))
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None, 
        tgt_key_padding_mask: Optional[torch.Tensor] = None, 
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, tgt_mask, tgt_key_padding_mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Cross-attention block
        residual = x
        x = self.norm2(x)
        x = self._cross_attention(x, memory, memory_key_padding_mask)
        x = self.dropout2(x)
        x = residual + x
        
        # FFN block
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout3(x)
        x = residual + x
        
        return x
    
    def _cross_attention(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-attention implementation."""
        B, L_q, D = query.shape
        L_kv = key_value.shape[1]
        
        num_heads = self.cross_attn.num_heads
        num_kv_heads = self.cross_attn.num_kv_heads
        head_dim = self.cross_attn.head_dim
        scale = self.cross_attn.scale
        
        # Project Q from query, K and V from key_value
        q = self.cross_attn.q_proj(query).view(B, L_q, num_heads, head_dim).transpose(1, 2)
        k = self.cross_attn.k_proj(key_value).view(B, L_kv, num_kv_heads, head_dim).transpose(1, 2)
        v = self.cross_attn.v_proj(key_value).view(B, L_kv, num_kv_heads, head_dim).transpose(1, 2)
        
        # Repeat K, V for grouped attention
        if self.cross_attn.num_groups > 1:
            k = k.repeat_interleave(self.cross_attn.num_groups, dim=1)
            v = v.repeat_interleave(self.cross_attn.num_groups, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.cross_attn.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.cross_attn.out_proj(out)
        
        return out


# ============================================================================
# COMPONENT 6: Complete Modern Transformer
# ============================================================================

class ModernTransformer(nn.Module):
    """State-of-the-art Transformer with RoPE, GQA, RMSNorm, and SwiGLU."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_q_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # Default to full MHA if num_kv_heads not specified
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        
        # RMSNorm for embeddings
        self.src_norm = RMSNorm(d_model)
        self.tgt_norm = RMSNorm(d_model)
        
        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            ModernTransformerEncoderLayer(d_model, num_q_heads, num_kv_heads, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            ModernTransformerDecoderLayer(d_model, num_q_heads, num_kv_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final layer norm
        self.encoder_final_norm = RMSNorm(d_model)
        self.decoder_final_norm = RMSNorm(d_model)
        
        # Output projection
        self.dropout_out = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
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
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequences (batch_size, src_seq_len)
            tgt: Target sequences (batch_size, tgt_seq_len)
        
        Returns:
            Output logits (batch_size, tgt_seq_len-1, tgt_vocab_size)
        """
        # Remove last token from target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_seq_len = tgt_input.size(1)
        
        # Create padding masks
        src_padding_mask = self.create_padding_mask(src, self.src_pad_idx)
        tgt_padding_mask = self.create_padding_mask(tgt_input, self.tgt_pad_idx)
        
        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        
        # Embed and scale
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)
        
        # Apply RMSNorm (RoPE is applied inside attention layers)
        src_embedded = self.src_norm(src_embedded)
        tgt_embedded = self.tgt_norm(tgt_embedded)
        
        # Encoder
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask=src_padding_mask)
        memory = self.encoder_final_norm(memory)
        
        # Decoder
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(
                output, 
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
        output = self.decoder_final_norm(output)
        
        # Output projection with dropout
        output = self.dropout_out(output)
        logits = self.fc_out(output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        start_token_idx: int = 2,
        end_token_idx: int = 3
    ) -> torch.Tensor:
        """Generate translation using greedy decoding."""
        self.eval()
        device = src.device
        
        # Create padding mask for source
        src_padding_mask = self.create_padding_mask(src, self.src_pad_idx)
        
        # Encode source
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.src_norm(src_embedded)
        
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask=src_padding_mask)
        memory = self.encoder_final_norm(memory)
        
        # Initialize target with start token
        ys = torch.tensor([[start_token_idx]], device=device)
        
        for _ in range(max_len):
            tgt_seq_len = ys.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
            
            # Embed target
            tgt_embedded = self.tgt_embedding(ys) * math.sqrt(self.d_model)
            tgt_embedded = self.tgt_norm(tgt_embedded)
            
            # Decode
            output = tgt_embedded
            for layer in self.decoder_layers:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )
            output = self.decoder_final_norm(output)
            
            # Get next token
            logits = self.fc_out(output[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Stop if end token
            if next_token.item() == end_token_idx:
                break
            
            # Append to sequence
            ys = torch.cat([ys, next_token], dim=1)
        
        return ys


def create_modern_model(config: dict, src_vocab_size: int, tgt_vocab_size: int) -> ModernTransformer:
    """Create modern transformer from config."""
    model_config = config['model']
    
    return ModernTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=model_config['embedding_dim'],
        num_q_heads=model_config['num_q_heads'],
        num_kv_heads=model_config.get('num_kv_heads'),
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len'],
        src_pad_idx=0,
        tgt_pad_idx=0
    )