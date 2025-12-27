
import torch
import torch.nn as nn
from common import SinusoidalPositionalEncoding


class BaselineTransformer(nn.Module):
    """Vanilla seq2seq Transformer using torch.nn.Transformer."""

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
        max_len: int = 4096,
        pad_id_src: int = 0,
        pad_id_tgt: int = 0,
    ):
        super().__init__()
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id_src)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id_tgt)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-norm (classic)
        )
        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

    @staticmethod
    def generate_square_subsequent_mask(t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor):
        # teacher forcing: predict tgt[:, 1:] from tgt[:, :-1]
        tgt_inp = tgt[:, :-1]
        tgt_len = tgt_inp.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)

        src_e = self.pos_enc(self.src_emb(src))
        tgt_e = self.pos_enc(self.tgt_emb(tgt_inp))

        out = self.transformer(
            src=src_e,
            tgt=tgt_e,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.lm_head(out)  # (B, T-1, V)
        return logits