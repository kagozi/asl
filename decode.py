from typing import Callable

import torch


def greedy_decode(
    model,
    src: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 100,
) -> torch.Tensor:
    """Greedy decoding for both BaselineTransformer and ModernTransformer.

    Returns: (B, <=max_len) token ids including <start> but excluding <end> (trimmed later).
    """
    device = src.device
    B = src.size(0)

    # Encode
    if hasattr(model, "encode"):
        memory = model.encode(src, src_key_padding_mask)
        decode_step = lambda ys: model.decode(ys, memory, tgt_key_padding_mask=None, memory_key_padding_mask=src_key_padding_mask)
        lm_head = model.lm_head
    else:
        # torch.nn.Transformer baseline path
        src_e = model.pos_enc(model.src_emb(src))
        memory = model.transformer.encoder(src_e, src_key_padding_mask=src_key_padding_mask)

        def decode_step(ys):
            tgt_e = model.pos_enc(model.tgt_emb(ys))
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
            out = model.transformer.decoder(
                tgt=tgt_e,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask,
            )
            return out

        lm_head = model.lm_head

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        dec_out = decode_step(ys)
        logits = lm_head(dec_out[:, -1, :])
        next_id = logits.argmax(dim=-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        finished |= next_id.eq(eos_id)
        if finished.all():
            break

    return ys