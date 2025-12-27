from typing import Callable
import torch
from typing import List, Tuple
from vocab import Vocab, decode
from train import RunConfig
from torch.utils.data import DataLoader


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

def decode_loader_full(
    model,
    loader: DataLoader,
    text_vocab: Vocab,
    gloss_vocab: Vocab,
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """
    Decode *all* items in loader (full set), returns (pred_strs, ref_strs).
    """
    model.eval()
    preds, refs = [], []

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_kpm = batch.src_key_padding_mask.to(device)
            tgt = batch.tgt.to(device)

            ys = greedy_decode(
                model,
                src=src,
                src_key_padding_mask=src_kpm,
                bos_id=gloss_vocab.bos_id,
                eos_id=gloss_vocab.eos_id,
                pad_id=gloss_vocab.pad_id,
                max_len=cfg.max_decode_len,
            )

            for i in range(src.size(0)):
                hyp_ids = ys[i].tolist()
                if gloss_vocab.eos_id in hyp_ids:
                    hyp_ids = hyp_ids[: hyp_ids.index(gloss_vocab.eos_id) + 1]
                hyp = decode(hyp_ids, gloss_vocab, skip_special=True)

                ref_ids = tgt[i].tolist()
                ref = decode(ref_ids, gloss_vocab, skip_special=True)

                preds.append(hyp)
                refs.append(ref)

    return preds, refs
