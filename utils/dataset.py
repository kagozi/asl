"""PyTorch Dataset and DataLoader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocab import Vocab, encode


class TranslationDataset(Dataset):
    def __init__(self, df, text_vocab: Vocab, gloss_vocab: Vocab):
        self.df = df
        self.text_vocab = text_vocab
        self.gloss_vocab = gloss_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        src_ids = encode(row["processed_text"], self.text_vocab, add_bos_eos=False)
        tgt_ids = encode(row["processed_gloss"], self.gloss_vocab, add_bos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


@dataclass
class Batch:
    src: torch.Tensor
    tgt: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], text_vocab: Vocab, gloss_vocab: Vocab) -> Batch:
    src_list = [b[0] for b in batch]
    tgt_list = [b[1] for b in batch]

    src = pad_sequence(src_list, batch_first=True, padding_value=text_vocab.pad_id)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=gloss_vocab.pad_id)

    src_key_padding_mask = src.eq(text_vocab.pad_id)
    tgt_key_padding_mask = tgt.eq(gloss_vocab.pad_id)

    return Batch(src=src, tgt=tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
