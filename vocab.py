from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Iterable


@dataclass
class Vocab:
    tokens: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    pad_id: int
    unk_id: int
    bos_id: int | None = None
    eos_id: int | None = None


def build_word_vocab(
    sentences: Iterable[str],
    specials: List[str],
    min_freq: int = 1,
) -> Vocab:
    counts = Counter()
    for s in sentences:
        if not s:
            continue
        counts.update(s.split())

    tokens = list(specials)
    for tok, c in sorted(counts.items()):
        if c >= min_freq and tok not in specials:
            tokens.append(tok)

    token_to_id = {t: i for i, t in enumerate(tokens)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    pad_id = token_to_id.get("<pad>", 0)
    unk_id = token_to_id.get("<unk>", 1)
    bos_id = token_to_id.get("<start>")
    eos_id = token_to_id.get("<end>")

    return Vocab(
        tokens=tokens,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )


def encode(sentence: str, vocab: Vocab, add_bos_eos: bool = False) -> List[int]:
    if not sentence:
        ids = []
    else:
        ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in sentence.split()]

    if add_bos_eos:
        if vocab.bos_id is None or vocab.eos_id is None:
            raise ValueError("Vocab missing <start>/<end> tokens")
        return [vocab.bos_id] + ids + [vocab.eos_id]

    return ids


def decode(ids: List[int], vocab: Vocab, skip_special: bool = True) -> str:
    out = []
    for i in ids:
        tok = vocab.id_to_token.get(int(i), "<unk>")
        if skip_special and tok in {"<pad>", "<start>", "<end>"}:
            continue
        out.append(tok)
    return " ".join(out)
