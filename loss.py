import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, V)
        target: (N,)
        """
        log_probs = logits.log_softmax(dim=-1)  # keeps grad

        # mask for non-pad positions
        mask = target.ne(self.padding_idx)
        if mask.sum() == 0:
            # avoid NaNs if a batch is all padding (rare but possible)
            return log_probs.sum() * 0.0

        # filter log_probs/target WITH grad still enabled
        log_probs = log_probs[mask]
        target = target[mask]

        # build the smoothed target distribution without tracking grads
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0.0

        return -(true_dist * log_probs).sum(dim=-1).mean()
