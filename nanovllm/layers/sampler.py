import torch
from torch import nn


def filter_logits(logits, top_ks, top_ps):
    """Apply exact top-k, then nucleus filtering on its renormalized distribution."""
    sorted_logits, indices = logits.sort(dim=-1, descending=True, stable=True)
    ranks = torch.arange(logits.size(-1), device=logits.device)
    ks = torch.where(top_ks == 0, logits.size(-1), top_ks)
    sorted_logits = sorted_logits.masked_fill(ranks >= ks.unsqueeze(1), -float("inf"))
    probs = sorted_logits.softmax(dim=-1)
    # Keep the token crossing p, and always retain the most likely token.
    remove = (probs.cumsum(dim=-1) - probs) >= top_ps.unsqueeze(1)
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    return torch.empty_like(logits).scatter(1, indices, sorted_logits)


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_ks: torch.Tensor | None = None, top_ps: torch.Tensor | None = None):
        greedy = logits.argmax(dim=-1)
        scales = torch.where(temperatures == 0, 1.0, temperatures)
        logits = logits.float() / scales.unsqueeze(1)
        if top_ks is not None:
            logits = filter_logits(logits, top_ks, top_ps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy, sample_tokens)
