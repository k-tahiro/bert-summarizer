import torch
from transformers import LogitsProccessor


class GlobalDistributionLogitsProcessor(LogitsProccessor):
    def __init__(self, distribution: torch.FloatTensor, lambda_: float):
        self.distribution = distribution
        self.lambda_ = lambda_

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = (1 - self.lambda_) * scores + self.lambda_ * self.distribution
        return scores
