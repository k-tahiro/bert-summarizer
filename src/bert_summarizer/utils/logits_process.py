from abc import ABC, abstractmethod
from typing import List

import torch
from transformers import LogitsProcessor


class PrefixAllowedTokensFnBase(ABC):
    @abstractmethod
    def __call__(self, batch_id: int, inputs_ids: torch.Tensor) -> List[int]:
        raise NotImplementedError


class NGramPrefixAllowedTokensFn(PrefixAllowedTokensFnBase):
    def __init__(self, vocab_size: int, ngrams: List[List[int]]):
        self.all_tokens = list(range(vocab_size))
        self.ngrams = ngrams

    def __call__(self, batch_id: int, inputs_ids: torch.Tensor) -> List[int]:
        return [
            ngram[-1]
            for ngram in self.ngrams
            if self.match(ngram, inputs_ids)
        ] or self.all_tokens

    def match(self, ngram: List[int], inputs_ids: torch.Tensor) -> bool:
        prefix = ngram[:-1]
        actual = inputs_ids[-len(prefix):].tolist()
        return actual == prefix


class GlobalDistributionLogitsProcessor(LogitsProcessor):
    def __init__(self, distribution: torch.FloatTensor, lambda_: float):
        self.distribution = distribution
        self.lambda_ = lambda_

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = (1 - self.lambda_) * scores + self.lambda_ * self.distribution
        return scores
