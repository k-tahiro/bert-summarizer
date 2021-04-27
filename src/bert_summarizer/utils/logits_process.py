from abc import ABC, abstractmethod
from typing import Callable, List

import torch
from transformers import LogitsProcessor


class PrefixAllowedTokensFnBase(ABC):
    @abstractmethod
    def __call__(self, batch_id: int, inputs_ids: torch.Tensor) -> List[int]:
        raise NotImplementedError


class NGramPrefixAllowedTokensFn(PrefixAllowedTokensFnBase):
    def __init__(self, vocab_size: int, eos_token_id: int, ngrams: List[List[int]]):
        self.all_tokens = list(range(vocab_size))
        self.eos_token_id = eos_token_id
        self.ngrams = ngrams

    def __call__(self, batch_id: int, inputs_ids: torch.Tensor) -> List[int]:
        allowed_tokens = [
            ngram[-1] for ngram in self.ngrams if self.match(ngram, inputs_ids)
        ]

        if allowed_tokens:
            allowed_tokens.append(self.eos_token_id)
            return allowed_tokens
        else:
            return self.all_tokens

    @staticmethod
    def match(ngram: List[int], inputs_ids: torch.Tensor) -> bool:
        prefix = ngram[:-1]
        actual: List[int] = inputs_ids[-len(prefix) :].tolist()
        return actual == prefix


class GlobalDistributionLogitsProcessor(LogitsProcessor):
    def __init__(self, distribution: torch.FloatTensor, lambda_: float):
        self.distribution = distribution
        self.lambda_ = lambda_

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores = (1 - self.lambda_) * scores + self.lambda_ * self.distribution
        return scores


class GlobalConditionalDistributionLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        conditional_distribution: Callable[[torch.LongTensor], torch.FloatTensor],
        lambda_: float,
    ):
        self.conditional_distribution = conditional_distribution
        self.lambda_ = lambda_

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores = (
            1 - self.lambda_
        ) * scores + self.lambda_ * self.conditional_distribution(input_ids)
        return scores
