import pytest
import torch

from bert_summarizer.utils.logits_process import NGramPrefixAllowedTokensFn, GlobalDistributionLogitsProcessor


class TestNGramPrefixAllowedTokensFn:
    @pytest.mark.parametrize('ngram, inputs_ids, expected', [
        ([0, 1, 2], torch.tensor([0, 1]), True),
        ([0, 1, 2], torch.tensor([1, 2]), False),
    ])
    def test_match(self, ngram, inputs_ids, expected):
        assert NGramPrefixAllowedTokensFn.match(ngram, inputs_ids) == expected

    @pytest.fixture
    def fn(self):
        return NGramPrefixAllowedTokensFn(
            32003,
            [
                [0, 1, 2],
                [3, 4, 5, 6],
            ],
        )

    @pytest.mark.parametrize('batch_id, inputs_ids, expected', [
        (0, torch.tensor([0, 1]), [2]),
        (1, torch.tensor([3, 4]), list(range(32003))),
    ])
    def test_call(self, fn, batch_id, inputs_ids, expected):
        assert fn(batch_id, inputs_ids) == expected


class TestGlobalDistributionLogitsProcessor:
    @pytest.fixture
    def logits_processor(self):
        return GlobalDistributionLogitsProcessor(
            torch.ones(2) / 2,
            0.5,
        )

    @pytest.mark.parametrize('input_ids, scores, expected', [
        (
            torch.ones(2, 2, dtype=torch.int),
            torch.tensor([[0.1, 0.9], [0.4, 0.6]]),
            torch.tensor([[0.3, 0.7], [0.45, 0.55]])
        ),
    ])
    def test_call(self, logits_processor, input_ids, scores, expected):
        assert (logits_processor(input_ids, scores) == expected).all()
