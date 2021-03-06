from typing import Dict, List

import pytest
import torch
from torch.optim import AdamW

from bert_summarizer.config.bertsum import BertSumAbsConfig
from bert_summarizer.models.bertsum import BertSumAbs
from bert_summarizer.trainers.lr_lambda import (
    TransformerScheduler,
    get_transformer_schedule_with_warmup,
)


class TestTransformerScheduler:
    @pytest.fixture
    def lr_lambda(self) -> TransformerScheduler:
        return TransformerScheduler(10000)

    def test_attribute(self, lr_lambda: TransformerScheduler) -> None:
        assert lr_lambda.num_warmup_steps == 10000

    @pytest.mark.parametrize(
        "current_step, expected", [(1, 1e-6), (10000, 1e-2), (1000000, 1e-3)]
    )
    def test_call(
        self, lr_lambda: TransformerScheduler, current_step: int, expected: float
    ) -> None:
        assert lr_lambda(current_step) == expected


class TestGetScheduler:
    @pytest.fixture
    def model(self) -> BertSumAbs:
        return BertSumAbs(BertSumAbsConfig())

    @pytest.fixture
    def optimizer(self, model: BertSumAbs) -> AdamW:
        params: List[Dict[str, torch.Tensor]] = [dict(params=[]), dict(params=[])]
        for i, p in enumerate(model.parameters()):
            params[i % 2]["params"].append(p)
        return AdamW(params)

    @pytest.mark.parametrize(
        "num_warmup_steps, expected",
        [(10000, [10000, 10000]), ([10000, 20000], [10000, 20000])],
    )
    def test_get_transformer_schedule_with_warmup(
        self, optimizer: AdamW, num_warmup_steps: int, expected: List[int]
    ) -> None:
        lr_scheduler = get_transformer_schedule_with_warmup(optimizer, num_warmup_steps)

        assert len(lr_scheduler.lr_lambdas) == len(expected)
        for lr_lambda, n in zip(lr_scheduler.lr_lambdas, expected):
            assert lr_lambda.num_warmup_steps == n
