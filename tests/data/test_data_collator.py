from typing import Dict, List

import pytest
import torch

from bert_summarizer.data import EncoderDecoderDataCollatorWithPadding
from bert_summarizer.data.datasets import BertSumAbsDataset

from .datasets.test_bertsum import encoded_data_abs


class TestEncoderDecoderDataCollatorWithPadding:
    @pytest.fixture
    def dataset(self) -> BertSumAbsDataset:
        return BertSumAbsDataset("bert-base-uncased", [])

    @pytest.fixture
    def train_data_collator(
        self, dataset: BertSumAbsDataset
    ) -> EncoderDecoderDataCollatorWithPadding:
        return EncoderDecoderDataCollatorWithPadding(tokenizer=dataset.tokenizer)  # type: ignore

    @pytest.fixture
    def eval_data_collator(
        self, dataset: BertSumAbsDataset
    ) -> EncoderDecoderDataCollatorWithPadding:
        return EncoderDecoderDataCollatorWithPadding(tokenizer=dataset.tokenizer).eval()  # type: ignore

    @pytest.fixture
    def expected_train_batch(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                [
                    [
                        101,
                        2023,
                        2003,
                        1996,
                        2034,
                        3793,
                        2005,
                        5604,
                        1012,
                        102,
                        101,
                        2023,
                        3793,
                        3397,
                        2048,
                        11746,
                        1012,
                        102,
                    ],
                    [
                        101,
                        2023,
                        2003,
                        1996,
                        2117,
                        3793,
                        2005,
                        5604,
                        1012,
                        102,
                        101,
                        2023,
                        3793,
                        3397,
                        2048,
                        11746,
                        1012,
                        102,
                    ],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            "token_type_ids": torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            "decoder_input_ids": torch.tensor(
                [
                    [2, 2034, 3231, 3793, 3],
                    [2, 2117, 3231, 3793, 3],
                ]
            ),
            "decoder_attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            ),
            "decoder_token_type_ids": torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            "labels": torch.tensor(
                [
                    [2, 2034, 3231, 3793, 3],
                    [2, 2117, 3231, 3793, 3],
                ]
            ),
        }

    @pytest.fixture
    def expected_eval_batch(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                [
                    [
                        101,
                        2023,
                        2003,
                        1996,
                        2034,
                        3793,
                        2005,
                        5604,
                        1012,
                        102,
                        101,
                        2023,
                        3793,
                        3397,
                        2048,
                        11746,
                        1012,
                        102,
                    ],
                    [
                        101,
                        2023,
                        2003,
                        1996,
                        2117,
                        3793,
                        2005,
                        5604,
                        1012,
                        102,
                        101,
                        2023,
                        3793,
                        3397,
                        2048,
                        11746,
                        1012,
                        102,
                    ],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            "token_type_ids": torch.tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        }

    def test_train_call(
        self,
        train_data_collator: EncoderDecoderDataCollatorWithPadding,
        encoded_data_abs: List[Dict[str, List[int]]],
        expected_train_batch: Dict[str, torch.Tensor],
    ) -> None:
        batch = train_data_collator(encoded_data_abs)
        assert len(batch) == len(expected_train_batch)
        for k in batch:
            assert (batch[k] == expected_train_batch[k]).all(), k

    def test_eval_call(
        self,
        eval_data_collator: EncoderDecoderDataCollatorWithPadding,
        encoded_data_abs: List[Dict[str, List[int]]],
        expected_eval_batch: Dict[str, torch.Tensor],
    ) -> None:
        batch = eval_data_collator(encoded_data_abs)
        assert len(batch) == len(expected_eval_batch)
        for k in batch:
            assert (batch[k] == expected_eval_batch[k]).all(), k
