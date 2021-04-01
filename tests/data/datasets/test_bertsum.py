from typing import Dict, List

import pytest

from bert_summarizer.data.datasets import BertSumAbsDataset, BertSumExtDataset


@pytest.fixture
def model_name() -> str:
    return "bert-base-uncased"


@pytest.fixture
def src() -> List[str]:
    return [
        "This is the first text for testing. This text contains two sentences.",
        "This is the second text for testing. This text contains two sentences.",
    ]


@pytest.fixture
def tgt_ext() -> List[List[str]]:
    return [
        ["This is the first text for testing."],
        ["This is the second text for testing."],
    ]


@pytest.fixture
def encoded_data_ext() -> List[Dict[str, List[int]]]:
    return [
        {
            "input_ids": [
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
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "cls_mask": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "label": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        {
            "input_ids": [
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
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "cls_mask": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "label": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
    ]


@pytest.fixture
def tgt_abs() -> List[str]:
    return [
        "First test text",
        "Second test text",
    ]


@pytest.fixture
def encoded_data_abs() -> List[Dict[str, List[int]]]:
    return [
        {
            "input_ids": [
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
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "decoder_input_ids": [2, 2034, 3231, 3793, 3],
            "decoder_token_type_ids": [0, 0, 0, 0, 0],
        },
        {
            "input_ids": [
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
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "decoder_input_ids": [2, 2117, 3231, 3793, 3],
            "decoder_token_type_ids": [0, 0, 0, 0, 0],
        },
    ]


class TestBertSumExtDataset:
    @pytest.fixture
    def bertsum_ext_dataset(
        self, model_name: str, src: List[str], tgt_ext: List[List[str]]
    ) -> BertSumExtDataset:
        return BertSumExtDataset(model_name, src, tgt_ext)

    def test_data(
        self,
        bertsum_ext_dataset: BertSumExtDataset,
        encoded_data_ext: List[Dict[str, List[int]]],
    ) -> None:
        assert bertsum_ext_dataset.data == encoded_data_ext


class TestBertSumAbsDataset:
    @pytest.fixture
    def bertsum_abs_dataset(
        self, model_name: str, src: List[str], tgt_abs: List[str]
    ) -> BertSumAbsDataset:
        return BertSumAbsDataset(model_name, src, tgt_abs)

    def test_data(
        self,
        bertsum_abs_dataset: BertSumAbsDataset,
        encoded_data_abs: List[Dict[str, List[int]]],
    ) -> None:
        assert bertsum_abs_dataset.data == encoded_data_abs
