import pytest
import torch

from bert_summarizer.data import EncoderDecoderDataCollatorWithPadding
from bert_summarizer.data.datasets import BertSumAbsDataset

from .datasets.test_bertsum import encoded_data_abs


class TestEncoderDecoderDataCollatorWithPadding:
    @pytest.fixture
    def dataset(self):
        return BertSumAbsDataset('bert-base-uncased', [])

    @pytest.fixture
    def data_collator(self, dataset):
        return EncoderDecoderDataCollatorWithPadding(
            tokenizer=dataset.tokenizer
        )

    @pytest.fixture
    def expected_batch(self):
        return {
            'input_ids': torch.tensor([
                [101, 2023, 2003, 1996, 2034, 3793, 2005, 5604, 1012,
                    102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
                [101, 2023, 2003, 1996, 2117, 3793, 2005, 5604, 1012,
                    102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
            ]),
            'attention_mask': torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            'token_type_ids': torch.tensor([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            'decoder_input_ids': torch.tensor([
                [2, 2034, 3231, 3793, 3],
                [2, 2117, 3231, 3793, 3],
            ]),
            'decoder_attention_mask': torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]),
            'decoder_token_type_ids': torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            'decoder_encoder_input_ids': torch.tensor([
                [101, 2023, 2003, 1996, 2034, 3793, 2005, 5604, 1012,
                    102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
                [101, 2023, 2003, 1996, 2117, 3793, 2005, 5604, 1012,
                    102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
            ]),
        }

    def test_call(self, data_collator, encoded_data_abs, expected_batch):
        batch = data_collator(encoded_data_abs)
        assert len(batch) == len(expected_batch)
        for k in batch:
            assert (batch[k] == expected_batch[k]).all(), k
