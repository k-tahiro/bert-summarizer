import pytest

from bert_summarizer.data.datasets import BertSumExtDataset, BertSumAbsDataset


@pytest.fixture
def model_name():
    return 'bert-base-uncased'


@pytest.fixture
def src():
    return [
        'This is the first text for testing. This text contains two sentences.',
        'This is the second text for testing. This text contains two sentences.',
    ]


@pytest.fixture
def tgt_ext():
    pass


@pytest.fixture
def tgt_abs():
    return [
        'First test text',
        'Second test text',
    ]


@pytest.fixture
def encoded_data_abs():
    return [
        {
            'input_ids': [101, 2023, 2003, 1996, 2034, 3793, 2005, 5604, 1012, 102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            'decoder_input_ids': [2, 2034, 3231, 3793, 3],
            'decoder_token_type_ids': [0, 0, 0, 0, 0],
        },
        {
            'input_ids': [101, 2023, 2003, 1996, 2117, 3793, 2005, 5604, 1012, 102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            'decoder_input_ids': [2, 2117, 3231, 3793, 3],
            'decoder_token_type_ids': [0, 0, 0, 0, 0],
        },
    ]


class TestBertSumExtDataset:
    pass


class TestBertSumAbsDataset:
    @pytest.fixture
    def bertsum_abs_dataset(self, model_name, src, tgt_abs):
        return BertSumAbsDataset(model_name, src, tgt_abs)

    def test_data(self, bertsum_abs_dataset, encoded_data_abs):
        assert bertsum_abs_dataset.data == encoded_data_abs
