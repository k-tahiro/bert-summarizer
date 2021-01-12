import pytest
from transformers import AutoTokenizer

from bert_summarizer.utils.bertsum import BertRouge, GreedySelector
from ..data.datasets.test_bertsum import src, tgt_ext, tgt_abs


@pytest.fixture
def sents_src():
    return [
        [
            'This is the first text for testing.',
            'This text contains two sentences.'
        ],
        [
            'This is the second text for testing.',
            'This text contains two sentences.'
        ],
    ]


@pytest.fixture
def sents_tgt():
    return [
        [
            'First test text',
        ],
        [
            'Second test text',
        ],
    ]


class TestBertRouge:
    @pytest.fixture
    def bert_rouge(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return BertRouge(tokenizer)

    def test_call(self, bert_rouge, src, tgt_abs):
        expected = [
            {
                'rouge-1': {
                    'p': 2 / 11,
                    'r': 2 / 3,
                    'f': 2 / 7
                },
                'rouge-2': {
                    'p': 0.0,
                    'r': 0.0,
                    'f': 0.0
                }
            },
            {
                'rouge-1': {
                    'p': 2 / 11,
                    'r': 2 / 3,
                    'f': 2 / 7
                },
                'rouge-2': {
                    'p': 0.0,
                    'r': 0.0,
                    'f': 0.0
                }
            },
        ]

        for a, b in zip(bert_rouge(src, tgt_abs), expected):
            for k in ['rouge-1', 'rouge-2']:
                assert a[k] == pytest.approx(b[k])


class TestGreedySelector:
    @pytest.fixture
    def greedy_selector(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        return GreedySelector(tokenizer)

    def test_call(self, greedy_selector, sents_src, sents_tgt, tgt_ext):
        for src, tgt, expected in zip(sents_src, sents_tgt, tgt_ext):
            assert greedy_selector(src, tgt) == expected
