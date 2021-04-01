import pytest

from bert_summarizer.utils.common import reduce_dict


@pytest.mark.parametrize(
    "f, a, b, expected",
    [(lambda x, y: x + y, {"k": [0, 1]}, {"k": [2, 3]}, {"k": [0, 1, 2, 3]})],
)
def test_reduce_dict(f, a, b, expected):
    assert reduce_dict(f, a, b) == expected
