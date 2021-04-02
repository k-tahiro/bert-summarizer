import pytest
from torch import nn

from bert_summarizer.utils.nn import get_n_params


@pytest.mark.parametrize("model, expected", [(nn.Linear(10, 10), 110)])
def test_get_n_params(model: nn.Module, expected: int) -> None:
    assert get_n_params(model) == expected
