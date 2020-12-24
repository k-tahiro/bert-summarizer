import pytest

from bert_summarizer.config import BertSumExtConfig, BertSumAbsConfig
from bert_summarizer.models.bertsum import BertSumExt, BertSumAbsDecoder, BertSumAbs


class TestBertSumExt:
    pass


class TestBertSumAbsDecoder:
    @pytest.fixture
    def config(self):
        return BertSumAbsConfig()

    @pytest.fixture
    def model(self, config):
        return BertSumAbsDecoder(config.decoder)

    def test_network_structure(self, config, model):
        pass

    def test_embeddings_weight(self, model):
        assert id(model.get_input_embeddings().weight) \
            == id(model.get_output_embeddings().weight)


class TestBertSumAbs:
    pass
