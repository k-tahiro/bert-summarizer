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

    def test_get_input_embeddings(self, model):
        model.decoder.embeddings.make_embedding.emb_luts[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model):
        model.set_input_embeddings(None)
        assert model.decoder.embeddings.make_embedding.emb_luts[0] \
            is None

    def test_embeddings_weight(self, model):
        assert id(model.get_input_embeddings().weight) \
            == id(model.get_output_embeddings().weight)

    def test_network_structure(self, config, model):
        pass


class TestBertSumAbs:
    pass
