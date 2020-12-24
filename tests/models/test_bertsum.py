import pytest

from bert_summarizer.config import BertSumExtConfig, BertSumAbsConfig
from bert_summarizer.models.bertsum import BertSumExt, BertSumAbsDecoder, BertSumAbs


class TestBertSumExt:
    pass


class TestBertSumAbsDecoder:
    @pytest.fixture
    def config(self):
        return BertSumAbsConfig().decoder

    @pytest.fixture
    def model(self, config):
        return BertSumAbsDecoder(config)

    def test_get_input_embeddings(self, model):
        model.decoder.embeddings.make_embedding.emb_luts[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model):
        model.set_input_embeddings(None)
        assert model.decoder.embeddings.make_embedding.emb_luts[0] \
            is None

    def test_get_output_embeddings(self, model):
        model.generator[0] = None
        assert model.get_output_embeddings() is None

    def test_embeddings_weight(self, config, model):
        assert id(model.get_input_embeddings().weight) \
            == id(model.get_output_embeddings().weight)

        input_embeddings = model.get_input_embeddings()
        assert input_embeddings.embedding_dim == config.hidden_size
        assert input_embeddings.num_embeddings == config.vocab_size

    def test_network_structure(self, config, model):
        pass


class TestBertSumAbs:
    pass
