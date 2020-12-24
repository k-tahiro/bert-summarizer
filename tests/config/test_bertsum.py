import pytest
from transformers import AutoConfig

from bert_summarizer.config.bertsum import BertSumExtConfig, BertSumAbsConfig


@pytest.fixture
def default_encoder_model_name_or_path():
    return 'bert-base-uncased'


@pytest.fixture
def default_encoder_config():
    return AutoConfig.from_pretrained('bert-base-uncased').to_dict()


@pytest.fixture
def default_decoder_config():
    config = AutoConfig.from_pretrained('bert-base-uncased').to_dict()
    config['is_decoder'] = True
    config['add_cross_attention'] = True
    return config


class TestBertSumExtConfig:
    pass


class TestBertSumAbsConfig:
    def test_default_config(
        self,
        default_encoder_model_name_or_path,
        default_encoder_config,
        default_decoder_config
    ):
        config = BertSumAbsConfig()
        assert config.encoder_model_name_or_path == default_encoder_model_name_or_path
        assert config.encoder.to_dict() == default_encoder_config
        assert config.decoder.to_dict() == default_decoder_config

    @pytest.mark.parametrize('kwargs', [
        dict(
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_act='gelu',
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            layer_norm_eps=1e-6,
        ),
    ])
    def test_custom_config(self, kwargs):
        config = BertSumAbsConfig(**kwargs)
        decoder_config = config.decoder.to_dict()
        for k, v in kwargs.items():
            assert k in decoder_config
            assert decoder_config[k] == v
