import pytest
from transformers import AutoConfig

from bert_summarizer.config.bertsum import BertSumExtConfig, BertSumAbsConfig


@pytest.fixture
def default_model_name_or_path():
    return 'bert-base-uncased'


@pytest.fixture
def default_bert_config():
    return AutoConfig.from_pretrained('bert-base-uncased').to_dict()


@pytest.fixture
def default_ext_encoder_config():
    return dict(
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12
    )


@pytest.fixture
def default_decoder_config():
    config = AutoConfig.from_pretrained('bert-base-uncased').to_dict()
    config['is_decoder'] = True
    config['add_cross_attention'] = True
    return config


class TestBertSumExtConfig:
    def test_default_config(
        self,
        default_model_name_or_path,
        default_bert_config,
        default_ext_encoder_config
    ):
        config = BertSumExtConfig()
        config = config.to_dict()

        base_model_name_or_path = config.pop('base_model_name_or_path')
        encoder_config = config.pop('encoder')

        assert config == default_bert_config
        assert base_model_name_or_path == default_model_name_or_path
        assert encoder_config == default_ext_encoder_config

    def test_custom_config(self, default_bert_config):
        config = BertSumExtConfig(encoder=default_bert_config)
        assert config.encoder == default_bert_config

    def test_error_config(self):
        with pytest.raises(AssertionError):
            config = BertSumExtConfig(
                encoder=dict(invalid_key='invalid_value')
            )


class TestBertSumAbsConfig:
    def test_default_config(
        self,
        default_model_name_or_path,
        default_bert_config,
        default_decoder_config
    ):
        config = BertSumAbsConfig()
        assert config.encoder_model_name_or_path == default_model_name_or_path
        assert config.encoder.to_dict() == default_bert_config
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
