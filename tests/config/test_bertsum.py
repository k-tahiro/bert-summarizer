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
