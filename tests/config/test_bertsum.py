from typing import Any, Dict

import pytest
from transformers import AutoConfig, BertConfig

from bert_summarizer.config.bertsum import BertSumAbsConfig, BertSumExtConfig


@pytest.fixture
def default_model_name_or_path() -> str:
    return "bert-base-uncased"


@pytest.fixture
def default_bert_config() -> BertConfig:
    return AutoConfig.from_pretrained("bert-base-uncased")


@pytest.fixture
def default_ext_encoder_config() -> BertConfig:
    return BertConfig(
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        initializer_range=0.0,
        xavier_initialization=True,
    )


@pytest.fixture
def default_decoder_config_dict() -> Dict[str, Any]:
    config: Dict[str, Any] = AutoConfig.from_pretrained("bert-base-uncased").to_dict()
    config["is_decoder"] = True
    config["add_cross_attention"] = True
    config["smoothing"] = 0.0
    return config


class TestBertSumExtConfig:
    def test_default_config(
        self,
        default_model_name_or_path: str,
        default_bert_config: BertConfig,
        default_ext_encoder_config: BertConfig,
    ) -> None:
        config = BertSumExtConfig().to_dict()

        base_model_name_or_path = config.pop("base_model_name_or_path")
        encoder_config = config.pop("encoder")

        assert config == default_bert_config.to_dict()
        assert base_model_name_or_path == default_model_name_or_path
        assert encoder_config == default_ext_encoder_config.to_dict()


class TestBertSumAbsConfig:
    def test_default_config(
        self,
        default_model_name_or_path: str,
        default_bert_config: BertConfig,
        default_decoder_config_dict: Dict[str, Any],
    ) -> None:
        config = BertSumAbsConfig()
        assert config.encoder_model_name_or_path == default_model_name_or_path
        assert config.encoder.to_dict() == default_bert_config.to_dict()
        assert config.decoder.to_dict() == default_decoder_config_dict

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=2048,
                hidden_act="gelu",
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.2,
                layer_norm_eps=1e-6,
                smoothing=0.1,
            ),
        ],
    )
    def test_custom_config(self, kwargs: Dict[str, Any]) -> None:
        config = BertSumAbsConfig(**kwargs)
        decoder_config = config.decoder.to_dict()
        for k, v in kwargs.items():
            assert k in decoder_config
            assert decoder_config[k] == v
