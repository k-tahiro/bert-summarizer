from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, Optional

from transformers import BertConfig, BertTokenizer, EncoderDecoderConfig


logger = getLogger(__name__)


class BertSumExtConfig(BertConfig):
    def __init__(
        self,
        base_model_name_or_path: str = 'bert-base-uncased',
        encoder_num_hidden_layers: int = 2,
        encoder_num_attention_heads: int = 8,
        encoder_intermediate_size: int = 2048,
        encoder_hidden_act: str = 'gelu',
        encoder_attention_probs_dropout_prob: float = 0.1,
        encoder_layer_norm_eps: float = 1e-6,
        encoder_initializer_range: float = 0.0,
        encoder_xavier_initialization: bool = True,
        **kwargs
    ):
        config = BertConfig.from_pretrained(base_model_name_or_path) \
                           .to_dict()
        config.update(kwargs)

        encoder_config = BertConfig(
            num_hidden_layers=encoder_num_hidden_layers,
            num_attention_heads=encoder_num_attention_heads,
            intermediate_size=encoder_intermediate_size,
            hidden_act=encoder_hidden_act,
            attention_probs_dropout_prob=encoder_attention_probs_dropout_prob,
            layer_norm_eps=encoder_layer_norm_eps,
            initializer_range=encoder_initializer_range,
            xavier_initialization=encoder_xavier_initialization
        )
        config.update(encoder=encoder_config)

        super().__init__(**config)
        self.base_model_name_or_path = base_model_name_or_path

    def to_dict(self):
        output = super().to_dict()
        output['encoder'] = self.encoder.to_dict()
        return output


class BertSumAbsConfig(EncoderDecoderConfig):
    def __init__(
        self,
        encoder_model_name_or_path: str = 'bert-base-uncased',
        **kwargs
    ):
        if 'encoder' in kwargs:
            encoder_config = kwargs.pop('encoder')
        else:
            encoder_config = BertConfig.from_pretrained(encoder_model_name_or_path) \
                                       .to_dict()

        if 'decoder' in kwargs:
            decoder_config = kwargs.pop('decoder')
        else:
            decoder_config = deepcopy(encoder_config)
            decoder_config.update(kwargs)
        decoder_config['is_decoder'] = True
        decoder_config['add_cross_attention'] = True

        logger.info(f'{encoder_config=}')
        logger.info(f'{decoder_config=}')

        super().__init__(encoder=encoder_config, decoder=decoder_config)
        self.encoder_model_name_or_path = encoder_model_name_or_path
