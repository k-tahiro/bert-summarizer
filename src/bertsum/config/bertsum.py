from copy import deepcopy

from transformers import BertConfig, EncoderDecoderConfig


class BertSumExtConfig(BertConfig):
    def __init__(self, pretrained_model_name_or_path, cls_token_id, **kwargs):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path) \
                           .to_dict()
        config.update(kwargs)

        super().__init__(**config)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cls_token_id = cls_token_id


class BertSumAbsConfig(EncoderDecoderConfig):
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        encoder_config = BertConfig.from_pretrained(pretrained_model_name_or_path) \
                                   .to_dict()
        decoder_config = deepcopy(encoder_config)
        decoder_config.update(kwargs)
        decoder_config['is_decoder'] = True
        decoder_config['add_cross_attention'] = True

        super().__init__(encoder=encoder_config, decoder=decoder_config)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
