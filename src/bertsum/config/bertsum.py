from copy import deepcopy

from transformers import BertConfig, BertTokenizer, EncoderDecoderConfig


class BertSumExtConfig(BertConfig):
    def __init__(
        self,
        bertsum_pretrained_model_name_or_path: str = 'bert-base-uncased',
        **kwargs
    ):
        config = BertConfig.from_pretrained(bertsum_pretrained_model_name_or_path) \
                           .to_dict()
        config.update(kwargs)

        super().__init__(**config)

        tokenizer = BertTokenizer.from_pretrained(
            bertsum_pretrained_model_name_or_path)

        self.bertsum_pretrained_model_name_or_path = bertsum_pretrained_model_name_or_path
        self.cls_token_id = tokenizer.cls_token_id


class BertSumAbsConfig(EncoderDecoderConfig):
    def __init__(
        self,
        bertsum_pretrained_model_name_or_path: str = 'bert-base-uncased',
        **kwargs
    ):
        encoder_config = BertConfig.from_pretrained(bertsum_pretrained_model_name_or_path) \
                                   .to_dict()
        decoder_config = deepcopy(encoder_config)
        decoder_config.update(kwargs)
        decoder_config['is_decoder'] = True
        decoder_config['add_cross_attention'] = True

        super().__init__(encoder=encoder_config, decoder=decoder_config)
        self.bertsum_pretrained_model_name_or_path = bertsum_pretrained_model_name_or_path
