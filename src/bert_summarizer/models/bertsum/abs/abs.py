import warnings
from logging import getLogger
from typing import Optional

from torch import nn
from transformers import BertModel, BertPreTrainedModel, EncoderDecoderModel

from ....config import BertSumAbsConfig
from .decoder import BertSumAbsDecoder

try:
    from .onmt_decoder import BertSumAbsOpenNMTDecoder
except ModuleNotFoundError:
    warnings.warn("Failed to import BertSumAbsOpenNMTDecoder")

logger = getLogger(__name__)


class BertSumAbs(EncoderDecoderModel):
    config_class = BertSumAbsConfig

    def __init__(
        self,
        config: Optional[BertSumAbsConfig] = None,
        encoder: Optional[BertPreTrainedModel] = None,
        decoder: Optional[BertPreTrainedModel] = None,
    ):
        if config is not None:
            if encoder is None:
                encoder = BertModel.from_pretrained(config.encoder_model_name_or_path)
            if decoder is None:
                if config.use_onmt_transformer:
                    decoder = BertSumAbsOpenNMTDecoder(config.decoder)
                else:
                    decoder = BertSumAbsDecoder(config.decoder)

        super().__init__(config=config, encoder=encoder, decoder=decoder)

        logger.debug(f"self.config={self.config}")

        decoder_embeddings = self.encoder._get_resized_embeddings(
            nn.Embedding.from_pretrained(
                self.encoder.get_input_embeddings().weight,
                freeze=False,
                padding_idx=self.config.encoder.pad_token_id,
            ),
            self.config.decoder.vocab_size,
        )
        self.decoder.set_input_embeddings(decoder_embeddings)
        self.decoder.tie_weights()
