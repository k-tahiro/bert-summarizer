from logging import getLogger
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertLMHeadModel,
    EncoderDecoderModel
)
from transformers.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertOnlyMLMHead
)

from ..config import BertSumExtConfig, BertSumAbsConfig

logger = getLogger(__name__)


class BertSumExt(BertPreTrainedModel):
    config_class = BertSumExtConfig

    def __init__(self, config: BertSumExtConfig):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(
            config.pretrained_model_name_or_path)

        self.encoder = nn.Sequential(
            BertEncoder(config),
            BertPooler(config)
        )
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(self,
                src: Dict[str, torch.Tensor],
                cls_idxs: Union[None, List[List[int]], torch.Tensor] = None):
        if cls_idxs is None:
            cls_idxs = src['input_ids'] == self.config.cls_token_id

        x = self.bert(**src)[0]
        x = self.encoder(x, encoder_attention_mask=cls_idxs)
        x = self.cls(x)
        return x


class BertSumAbs(EncoderDecoderModel):
    config_class = BertSumAbsConfig

    def __init__(
        self,
        config: Optional[BertSumAbsConfig] = None,
        encoder: Optional[BertPreTrainedModel] = None,
        decoder: Optional[BertPreTrainedModel] = None
    ):
        if config is not None:
            encoder = BertModel.from_pretrained(
                config.pretrained_model_name_or_path)
            decoder = BertLMHeadModel(config.decoder)

            super().__init__(encoder=encoder, decoder=decoder)
        else:
            super().__init__(config=config, encoder=encoder, decoder=decoder)

        # TODO: copy encoder emmbedding layer to decoder embedding and Linear layer
