from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoModel
import torch
from torch import nn

from .common import PositionalEncoding
from .decoders.transformer import TransformerDecoder

logger = getLogger(__name__)


class BertSum(nn.Module):
    def __init__(self, model_type: str):
        super(BertSum, self).__init__()
        self.model_type = model_type


class BertSumExt(BertSum):
    def __init__(self,
                 model_type: str,
                 cls_token_id: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 2,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 bias: bool = True):
        super(BertSumExt, self).__init__(model_type)

        # sentence embedding layer
        self.bert = AutoModel.from_pretrained(model_type)
        hidden_size = self.bert.config.hidden_size
        self.cls_token_id = cls_token_id  # cls_token_id is not contained in BertConfig

        # inter-sentence contextual embedding layer
        self.pos_emb = PositionalEncoding(dropout, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        if norm is None and eps:
            norm = nn.LayerNorm(hidden_size, eps=eps)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers,
                                             norm=norm)

        # classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self,
                src: Dict[str, torch.Tensor],
                cls_idxs: Union[None, List[List[int]], torch.Tensor] = None):
        if cls_idxs is None:
            cls_idxs = src['input_ids'] == self.cls_token_id

        x = self.bert(**src)[0]
        x = self.pos_emb(x)

        x = x.permute(1, 0, 2).contiguous()
        x = self.encoder(x, src_key_padding_mask=cls_idxs)
        x = x.permute(1, 0, 2).contiguous()

        x = self.classifier(x)
        return x


class BertSumAbs(BertSum):
    def __init__(self,
                 model_type: str,
                 vocab_size: Optional[int] = None,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None,
                 eps: float = 1e-6,
                 bias: bool = False):
        super(BertSumAbs, self).__init__(model_type)

        # encoder
        self.encoder = AutoModel.from_pretrained(model_type)
        hidden_size = self.encoder.config.hidden_size

        # decoder
        vocab_size = vocab_size or self.encoder.config.vocab_size
        self.decoder = TransformerDecoder(vocab_size,
                                          hidden_size,
                                          self.encoder.config.pad_token_id,
                                          nhead,
                                          dim_feedforward,
                                          dropout,
                                          activation,
                                          num_layers,
                                          norm,
                                          eps,
                                          bias)

        self.loss = nn.NLLLoss(ignore_index=self.encoder.config.pad_token_id,
                               reduction='sum')

    def forward(self,
                src_input_ids: torch.Tensor,
                src_token_type_ids: torch.Tensor,
                src_attention_mask: torch.Tensor,
                tgt_input_ids: torch.Tensor,
                tgt_token_type_ids: torch.Tensor,
                tgt_attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        src = {
            'input_ids': src_input_ids,
            'token_type_ids': src_token_type_ids,
            'attention_mask': src_attention_mask
        }
        tgt = {
            'input_ids': tgt_input_ids,
            'token_type_ids': tgt_token_type_ids,
            'attention_mask': tgt_attention_mask
        }

        # encoder -> decoder
        logits = self._decode(tgt, self._encode(src))

        if labels is not None:
            loss = self._calc_loss(logits[:, :-1].contiguous(), labels)
            return loss, logits
        else:
            return logits

    def _encode(self, src: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory = self.encoder(**src)[0]
        memory_key_padding_mask = src['attention_mask'] == 0
        return {
            # sequence x batch x embedding
            'memory': memory.permute(1, 0, 2).contiguous(),
            'memory_key_padding_mask': memory_key_padding_mask
        }

    def _decode(self,
                tgt: Dict[str, torch.Tensor],
                decoder_args: Dict[str, torch.Tensor]) -> torch.Tensor:
        tgt_key_padding_mask = tgt['attention_mask'] == 0
        return self.decoder(tgt['input_ids'],
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            **decoder_args)

    def _calc_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # batch x sequence x embedding -> batch_sequence x embedding
        output = output.view(-1, output.size(-1))

        # batch x sequence -> batch_sequence
        target = labels.view(-1)

        return self.loss(output, target)
