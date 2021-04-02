from typing import Optional

import pytest
import torch

from bert_summarizer.config import BertSumExtConfig
from bert_summarizer.models import BertSumExt

from .util import skip_on_ga


class TestBertSumExt:
    @pytest.fixture
    def config(self) -> BertSumExtConfig:
        return BertSumExtConfig()

    @pytest.fixture
    def model(self, config: BertSumExtConfig) -> BertSumExt:
        return BertSumExt(config)

    def test_network_structure(
        self, config: BertSumExtConfig, model: BertSumExt
    ) -> None:
        assert len(model.encoder.layers) == config.encoder.num_hidden_layers
        assert model.encoder.norm.normalized_shape[0] == config.hidden_size
        assert model.encoder.norm.eps == config.encoder.layer_norm_eps

        encoder_layer = model.encoder.layers[0]
        assert encoder_layer.self_attn.embed_dim == config.hidden_size
        assert encoder_layer.self_attn.num_heads == config.encoder.num_attention_heads
        assert (
            encoder_layer.self_attn.dropout
            == config.encoder.attention_probs_dropout_prob
        )
        assert encoder_layer.linear1.in_features == config.hidden_size
        assert encoder_layer.linear1.out_features == config.encoder.intermediate_size

        assert model.classifier.in_features == config.hidden_size
        assert model.classifier.out_features == 1

    @skip_on_ga
    @pytest.mark.parametrize(
        "cls_mask,labels,return_dict,expected_len",
        [
            (
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                None,
                None,
                1,
            ),
            (
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                None,
                True,
                3,
            ),
            (
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                None,
                2,
            ),
            (
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                torch.tensor(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                True,
                4,
            ),
        ],
    )
    def test_forward(
        self,
        config: BertSumExtConfig,
        model: BertSumExt,
        cls_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        return_dict: Optional[bool],
        expected_len: int,
    ) -> None:
        batch_size = 2
        input_size = 18
        input_ids = torch.tensor(
            [
                [
                    101,
                    2023,
                    2003,
                    1996,
                    2034,
                    3793,
                    2005,
                    5604,
                    1012,
                    102,
                    101,
                    2023,
                    3793,
                    3397,
                    2048,
                    11746,
                    1012,
                    102,
                ],
                [
                    101,
                    2023,
                    2003,
                    1996,
                    2117,
                    3793,
                    2005,
                    5604,
                    1012,
                    102,
                    101,
                    2023,
                    3793,
                    3397,
                    2048,
                    11746,
                    1012,
                    102,
                ],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        token_type_ids = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            cls_mask=cls_mask,
            labels=labels,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict,
        )

        assert len(outputs) == expected_len

        loss = logits = None
        if return_dict:
            loss = outputs.loss
            logits = outputs.logits
        else:
            if labels is not None:
                loss, logits = outputs
            else:
                logits = outputs[0]

        if loss is not None:
            assert isinstance(loss.item(), float)

        assert len(logits.size()) == 2
        assert logits.size(0) == batch_size
        assert logits.size(1) == input_size
