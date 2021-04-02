from typing import Optional

import pytest
import torch
from transformers import BertConfig

from bert_summarizer.config import BertSumAbsConfig
from bert_summarizer.models.bertsum import (
    BertSumAbs,
    BertSumAbsDecoder,
    BertSumAbsOpenNMTDecoder,
)

from .util import skip_on_ga


class TestBertSumAbsOpenNMTDecoder:
    @pytest.fixture
    def config(self) -> BertConfig:
        return BertSumAbsConfig(smoothing=0.1).decoder

    @pytest.fixture
    def model(self, config: BertConfig) -> BertSumAbsOpenNMTDecoder:
        return BertSumAbsOpenNMTDecoder(config)

    def test_get_input_embeddings(self, model: BertSumAbsOpenNMTDecoder) -> None:
        model.decoder.embeddings.make_embedding.emb_luts[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model: BertSumAbsOpenNMTDecoder) -> None:
        model.set_input_embeddings(None)
        assert model.decoder.embeddings.make_embedding.emb_luts[0] is None

    def test_get_output_embeddings(self, model: BertSumAbsOpenNMTDecoder) -> None:
        model.generator = None
        assert model.get_output_embeddings() is None

    def test_embeddings_weight(
        self, config: BertConfig, model: BertSumAbsOpenNMTDecoder
    ) -> None:
        assert id(model.get_input_embeddings().weight) == id(
            model.get_output_embeddings().weight
        )

        input_embeddings = model.get_input_embeddings()
        assert input_embeddings.embedding_dim == config.hidden_size
        assert input_embeddings.num_embeddings == config.vocab_size

    def test_network_structure(
        self, config: BertConfig, model: BertSumAbsOpenNMTDecoder
    ) -> None:
        assert len(model.decoder.transformer_layers) == config.num_hidden_layers

        transformer = model.decoder.transformer_layers[0]
        dim_per_head = config.hidden_size // config.num_attention_heads
        attn_hidden_size = config.num_attention_heads * dim_per_head

        self_attn = transformer.self_attn
        assert self_attn.dim_per_head == dim_per_head
        assert self_attn.model_dim == config.hidden_size
        assert self_attn.head_count == config.num_attention_heads
        assert self_attn.linear_keys.in_features == config.hidden_size
        assert self_attn.linear_keys.out_features == attn_hidden_size
        assert self_attn.linear_values.in_features == config.hidden_size
        assert self_attn.linear_values.out_features == attn_hidden_size
        assert self_attn.linear_query.in_features == config.hidden_size
        assert self_attn.linear_query.out_features == attn_hidden_size
        assert self_attn.dropout.p == config.attention_probs_dropout_prob
        assert self_attn.final_linear.in_features == config.hidden_size
        assert self_attn.final_linear.out_features == config.hidden_size

        context_attn = transformer.context_attn
        assert context_attn.dim_per_head == dim_per_head
        assert context_attn.model_dim == config.hidden_size
        assert context_attn.head_count == config.num_attention_heads
        assert context_attn.linear_keys.in_features == config.hidden_size
        assert context_attn.linear_keys.out_features == attn_hidden_size
        assert context_attn.linear_values.in_features == config.hidden_size
        assert context_attn.linear_values.out_features == attn_hidden_size
        assert context_attn.linear_query.in_features == config.hidden_size
        assert context_attn.linear_query.out_features == attn_hidden_size
        assert context_attn.dropout.p == config.attention_probs_dropout_prob
        assert context_attn.final_linear.in_features == config.hidden_size
        assert context_attn.final_linear.out_features == config.hidden_size

        feed_forward = transformer.feed_forward
        assert feed_forward.w_1.in_features == config.hidden_size
        assert feed_forward.w_1.out_features == config.intermediate_size
        assert feed_forward.w_2.in_features == config.intermediate_size
        assert feed_forward.w_2.out_features == config.hidden_size
        assert feed_forward.layer_norm.normalized_shape[0] == config.hidden_size
        assert feed_forward.dropout_1.p == config.hidden_dropout_prob
        assert feed_forward.dropout_2.p == config.hidden_dropout_prob

        layer_norm_1 = transformer.layer_norm_1
        assert layer_norm_1.normalized_shape[0] == config.hidden_size
        layer_norm_2 = transformer.layer_norm_2
        assert layer_norm_2.normalized_shape[0] == config.hidden_size
        drop = transformer.drop
        assert drop.p == config.hidden_dropout_prob

    def test_loss(self, config: BertConfig, model: BertSumAbsOpenNMTDecoder) -> None:
        assert model.loss.cls == config.vocab_size
        assert model.loss.smoothing == config.smoothing

    @skip_on_ga
    @pytest.mark.parametrize(
        "labels,return_dict,expected_len",
        [
            (None, None, 5),
            (None, True, 1),
            (True, None, 6),
            (True, True, 2),
        ],
    )
    def test_forward(
        self,
        config: BertConfig,
        model: BertSumAbsOpenNMTDecoder,
        labels: Optional[bool],
        return_dict: Optional[bool],
        expected_len: int,
    ) -> None:
        batch_size = 32
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        src_len = 512
        tgt_len = 64

        input_ids = torch.randint(vocab_size, (batch_size, tgt_len), dtype=torch.long)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones((batch_size, tgt_len), dtype=torch.long),
            encoder_hidden_states=torch.rand(
                (batch_size, src_len, hidden_size),
            ),
            encoder_attention_mask=torch.ones((batch_size, src_len), dtype=torch.long),
            labels=input_ids if labels else None,
            return_dict=return_dict,
        )

        assert len(outputs) == expected_len

        loss = logits = None
        if return_dict:
            loss = outputs.loss
            logits = outputs.logits
        else:
            if labels:
                loss, logits, _, _, _, _ = outputs
            else:
                logits = outputs[0]

        if loss is not None:
            assert isinstance(loss.item(), float)

        assert len(logits.size()) == 3
        assert logits.size(0) == batch_size
        assert logits.size(1) == tgt_len
        assert logits.size(2) == vocab_size


class TestBertSumAbsDecoder:
    @pytest.fixture
    def config(self) -> BertConfig:
        return BertSumAbsConfig(smoothing=0.1).decoder

    @pytest.fixture
    def model(self, config: BertConfig) -> BertSumAbsDecoder:
        return BertSumAbsDecoder(config)

    def test_get_input_embeddings(self, model: BertSumAbsDecoder) -> None:
        model.embeddings[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model: BertSumAbsDecoder) -> None:
        model.set_input_embeddings(None)
        assert model.embeddings[0] is None

    def test_get_output_embeddings(self, model: BertSumAbsDecoder) -> None:
        model.generator = None
        assert model.get_output_embeddings() is None

    def test_embeddings_weight(
        self, config: BertConfig, model: BertSumAbsDecoder
    ) -> None:
        assert id(model.get_input_embeddings().weight) == id(
            model.get_output_embeddings().weight
        )

        input_embeddings = model.get_input_embeddings()
        assert input_embeddings.embedding_dim == config.hidden_size
        assert input_embeddings.num_embeddings == config.vocab_size

    def test_network_structure(
        self, config: BertConfig, model: BertSumAbsDecoder
    ) -> None:
        assert len(model.decoder.layers) == config.num_hidden_layers
        assert model.decoder.norm.normalized_shape[0] == config.hidden_size
        assert model.decoder.norm.eps == config.layer_norm_eps

        decoder_layer = model.decoder.layers[0]
        assert decoder_layer.self_attn.embed_dim == config.hidden_size
        assert decoder_layer.self_attn.num_heads == config.num_attention_heads
        assert decoder_layer.self_attn.dropout == config.attention_probs_dropout_prob
        assert decoder_layer.linear1.in_features == config.hidden_size
        assert decoder_layer.linear1.out_features == config.intermediate_size

        assert model.generator.in_features == config.hidden_size
        assert model.generator.out_features == config.vocab_size

    def test_loss(self, config: BertConfig, model: BertSumAbsDecoder) -> None:
        assert model.loss.cls == config.vocab_size
        assert model.loss.smoothing == config.smoothing

    @skip_on_ga
    @pytest.mark.parametrize(
        "labels,return_dict,expected_len",
        [
            (None, None, 5),
            (None, True, 1),
            (True, None, 6),
            (True, True, 2),
        ],
    )
    def test_forward(
        self,
        config: BertConfig,
        model: BertSumAbsDecoder,
        labels: Optional[bool],
        return_dict: Optional[bool],
        expected_len: int,
    ) -> None:
        batch_size = 32
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        src_len = 512
        tgt_len = 64

        input_ids = torch.randint(vocab_size, (batch_size, tgt_len), dtype=torch.long)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones((batch_size, tgt_len), dtype=torch.long),
            encoder_hidden_states=torch.rand(
                (batch_size, src_len, hidden_size),
            ),
            encoder_attention_mask=torch.ones((batch_size, src_len), dtype=torch.long),
            labels=input_ids if labels else None,
            return_dict=return_dict,
        )

        assert len(outputs) == expected_len

        loss = logits = None
        if return_dict:
            loss = outputs.loss
            logits = outputs.logits
        else:
            if labels:
                loss, logits, _, _, _, _ = outputs
            else:
                logits = outputs[0]

        if loss is not None:
            assert isinstance(loss.item(), float)

        assert len(logits.size()) == 3
        assert logits.size(0) == batch_size
        assert logits.size(1) == tgt_len
        assert logits.size(2) == vocab_size


class TestBertSumAbs:
    @pytest.fixture
    def config(self) -> BertSumAbsConfig:
        return BertSumAbsConfig(vocab_size=32003)

    @pytest.fixture
    def model(self, config: BertSumAbsConfig) -> BertSumAbs:
        return BertSumAbs(config)

    def test_embeddings_weight(self, model: BertSumAbs) -> None:
        enc_input_embeddings_weight = model.encoder.get_input_embeddings().weight
        dec_input_embeddings_weight = model.decoder.get_input_embeddings().weight
        dec_output_embeddings_weight = model.decoder.get_output_embeddings().weight

        enc_num_tokens = enc_input_embeddings_weight.size(0)
        dec_num_tokens = dec_input_embeddings_weight.size(0)
        num_shared_tokens = min(enc_num_tokens, dec_num_tokens)

        assert (
            enc_input_embeddings_weight[:num_shared_tokens, :]
            == dec_input_embeddings_weight[:num_shared_tokens, :]
        ).all()

        assert id(dec_input_embeddings_weight) == id(dec_output_embeddings_weight)
