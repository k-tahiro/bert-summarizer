import os

import pytest
import torch

from bert_summarizer.config import BertSumExtConfig, BertSumAbsConfig
from bert_summarizer.models.bertsum import BertSumExt, BertSumAbsDecoder, BertSumAbs


skip_on_ga = pytest.mark.skipif(
    os.getenv('TEST_ENVIRONMENT') == 'GitHub Actions',
    reason='Skip unittest to save memory'
)


class TestBertSumExt:
    @pytest.fixture
    def config(self):
        return BertSumExtConfig()

    @pytest.fixture
    def model(self, config):
        return BertSumExt(config)

    def test_network_structure(self, config, model):
        assert len(model.encoder.layers) == config.encoder.num_hidden_layers
        assert model.encoder.norm.normalized_shape[0] == config.hidden_size
        assert model.encoder.norm.eps == config.encoder.layer_norm_eps

        encoder_layer = model.encoder.layers[0]
        assert encoder_layer.self_attn.embed_dim == config.hidden_size
        assert encoder_layer.self_attn.num_heads == config.encoder.num_attention_heads
        assert encoder_layer.self_attn.dropout == config.encoder.attention_probs_dropout_prob
        assert encoder_layer.linear1.in_features == config.hidden_size
        assert encoder_layer.linear1.out_features == config.encoder.intermediate_size

        assert model.classifier.in_features == config.hidden_size
        assert model.classifier.out_features == 1

    @skip_on_ga
    @pytest.mark.parametrize('cls_mask,labels,return_dict,expected_len', [
        (
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
            None,
            None,
            1
        ),
        (
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
            None,
            True,
            3
        ),
        (
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            None,
            2
        ),
        (
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
            torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            True,
            4
        ),
    ])
    def test_forward(self, config, model, cls_mask, labels, return_dict, expected_len):
        batch_size = 2
        input_size = 18
        input_ids = torch.tensor([
            [101, 2023, 2003, 1996, 2034, 3793, 2005, 5604, 1012,
                102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
            [101, 2023, 2003, 1996, 2117, 3793, 2005, 5604, 1012,
                102, 101, 2023, 3793, 3397, 2048, 11746, 1012, 102],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        token_type_ids = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ])

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


class TestBertSumAbsDecoder:
    @pytest.fixture
    def config(self):
        return BertSumAbsConfig().decoder

    @pytest.fixture
    def model(self, config):
        return BertSumAbsDecoder(config)

    def test_get_input_embeddings(self, model):
        model.embeddings[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model):
        model.set_input_embeddings(None)
        assert model.embeddings[0] is None

    def test_get_output_embeddings(self, model):
        model.generator[0] = None
        assert model.get_output_embeddings() is None

    def test_embeddings_weight(self, config, model):
        assert id(model.get_input_embeddings().weight) \
            == id(model.get_output_embeddings().weight)

        input_embeddings = model.get_input_embeddings()
        assert input_embeddings.embedding_dim == config.hidden_size
        assert input_embeddings.num_embeddings == config.vocab_size

    def test_network_structure(self, config, model):
        assert len(model.decoder.layers) == config.num_hidden_layers
        assert model.decoder.norm.normalized_shape[0] == config.hidden_size
        assert model.decoder.norm.eps == config.layer_norm_eps

        decoder_layer = model.decoder.layers[0]
        assert decoder_layer.self_attn.embed_dim == config.hidden_size
        assert decoder_layer.self_attn.num_heads == config.num_attention_heads
        assert decoder_layer.self_attn.dropout == config.attention_probs_dropout_prob
        assert decoder_layer.linear1.in_features == config.hidden_size
        assert decoder_layer.linear1.out_features == config.intermediate_size

        assert model.generator[0].in_features == config.hidden_size
        assert model.generator[0].out_features == config.vocab_size

    @skip_on_ga
    @pytest.mark.parametrize('labels,return_dict,expected_len', [
        (None, None, 5),
        (None, True, 1),
        (True, None, 6),
        (True, True, 2),
    ])
    def test_forward(self, config, model, labels, return_dict, expected_len):
        batch_size = 32
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        src_len = 512
        tgt_len = 64

        input_ids = torch.randint(
            vocab_size,
            (batch_size, tgt_len),
            dtype=torch.long
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones(
                (batch_size, tgt_len),
                dtype=torch.long
            ),
            encoder_input_ids=torch.randint(
                vocab_size,
                (batch_size, src_len),
                dtype=torch.long
            ),
            encoder_hidden_states=torch.rand(
                (batch_size, src_len, hidden_size),
            ),
            encoder_attention_mask=torch.ones(
                (batch_size, src_len),
                dtype=torch.long
            ),
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
    def config(self):
        return BertSumAbsConfig(vocab_size=32003)

    @pytest.fixture
    def model(self, config):
        return BertSumAbs(config)

    def test_embeddings_weight(self, model):
        enc_input_embeddings_weight = model.encoder.get_input_embeddings().weight
        dec_input_embeddings_weight = model.decoder.get_input_embeddings().weight
        dec_output_embeddings_weight = model.decoder.get_output_embeddings().weight

        enc_num_tokens = enc_input_embeddings_weight.size(0)
        dec_num_tokens = dec_input_embeddings_weight.size(0)
        num_shared_tokens = min(enc_num_tokens, dec_num_tokens)

        assert (enc_input_embeddings_weight[:num_shared_tokens, :] ==
                dec_input_embeddings_weight[:num_shared_tokens, :]).all()

        assert id(dec_input_embeddings_weight) \
            == id(dec_output_embeddings_weight)

    @pytest.mark.parametrize('input_ids,kwargs,expected_update', [
        (torch.tensor([0, 1, 2]), dict(), dict()),
        (
            torch.tensor([0, 1, 2]),
            dict(decoder_encoder_input_ids=torch.tensor([0, 1, 2])),
            dict(decoder_encoder_input_ids=torch.tensor([0, 1, 2]))
        ),
        (
            torch.tensor([0, 1, 2]),
            dict(invalid_arg=torch.tensor([0, 1, 2])),
            dict()
        ),
    ])
    def test_prepare_inputs_for_generation(self, model, input_ids, kwargs, expected_update):
        expected = super(
            BertSumAbs, model
        ).prepare_inputs_for_generation(input_ids, **kwargs)
        expected.update(expected_update)

        input_dict = model.prepare_inputs_for_generation(input_ids, **kwargs)

        assert input_dict.keys() == expected.keys()
        for k in input_dict:
            a = input_dict[k]
            b = expected[k]
            if a is None:
                assert a is b is None
            else:
                assert torch.all(a.eq(b))
