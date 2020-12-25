import pytest
import torch

from bert_summarizer.config import BertSumExtConfig, BertSumAbsConfig
from bert_summarizer.models.bertsum import BertSumExt, BertSumAbsDecoder, BertSumAbs


class TestBertSumExt:
    @pytest.fixture
    def config(self):
        return BertSumExtConfig()

    @pytest.fixture
    def model(self, config):
        return BertSumExt(config)

    def test_network_structure(self, config, model):
        assert len(model.encoder.layers) == config.encoder.num_hidden_layers


class TestBertSumAbsDecoder:
    @pytest.fixture
    def config(self):
        return BertSumAbsConfig().decoder

    @pytest.fixture
    def model(self, config):
        return BertSumAbsDecoder(config)

    def test_get_input_embeddings(self, model):
        model.decoder.embeddings.make_embedding.emb_luts[0] = None
        assert model.get_input_embeddings() is None

    def test_set_input_embeddings(self, model):
        model.set_input_embeddings(None)
        assert model.decoder.embeddings.make_embedding.emb_luts[0] \
            is None

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

    @pytest.mark.parametrize('labels,return_dict,expected_len', [
        (None, None, 3),
        (None, True, 2),
        (True, None, 4),
        (True, True, 3),
    ])
    def test_forward(self, config, model, labels, return_dict, expected_len):
        batch_size = 32
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        src_len = 512
        tgt_len = 64

        input_ids = torch.randint(vocab_size, (batch_size, tgt_len))
        outputs = model(
            input_ids=input_ids,
            encoder_input_ids=torch.randint(vocab_size, (batch_size, src_len)),
            encoder_hidden_states=torch.rand(
                (batch_size, src_len, hidden_size)
            ),
            encoder_attention_mask=torch.ones((batch_size, src_len)),
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
                loss, logits, _, _ = outputs
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
        return BertSumAbsConfig()

    @pytest.fixture
    def model(self, config):
        return BertSumAbs(config)

    def test_embeddings_weight(self, model):
        assert (model.encoder.get_input_embeddings().weight
                == model.decoder.get_input_embeddings().weight).all()

    @pytest.mark.parametrize('input_ids,kwargs,expected_update', [
        ([0, 1, 2], dict(), dict()),
        (
            [0, 1, 2],
            dict(decoder_encoder_input_ids=[0, 1, 2]),
            dict(decoder_encoder_input_ids=[0, 1, 2])
        ),
        ([0, 1, 2], dict(invalid_arg=[0, 1, 2]), dict()),
    ])
    def test_prepare_inputs_for_generation(self, model, input_ids, kwargs, expected_update):
        input_dict = super(
            BertSumAbs, model
        ).prepare_inputs_for_generation(input_ids, **kwargs)
        input_dict.update(expected_update)

        assert model.prepare_inputs_for_generation(input_ids, **kwargs) \
            == input_dict
