from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from ..datasets.bertsum import BertSumDataset, BertSumExtDataset, BertSumAbsDataset
from ..models.bertsum import BertSum, BertSumExt, BertSumAbs


class BertSumSummarizer:
    Dataset = BertSumDataset

    def __init__(self,
                 model: BertSum,
                 batch_size: Optional[int] = None,
                 device: Optional[torch.device] = None):
        if device:
            self.model.to(device)

        self.model = model
        self.batch_size = batch_size
        self.device = device or model.device

    def __call__(self, src: Union[str, List[str]]):
        data_loader = self._create_data_loader(src)
        with torch.no_grad():
            self._run(data_loader)

    def _create_data_loader(self, src: Union[str, List[str]]) -> DataLoader:
        if isinstance(src, str):
            src = [src]

        batch_size = self.batch_size or len(src)
        tgt = batch_size * ['']  # dummy target
        dataset = self.Dataset(src, tgt, self.model.model_type)
        return DataLoader(dataset, batch_size=batch_size)

    def _run(self, data_loader: DataLoader):
        raise NotImplementedError


class BertSumExtSummarizer(BertSumSummarizer):
    Dataset = BertSumExtDataset


class BertSumAbsSummarizer(BertSumSummarizer):
    Dataset = BertSumAbsDataset

    def __init__(self,
                 model: BertSumAbs,
                 bos_token_id: int,
                 eos_token_id: int,
                 batch_size: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 max_length: Optional[int] = None,
                 min_length: int = 0,
                 alpha: float = 1.,
                 beam_size: int = 5):
        super(BertSumAbsSummarizer, self).__init__(model, batch_size, device)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.beam_size = beam_size
        self.max_length = max_length or self.model.encoder.config.max_position_embeddings
        self.min_length = min_length
        self.alpha = alpha

    def _length_penalty(self, step: int):
        return ((5.0 + (step + 1)) / 6.0) ** self.alpha

    def _run(self, data_loader: DataLoader):
        batch_size = data_loader.batch_size
        for src, _ in data_loader:
            src_features, _ = self.model.encoder(**src)

    # def _init_decoder_state(self, src, src_features, batch_size):
            dec_states = TransformerDecoderState(src)
            dec_states._init_cache(src_features, self.model.decoder.num_layers)

            # Tile states and memory beam_size times.
            dec_states.map_batch_fn(
                lambda state, dim: tile(state, self.beam_size, dim=dim))
            src_features = tile(src_features, self.beam_size, dim=0)
            batch_offset = torch.arange(batch_size,
                                        dtype=torch.long,
                                        device=self.device)
            beam_offset = torch.arange(0,
                                       batch_size * self.beam_size,
                                       step=self.beam_size,
                                       dtype=torch.long,
                                       device=self.device)
            alive_seq = torch.full([batch_size * self.beam_size, 1],
                                   self.bos_token_id,
                                   dtype=torch.long,
                                   device=self.device)

            # Give full probability to the first beam on the first step.
            topk_log_probs = torch.tensor(
                [0.0] + [float("-inf")] * (self.beam_size - 1),
                device=self.device).repeat(batch_size)

            # Structure that holds finished hypotheses.
            hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

            results = {}
            results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["gold_score"] = [0] * batch_size
            results["batch"] = batch

            for step in range(self.max_length):
                decoder_input = alive_seq[:, -1].view(1, -1)

                # Decoder forward.
                decoder_input = decoder_input.transpose(0, 1)

                dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                         step=step)

                # Generator forward.
                log_probs = self.model.generator.forward(
                    dec_out.transpose(0, 1).squeeze(0))
                vocab_size = log_probs.size(-1)

                if step < self.min_length:
                    log_probs[:, self.eos_token_id] = float('-inf')

                # Multiply probs by the beam probability.
                log_probs += topk_log_probs.view(-1).unsqueeze(1)

                alpha = self.global_scorer.alpha
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                # Flatten probs into a list of possibilities.
                curr_scores = log_probs / length_penalty

                if(self.args.block_trigram):

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                # Recover log probs.
                topk_log_probs = topk_scores * length_penalty

                # Resolve beam origin and true word ids.
                topk_beam_index = topk_ids.div(vocab_size)
                topk_ids = topk_ids.fmod(vocab_size)

                # Map beam_index to batch_index in the flat representation.
                batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                select_indices = batch_index.view(-1)

                # Append last prediction.
                alive_seq = torch.cat(
                    [alive_seq.index_select(0, select_indices),
                     topk_ids.view(-1, 1)], -1)

                is_finished = topk_ids.eq(self.end_token)
                if step + 1 == max_length:
                    is_finished.fill_(1)
                # End condition is top beam is finished.
                end_condition = is_finished[:, 0].eq(1)
                # Save finished hypotheses.
                if is_finished.any():
                    predictions = alive_seq.view(-1,
                                                 beam_size, alive_seq.size(-1))
                    for i in range(is_finished.size(0)):
                        b = batch_offset[i]
                        if end_condition[i]:
                            is_finished[i].fill_(1)
                        finished_hyp = is_finished[i].nonzero().view(-1)
                        # Store finished hypotheses for this batch.
                        for j in finished_hyp:
                            hypotheses[b].append((
                                topk_scores[i, j],
                                predictions[i, j, 1:]))
                        # If the batch reached the end, save the n_best hypotheses.
                        if end_condition[i]:
                            best_hyp = sorted(
                                hypotheses[b], key=lambda x: x[0], reverse=True)
                            score, pred = best_hyp[0]

                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                    non_finished = end_condition.eq(0).nonzero().view(-1)
                    # If all sentences are translated, no need to go further.
                    if len(non_finished) == 0:
                        break
                    # Remove finished batches for the next step.
                    topk_log_probs = topk_log_probs.index_select(
                        0, non_finished)
                    batch_index = batch_index.index_select(0, non_finished)
                    batch_offset = batch_offset.index_select(0, non_finished)
                    alive_seq = predictions.index_select(0, non_finished) \
                        .view(-1, alive_seq.size(-1))
                # Reorder states.
                select_indices = batch_index.view(-1)
                src_features = src_features.index_select(0, select_indices)
                dec_states.map_batch_fn(
                    lambda state, dim: state.index_select(dim, select_indices))

        def _block_trigram(self, alive_seq: torch.Tensor, curr_scores):
            cur_len = alive_seq.size(1)
            if(cur_len > 3):
                for i in range(alive_seq.size(0)):
                    fail = False
                    words = [int(w) for w in alive_seq[i]]
                    words = [self.vocab.convert_ids_to_tokens([w])[0]
                             for w in words]
                    words = ' '.join(words).replace(' ##', '').split()
                    if(len(words) <= 3):
                        continue
                    trigrams = [(words[i-1], words[i], words[i+1])
                                for i in range(1, len(words)-1)]
                    trigram = tuple(trigrams[-1])
                    if trigram in trigrams[:-1]:
                        fail = True
                    if fail:
                        curr_scores[i] = -10e20
            return curr_scores


class DecoderState:
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
