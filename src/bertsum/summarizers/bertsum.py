from logging import getLogger
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from ..datasets.bertsum import BertSumDataset, BertSumExtDataset, BertSumAbsDataset
from ..models.bertsum import BertSum, BertSumExt, BertSumAbs
from ..preprocessing.tokenization import BertSumTokenizer
from ..utils.tensor import tile


logger = getLogger(__name__)


class BeamSearch:
    def __init__(self,
                 tokenizer: BertTokenizer,
                 batch_size: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 beam_size: int = 5,
                 device: Optional[torch.device] = None,
                 alpha: float = .6,
                 block_trigram: bool = True):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.beam_size = beam_size
        self.alpha = alpha
        self.block_trigram = block_trigram
        self.device = device
        self.beam_batch_size = self.beam_size * batch_size

    @property
    def seq_len(self):
        return self.alive_seq.size(-1)

    @property
    def length_penalty(self):
        return ((5.0 + self.seq_len) / 6.0) ** self.alpha

    @property
    def finished_length_penalty(self):
        """ignore final token because it should be EOS token."""
        return ((5.0 + self.seq_len - 1) / 6.0) ** self.alpha

    @property
    def scores(self):
        return self.topk_log_probs.view(-1, self.beam_size) / self.length_penalty

    @property
    def finished_scores(self):
        return self.topk_log_probs.view(-1, self.beam_size) / self.finished_length_penalty

    def initialize(self):
        self.batch_index = None
        self.batch_offset = torch.arange(self.batch_size,
                                         dtype=torch.long,
                                         device=self.device)
        self.beam_offset = torch.arange(0,
                                        self.beam_batch_size,
                                        step=self.beam_size,
                                        dtype=torch.long,
                                        device=self.device)
        self.alive_seq = torch.full([self.beam_batch_size, 1],
                                    self.bos_token_id,
                                    dtype=torch.long,
                                    device=self.device)
        self.topk_log_probs = torch.full((self.batch_size, self.beam_size),
                                         float('-inf'),
                                         device=self.device)
        self.topk_log_probs[:, 0] = 0.0
        self.hypothesis = [[] for _ in range(self.batch_size)]
        self.results = [
            {
                'scores': [],
                'predictions': []
            }
            for _ in self.batch_offset
        ]
        self.is_end = False

    def run(self, log_probs: torch.Tensor):
        self.update_beam(log_probs)
        self.finalize()

    def update_beam(self, log_probs: torch.Tensor):
        vocab_size = log_probs.size(-1)

        # Multiply probs by the beam probability.
        logger.debug(f'{self.topk_log_probs.size()=}')
        log_probs += self.topk_log_probs.view(-1).unsqueeze(1)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / self.length_penalty
        curr_scores[:, self.eos_token_id] *= self.length_penalty
        curr_scores[:, self.eos_token_id] /= self.finished_length_penalty

        if self.block_trigram:
            curr_scores = self._block_trigram(curr_scores)

        curr_scores = curr_scores.reshape(-1,
                                          self.beam_size * vocab_size)
        topk_scores, topk_ids = curr_scores.topk(self.beam_size,
                                                 dim=-1)

        # Resolve beam origin and true word ids.
        topk_beam_index = topk_ids.floor_divide(vocab_size)
        topk_ids = topk_ids.fmod(vocab_size)

        # Recover log probs.
        eos_positions = topk_ids.eq(self.eos_token_id)
        topk_scores[eos_positions] *= self.finished_length_penalty
        topk_scores[eos_positions] /= self.length_penalty
        self.topk_log_probs = topk_scores * self.length_penalty

        # Map beam_index to batch_index in the flat representation.
        batch_index = topk_beam_index + \
            self.beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
        select_indices = batch_index.view(-1)

        # Append last prediction.
        alive_seq = self.alive_seq.index_select(0, select_indices)
        self.alive_seq = torch.cat([alive_seq, topk_ids.view(-1, 1)], -1)

        self.batch_index = batch_index

    def finalize(self):
        predictions = self.alive_seq.view(-1,
                                          self.beam_size,
                                          self.alive_seq.size(-1))

        is_finished = predictions[:, :, -1].eq(self.eos_token_id)
        if self.is_end:
            is_finished.fill_(True)

        # End condition is top beam is finished.
        end_condition = is_finished[:, 0].eq(True)

        # Save finished hypothesis.
        if is_finished.any():
            for i in range(is_finished.size(0)):
                b = self.batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero().view(-1)

                # Store finished hypotheses for this batch.
                for j in finished_hyp:
                    self.hypothesis[b].append((self.finished_scores[i, j],
                                               predictions[i, j, 1:] if self.is_end else predictions[i, j, 1:-1]))

                # If the batch reached the end, save the n_best hypothesis.
                if end_condition[i]:
                    best_hyp = sorted(self.hypothesis[b],
                                      key=lambda x: x[0],
                                      reverse=True)
                    score, pred = best_hyp[0]

                    self.results[b]['scores'].append(score)
                    self.results[b]['predictions'].append(pred)

            non_finished = end_condition.eq(False).nonzero().view(-1)
            # If all sentences are translated, no need to go further.
            if non_finished.size() == 0:
                self.is_end = True
                return

            # Remove finished batches for the next step.
            self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                                   non_finished)
            self.batch_index = self.batch_index.index_select(0, non_finished)
            self.batch_offset = self.batch_offset.index_select(0, non_finished)
            self.alive_seq = predictions.index_select(0, non_finished) \
                                        .view(-1, self.alive_seq.size(-1))

    def _block_trigram(self, curr_scores: torch.Tensor) -> torch.Tensor:
        convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens
        cur_len = self.alive_seq.size(1)
        if cur_len > 3:
            for i, token_ids in enumerate(self.alive_seq.tolist()):
                words = convert_ids_to_tokens(token_ids)
                words = ' '.join(words).replace(' ##', '').split()
                if len(words) <= 3:
                    continue
                trigrams = [(words[j-2], words[j-1], words[j])
                            for j in range(2, len(words))]
                trigram = trigrams[-1]
                if trigram in trigrams[:-1]:
                    curr_scores[i] = float('-inf')
        return curr_scores


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
        self.device = device

    def __call__(self, src: Union[str, List[str]], *args, **kwargs) -> dict:
        data_loader = self._create_data_loader(src)
        with torch.no_grad():
            return self._run(data_loader, *args, **kwargs)

    def _create_data_loader(self, src: Union[str, List[str]]) -> DataLoader:
        if isinstance(src, str):
            src = [src]

        batch_size = self.batch_size or len(src)
        tgt = batch_size * ['dummy']  # dummy target
        dataset = self.Dataset(src, tgt, self.model.model_type)
        self.tokenizer = dataset.tgt_tokenizer.tokenizer
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        return DataLoader(dataset, batch_size=batch_size)

    def _run(self, data_loader: DataLoader):
        raise NotImplementedError


class BertSumExtSummarizer(BertSumSummarizer):
    Dataset = BertSumExtDataset


class BertSumAbsSummarizer(BertSumSummarizer):
    Dataset = BertSumAbsDataset

    def _run(self,
             data_loader: DataLoader,
             max_length: Optional[int] = None,
             min_length: int = 0,
             alpha: float = .6,
             beam_size: int = 5,
             block_trigram: bool = True):
        max_length = max_length or self.model.encoder.config.max_position_embeddings
        beam_search = BeamSearch(self.tokenizer,
                                 data_loader.batch_size,
                                 self.bos_token_id,
                                 self.eos_token_id,
                                 beam_size,
                                 self.device,
                                 alpha,
                                 block_trigram)

        results = []
        for src, _ in data_loader:
            beam_search.initialize()

            memory, _ = self.model.encoder(**src)
            memory = memory.permute(1, 0, 2).contiguous()
            memory_key_padding_mask = src['attention_mask'] == 0

            # beamed batch features
            memory = tile(memory, beam_search.beam_size, dim=1)
            memory_key_padding_mask = tile(memory_key_padding_mask,
                                           beam_search.beam_size,
                                           dim=0)

            for step in range(max_length):
                log_probs = self._calc_token_probs(beam_search.alive_seq,
                                                   step,
                                                   memory,
                                                   memory_key_padding_mask,
                                                   min_length)
                is_end = step + 1 == max_length
                if is_end:
                    beam_search.is_end = True
                beam_search.run(log_probs)

                # Reorder states.
                select_indices = beam_search.batch_index.view(-1)
                memory = memory.index_select(1, select_indices)
                memory_key_padding_mask = memory_key_padding_mask.index_select(0,
                                                                               select_indices)

            results.extend(beam_search.results)

        return results

    def _calc_token_probs(self,
                          alive_seq: torch.Tensor,
                          step: int,
                          memory: torch.Tensor,
                          memory_key_padding_mask: torch.Tensor,
                          min_length: int) -> torch.Tensor:
        # decode
        logger.debug(f'{alive_seq.size()=}')
        tgt = self.model.embeddings(alive_seq)
        tgt = self.model.pos_emb(tgt, step=step)
        tgt = tgt.permute(1, 0, 2).contiguous()
        dec_out = self.model.decoder(tgt, memory,
                                     memory_key_padding_mask=memory_key_padding_mask)

        # Generator forward.
        dec_out = dec_out.permute(1, 0, 2).contiguous()
        log_probs = self.model.generator(dec_out)
        log_probs[:, :min_length+1, self.eos_token_id] = float('-inf')
        log_probs = log_probs.index_select(1, torch.tensor(log_probs.size(1)-1)) \
                             .squeeze()  # select final position

        logger.debug(f'{log_probs.size()=}')
        return log_probs
