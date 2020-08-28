from logging import getLogger
from typing import Optional

import torch

logger = getLogger(__name__)


class BeamSearch:
    def __init__(self,
                 batch_size: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 beam_size: int = 5,
                 device: Optional[torch.device] = None,
                 alpha: float = .6,
                 block_trigram: bool = True):
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
        cur_len = self.alive_seq.size(1)
        if cur_len > 6:
            for i, token_ids in enumerate(self.alive_seq.tolist()):
                trigrams = [(token_ids[j-2], token_ids[j-1], token_ids[j])
                            for j in range(2, len(token_ids))]
                bigram = trigrams[-1][1:]
                for trigram in trigrams:
                    if bigram == trigram[:2]:
                        curr_scores[i, trigram[2]] = float('-inf')
        return curr_scores
