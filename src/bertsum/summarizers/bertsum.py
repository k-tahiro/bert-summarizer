from logging import getLogger
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from ..datasets.bertsum import BertSumDataset, BertSumExtDataset, BertSumAbsDataset
from ..models.bertsum import BertSum
from ..utils.beam_search import BeamSearch
from ..utils.tensor import tile


logger = getLogger(__name__)


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

    def _create_data_loader(self, src: Union[str, List[str], BertSumDataset]) -> DataLoader:
        if isinstance(src, str):
            src = [src]
        batch_size = self.batch_size or len(src)

        if isinstance(src, (str, list)):
            tgt = batch_size * ['dummy']  # dummy target
            dataset = self.Dataset(src,
                                   tgt,
                                   self.model.model_type,
                                   return_labels=False)
        else:
            dataset = src

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
        beam_search = BeamSearch(data_loader.batch_size,
                                 self.bos_token_id,
                                 self.eos_token_id,
                                 beam_size,
                                 self.device,
                                 alpha,
                                 block_trigram)

        results = []
        for batch in data_loader:
            beam_search.initialize()

            src = {
                k.replace('src_', ''): v
                for k, v in batch.items()
                if k.startswith('src_')
            }

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

            results.extend(beam_search.hypothesis)

        return [
            [
                (hyp[0], self.tokenizer.convert_ids_to_tokens(hyp[1]))
                for hyp in result
            ]
            for result in results
        ]

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
