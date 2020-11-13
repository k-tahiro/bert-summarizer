from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class EncoderDecoderDataCollatorWithPadding(DataCollatorWithPadding):
    decoder_tokenizer: Optional[Union[PreTrainedTokenizer,
                                      PreTrainedTokenizerFast]] = None
    generate_labels: bool = False

    def __post_init__(self):
        if self.decoder_tokenizer is None:
            self.decoder_tokenizer = self.tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        encoder_features = []
        decoder_features = []

        for feature in features:
            encoder_feature = dict()
            decoder_feature = dict()
            for k, v in feature.items():
                if k.startswith('decoder_'):
                    decoder_feature[k[len('decoder_'):]] = v
                else:
                    encoder_feature[k] = v
            encoder_features.append(encoder_feature)
            decoder_features.append(decoder_feature)

        encoder_batch = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        decoder_batch = self.decoder_tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        batch = encoder_batch
        batch.update(dict(
            (f'decoder_{k}', v)
            for k, v in decoder_batch.items()
        ))

        batch['decoder_encoder_input_ids'] = batch['input_ids']

        if self.generate_labels:
            batch['labels'] = batch['decoder_input_ids']

        return batch
