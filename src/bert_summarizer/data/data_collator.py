from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class DataCollatorWithPaddingWithAdditionalFeatures(DataCollatorWithPadding):
    additional_features: List[str] = field(default_factory=list)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        additional_features = defaultdict(list)
        for feature in features:
            for key in self.additional_features:
                additional_features[key].append(feature[key])
        features = [
            {
                k: v
                for k, v in feature.items()
                if k not in self.additional_features
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        max_length = batch['input_ids'].size(1)
        batch.update({
            key: self.pad(value, max_length)
            for key, value in additional_features.items()
        })

        if 'label' in batch:
            batch['labels'] = batch['label']
            del batch['label']

        return batch

    def pad(self, feature: List[Union[List[int], torch.Tensor]], max_length: int) -> torch.Tensor:
        matrix = []
        for row in feature:
            difference = max_length - len(row)
            row_filled = row + [0] * difference
            matrix.append(row_filled)
        return torch.tensor(matrix)


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
        if any(decoder_features):
            decoder_batch = self.decoder_tokenizer.pad(
                decoder_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors='pt',
            )
        else:
            decoder_batch = dict()

        batch = encoder_batch
        batch.update(dict(
            (f'decoder_{k}', v)
            for k, v in decoder_batch.items()
        ))

        batch['decoder_encoder_input_ids'] = batch['input_ids']

        if self.generate_labels:
            batch['labels'] = batch['decoder_input_ids']

        return batch
