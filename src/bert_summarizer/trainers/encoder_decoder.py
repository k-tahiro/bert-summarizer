from logging import getLogger
from typing import Dict, Optional

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments

from .lr_lambda import get_transformer_schedule_with_warmup
from ..utils.nn import get_n_params

logger = getLogger(__name__)


class EncoderDecoderTrainer(Trainer):
    NO_DECAY = {'bias', 'LayerNorm.weight'}

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        encoder_learning_rate: float = 0.002,
        decoder_learning_rate: float = 0.2,
        encoder_warmup_steps: int = 20000,
        decoder_warmup_steps: int = 10000,
        **kwargs
    ):
        logger.info(f'# of encoder parameters={get_n_params(model.encoder)}')
        logger.info(f'# of decoder parameters={get_n_params(model.decoder)}')

        self.is_given_optims = 'optimizers' in kwargs
        if not self.is_given_optims:
            encoder_params = self._get_params(
                model.encoder,
                args.weight_decay,
                encoder_learning_rate
            )
            decoder_params = self._get_params(
                model.decoder,
                args.weight_decay,
                decoder_learning_rate
            )
            params = encoder_params + decoder_params
            optimizer = AdamW(
                params,
                eps=args.adam_epsilon,
            )

            lr_scheduler = get_transformer_schedule_with_warmup(
                optimizer,
                [encoder_warmup_steps] * 2 + [decoder_warmup_steps] * 2
            )

            optimizers = (optimizer, lr_scheduler)
            kwargs['optimizers'] = optimizers

        super(EncoderDecoderTrainer, self).__init__(model, args, **kwargs)

    @classmethod
    def _get_params(cls, model: nn.Module, weight_decay: float, learning_rate: float) -> list:
        return [
            {
                'params': [
                    param
                    for name, param in model.named_parameters()
                    if not any(nd in name for nd in cls.NO_DECAY)
                ],
                'weight_decay': weight_decay,
                'lr': learning_rate
            },
            {
                'params': [
                    param
                    for name, param in model.named_parameters()
                    if any(nd in name for nd in cls.NO_DECAY)
                ],
                'weight_decay': 0.0,
                'lr': learning_rate
            },
        ]

    def log(self, logs: Dict[str, float]) -> None:
        lrs = self.lr_scheduler.get_last_lr()
        if self.is_given_optims:
            for i, lr in enumerate(lrs):
                logs[f'learning_rate/{i}'] = lr
        else:
            logs['learning_rate/encoder_w_decay'] = lrs[0]
            logs['learning_rate/encoder_wo_decay'] = lrs[1]
            logs['learning_rate/decoder_w_decay'] = lrs[2]
            logs['learning_rate/decoder_wo_decay'] = lrs[3]

        logs.pop('learning_rate', None)
        super().log(logs)
