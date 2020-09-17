from functools import partial
from logging import getLogger
from typing import Dict, Optional

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments

logger = getLogger(__name__)


class EncoderDecoderTrainer(Trainer):
    NO_DECAY = {'bias', 'LayerNorm.weight'}

    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 encoder_learning_rate: float = 0.002,
                 decoder_learning_rate: float = 0.2,
                 encoder_warmup_steps: int = 20000,
                 decoder_warmup_steps: int = 10000,
                 **kwargs):
        n_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
        logger.info(f'# of parameters={n_params}')
        super(EncoderDecoderTrainer, self).__init__(model, args, **kwargs)

        self.is_given_optims = 'optimizers' in kwargs
        if not self.is_given_optims:
            encoder_params = self._get_params(self.model.encoder,
                                              self.args.weight_decay,
                                              encoder_learning_rate)
            decoder_params = self._get_params(self.model.decoder,
                                              self.args.weight_decay,
                                              decoder_learning_rate)
            params = encoder_params + decoder_params
            self.optimizer = AdamW(
                params,
                eps=self.args.adam_epsilon,
            )

            if self.args.max_steps > 0:
                num_training_steps = self.args.max_steps
            else:
                train_dataloader = self.get_train_dataloader()
                num_training_steps = int(len(train_dataloader)
                                         // self.args.gradient_accumulation_steps
                                         * self.args.num_train_epochs)

            def lr_lambda(num_warmup_steps: int, current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) /
                    float(max(1, num_training_steps - num_warmup_steps))
                )

            encoder_lr_lambda = partial(lr_lambda, encoder_warmup_steps)
            decoder_lr_lambda = partial(lr_lambda, decoder_warmup_steps)

            lr_lambdas = [encoder_lr_lambda] * 2 + [decoder_lr_lambda] * 2

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambdas)

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

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
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
        super().log(logs, iterator)
