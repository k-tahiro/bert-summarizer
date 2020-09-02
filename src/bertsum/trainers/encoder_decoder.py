from functools import partial

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel, Trainer, TrainingArguments


class EncoderDecoderTrainer(Trainer):
    NO_DECAY = {'bias', 'LayerNorm.weight'}

    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 encoder_learning_rate: float = 0.0002,
                 decoder_learning_rate: float = 0.2,
                 encoder_warmup_steps: int = 20000,
                 decoder_warmup_steps: int = 10000,
                 **kwargs):
        super(EncoderDecoderTrainer, self).__init__(model, args, **kwargs)

        if 'optimizers' not in kwargs:
            encoder_params = self._get_params(self.model.encoder,
                                              self.args.weight_decay,
                                              encoder_learning_rate)
            decoder_params = self._get_params(self.model.decoder,
                                              self.args.weight_decay,
                                              decoder_learning_rate)
            params = encoder_params + decoder_params
            self.optimizer = AdamW(
                params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
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
                'learning_rate': learning_rate
            },
            {
                'params': [
                    param
                    for name, param in model.named_parameters()
                    if any(nd in name for nd in cls.NO_DECAY)
                ],
                'weight_decay': 0.0,
                'learning_rate': learning_rate
            },
        ]
