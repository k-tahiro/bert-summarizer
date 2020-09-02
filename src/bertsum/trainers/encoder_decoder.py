from functools import partial

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel, Trainer, TrainingArguments


class EncoderDecoderTrainer:
    NO_DECAY = {'bias', 'LayerNorm.weight'}

    def __init__(self,
                 model: PreTrainedModel,
                 args: TrainingArguments,
                 encoder_learning_rate: float = 0.0002,
                 decoder_learning_rate: float = 0.2,
                 encoder_warmup_steps: int = 20000,
                 decoder_warmup_steps: int = 10000,
                 **kwargs):
        params = [
            {
                "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in self.NO_DECAY)],
                "weight_decay": args.weight_decay,
                'learning_rate': encoder_learning_rate
            },
            {
                "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in self.NO_DECAY)],
                "weight_decay": 0.0,
                'learning_rate': encoder_learning_rate
            },
            {
                "params": [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in self.NO_DECAY)],
                "weight_decay": args.weight_decay,
                'learning_rate': decoder_learning_rate
            },
            {
                "params": [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in self.NO_DECAY)],
                "weight_decay": 0.0,
                'learning_rate': decoder_learning_rate
            },
        ]
        optimizer = AdamW(
            params,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )

        def lr_lambda(num_warmup_steps: int, num_training_steps: int, current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )

        lr_lambdas = [
            partial(lr_lambda, encoder_warmup_steps, 0),
            partial(lr_lambda, encoder_warmup_steps, 0),
            partial(lr_lambda, decoder_warmup_steps, 0),
            partial(lr_lambda, decoder_warmup_steps, 0),
        ]

        lr_scheduler = LambdaLR(optimizer, lr_lambdas)
        self.trainer = Trainer(model,
                               args,
                               optimizers=(optimizer, lr_scheduler),
                               **kwargs)

    def train(self, **kwargs):
        return self.trainer.train(**kwargs)
