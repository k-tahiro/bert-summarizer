from typing import List, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class TransformerScheduler:
    def __init__(self, num_warmup_steps: int):
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return current_step * self.num_warmup_steps ** -1.5
        return current_step ** -.5


def get_transformer_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: Union[int, List[int]],
    last_epoch: int = -1
) -> LambdaLR:
    if isinstance(num_warmup_steps, int):
        lr_lambda = TransformerScheduler(num_warmup_steps)
    else:
        lr_lambda = [
            TransformerScheduler(n)
            for n in num_warmup_steps
        ]

    return LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch=last_epoch
    )
