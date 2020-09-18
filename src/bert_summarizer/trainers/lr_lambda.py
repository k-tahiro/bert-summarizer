def linear(num_training_steps: int, num_warmup_steps: int, current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step) /
        float(max(1, num_training_steps - num_warmup_steps))
    )


def linear_then_invsqrt(num_warmup_steps: int, current_step: int):
    current_step += 1
    return min(
        current_step ** -.5,
        current_step * num_warmup_steps ** -1.5
    )
