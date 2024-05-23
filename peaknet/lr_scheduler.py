from torch.optim.lr_scheduler import _LRScheduler

import math

class CosineLRScheduler(_LRScheduler):
    """ Iteration can mean an epoch, a micro batch or a mini batch.
    """
    def __init__(self, optimizer, warmup_iterations, total_iterations, min_lr=0, last_iteration=-1):
        self.warmup_iterations = warmup_iterations
        self.total_iterations  = total_iterations
        self.min_lr            = min_lr
        self.decay_iterations  = self.total_iterations - self.warmup_iterations
        self.last_iteration    = last_iteration
        self.base_lrs          = [group['lr'] for group in optimizer.param_groups]

    def get_lr(self):
        """
        This is called in `.step()` that is at the end of a training loop.

        For example, last_iteration will turn to 1 from (-1) after the very initail
        loop.
        """
        # Warming up???
        if self.last_iteration < self.warmup_iterations:
            return [base_lr * self.last_iteration / self.warmup_iterations for base_lr in self.base_lrs]

        # After decay???
        if self.last_iteration > self.total_iterations:
            return [self.min_lr]

        # Cosine decay...
        decay_ratio = (self.last_iteration - self.warmup_iterations) / self.decay_iterations
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

    def sync_with_optimizer(self, new_lrs):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lrs[i]

    def step(self):
        self.last_iteration += 1
        new_lrs = self.get_lr()
        self.sync_with_optimizer(new_lrs)

    def reset(self):
        # Reset last_iteration to its initial value
        self.last_iteration = -1

        # Reset optimizer's learning rates to initial base_lrs
        self.sync_with_optimizer(self.base_lrs)

    def state_dict(self):
        return {
            'last_iteration'   : self.last_iteration,
            'warmup_iterations': self.warmup_iterations,
            'total_iterations' : self.total_iterations,
            'min_lr'           : self.min_lr,
            'base_lrs'         : self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.last_iteration    = state_dict['last_iteration']
        self.warmup_iterations = state_dict['warmup_iterations']
        self.total_iterations  = state_dict['total_iterations']
        self.min_lr            = state_dict['min_lr']
        self.base_lrs          = state_dict['base_lrs']
