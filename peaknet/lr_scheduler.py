from torch.optim.lr_scheduler import LRScheduler

import math

class CosineLRScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.decay_epochs = self.total_epochs - self.warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        This is called in `.step()` that is at the end of a training loop.  

        For example, last_epoch will turn to 1 from (-1) after the very initail
        loop.
        """
        # Warming up???
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]

        # After decay???
        if self.last_epoch > self.total_epochs:
            return [self.min_lr]

        # Cosine decay...
        decay_ratio = (self.last_epoch - self.warmup_epochs) / self.decay_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]
