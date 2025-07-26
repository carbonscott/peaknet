import math

class CosineLRScheduler:
    """Simple cosine learning rate scheduler with linear warmup.
    
    Features:
    - Linear warmup from 0 to base_lr over warmup_iterations
    - Cosine annealing from base_lr to min_lr over remaining iterations
    - Deterministic based on step count (perfect for multi-rank and resumption)
    - Zero external dependencies
    """
    def __init__(self, optimizer, warmup_iterations, total_iterations, min_lr=0, last_iteration=-1):
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations
        self.min_lr = min_lr
        self.last_iteration = last_iteration
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        """Calculate current learning rate based on step number."""
        # Warmup phase: linear scaling from 0 to base_lr
        if self.last_iteration < self.warmup_iterations:
            if self.warmup_iterations == 0:
                return self.base_lrs
            return [base_lr * self.last_iteration / self.warmup_iterations for base_lr in self.base_lrs]

        # After training: keep at min_lr
        if self.last_iteration >= self.total_iterations:
            return [self.min_lr for _ in self.base_lrs]

        # Cosine annealing phase
        decay_steps = self.total_iterations - self.warmup_iterations
        decay_ratio = (self.last_iteration - self.warmup_iterations) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

    def step(self):
        """Update step counter and apply new learning rate to optimizer."""
        self.last_iteration += 1
        new_lrs = self.get_lr()
        # Update optimizer learning rates directly
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lrs[i]

    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {
            'last_iteration': self.last_iteration,
            'warmup_iterations': self.warmup_iterations,
            'total_iterations': self.total_iterations,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary from checkpoint."""
        self.last_iteration = state_dict['last_iteration']
        self.warmup_iterations = state_dict['warmup_iterations']
        self.total_iterations = state_dict['total_iterations']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']
