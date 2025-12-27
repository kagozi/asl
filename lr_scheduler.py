import math

class NoamLR:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        self.last_lr = 0.0

    def step(self):
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5))))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.last_lr = lr
        return lr
