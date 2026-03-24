import math

import torch
from typing import Optional
from collections.abc import Callable, Iterable


# 注意，既要防止正无穷溢出，也要防止负无穷溢出
def cross_entropy(inputs, targets):
    batch_size = len(targets)
    max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
    stable_x = inputs - max_val
    exp_x = torch.exp(stable_x)
    sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
    log_exp = torch.log(sum_exp)
    loss = - torch.mean(stable_x[torch.arange(batch_size), targets] - log_exp)
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 将超参数存入defaults
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                # state['v'] = beta2 * state['v'] + (1 - beta2) * grad**2
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = lr * math.sqrt(1 - beta2**state['t']) / (1 - beta1**state['t'])

                # p.data -= alpha_t * state['m'] / (state['v']**0.5 + eps)
                p.data.addcdiv_(state['m'], state['v'].sqrt().add_(eps), value=-alpha_t)

                # p.data = (1 - lr * wd) * p.data
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                state['t'] += 1
        return loss


def learning_rate_schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        alpha_t = t / t_w * alpha_max
    elif t <= t_c:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t


# 对梯度进行全局裁剪
def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    total_l2_norm = 0
    for param in parameters:
        if param.grad is not None:
            total_l2_norm += (param.grad.data**2).sum()
    total_l2_norm = math.sqrt(total_l2_norm)
    if total_l2_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(max_l2_norm / (total_l2_norm + eps))


