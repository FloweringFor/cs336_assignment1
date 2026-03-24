import torch


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

