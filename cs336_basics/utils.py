import torch
import torch.nn as nn
from torchinfo import summary

import pre_norm_transformer_block


vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

model = pre_norm_transformer_block.TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    num_layers=num_layers,
    num_heads=num_heads,
    d_model=d_model,
    d_ff=d_ff,
    rope_theta=1000.0
)


def transformer_accounting(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_params}")
    summary(model, input_size=(64, context_length), dtypes=[torch.long])


transformer_accounting(model)
