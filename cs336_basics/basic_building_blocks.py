import math
from typing import Optional

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,                     # final dimension of the input
            out_features: int,                    # final dimension of the output
            device: Optional[torch.device] = None,   # Device to store the parameters on
            dtype: Optional[torch.dtype] = None      # Data type of the parameters
    ):
        """
        Construct a linear transformation module.
        """
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return x @ self.weights.T


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,                  # Size of the vocabulary
            embedding_dim: int,                   # Dimension of the embedding vectors, i.e., d_model
            device: Optional[torch.device] = None,   # Device to store the parameters on
            dtype: Optional[torch.dtype] = None      # Data type of the parameters
    ):
        """
        Construct an embedding module.
        """
        super(Embedding, self).__init__()
        self.emb = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.emb, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]

