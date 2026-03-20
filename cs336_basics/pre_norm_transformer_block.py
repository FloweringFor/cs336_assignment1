import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,                         # Hidden dimension of the model
            eps: float = 1e-5,                    # Epsilon value for numerical stability
            device: torch.device | None = None,   # Device to store the parameters on
            dtype: torch.dtype | None = None      # Data type of the parameters
    ):
        """
        Construct the RMSNorm module.
        """
        super(RMSNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        """
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        rms_norm = x * self.gain / rms
        return rms_norm.to(in_type)
