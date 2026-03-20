import torch
import torch.nn as nn
import cs336_basics.basic_building_blocks as basic_building_blocks


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,  # Hidden dimension of the model
            eps: float = 1e-5,  # Epsilon value for numerical stability
            device: torch.device | None = None,  # Device to store the parameters on
            dtype: torch.dtype | None = None  # Data type of the parameters
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


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            # 1. 计算 8/3 * d_model
            raw_d_ff = int(8 / 3 * d_model)
            # 2. 向上取整到 64 的倍数 (Hardware alignment)
            # 公式：(n + alignment - 1) // alignment * alignment
            d_ff = (raw_d_ff + 63) // 64 * 64
        self.w1 = basic_building_blocks.Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = basic_building_blocks.Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = basic_building_blocks.Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        w1x = self.w1(x)
        w3x = self.w3(x)
        return self.w2(w1x * torch.sigmoid(w1x) * w3x)

