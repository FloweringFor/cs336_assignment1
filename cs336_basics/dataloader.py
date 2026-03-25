import os
import typing
from typing import Union

import torch
import numpy as np
import numpy.typing as npt


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    # 1. 将 numpy 转换为 torch tensor 并移至对应设备
    data = torch.from_numpy(dataset.astype(np.int64)).to(device)
    n = len(data)

    # 2. 计算所有可能的起始位置
    # 每个样本需要 context_length 个输入 + 1 个目标值
    max_start_idx = n - context_length - 1
    start_indices = np.arange(max_start_idx + 1)

    # 3. 打乱索引（训练时通常需要）
    np.random.shuffle(start_indices)

    # 4. 按 batch_size 循环处理
    for i in range(0, len(start_indices), batch_size):
        if i + batch_size < len(start_indices):
            batch_start_steps = start_indices[i: i + batch_size]
            x_batch = torch.stack([data[idx: idx + context_length] for idx in batch_start_steps])
            y_batch = torch.stack([data[idx + 1: idx + context_length + 1] for idx in batch_start_steps])

            yield x_batch, y_batch


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    将模型、优化器状态及迭代次数保存到指定路径或文件对象中。
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
