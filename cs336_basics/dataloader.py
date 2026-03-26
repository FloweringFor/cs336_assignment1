import os
import typing
from typing import Union

import torch
import numpy as np
import numpy.typing as npt


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    """
        极简、高效的随机采样 DataLoader
        """
    # 随机产生 batch_size 个起始位置
    # 注意：在 numpy 层面做随机索引，不要先把整个数据集转成 tensor
    n = len(dataset)
    ix = np.random.randint(0, n - context_length - 1, (batch_size,))

    # 提取数据并转换为 Tensor
    # 这里用 list comprehension 配合 torch.from_numpy 是最快的
    x = torch.stack([torch.from_numpy(dataset[i: i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1: i + 1 + context_length].astype(np.int64)) for i in ix])

    # 最后统一移动到设备
    return x.to(device), y.to(device)


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
