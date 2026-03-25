import argparse
import torch
import numpy as np
import cs336_basics.pre_norm_transformer_block as pre_norm_transformer_block
import cs336_basics.optim as optim
import cs336_basics.bpe as bpe
import cs336_basics.tokenizer as tokenizer


def main():
    parser = argparse.ArgumentParser()
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    # 训练超参数
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    # 路径
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 核心：使用 np.memmap 加载大数据集 (Memory-efficient)
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(args.valid_path, dtype=np.uint16, mode='r')

    # 初始化模型
    model = pre_norm_transformer_block.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 启动训练循环
    train_loop(model, optimizer, )


def train_loop(model, optimizer, ):
    pass
