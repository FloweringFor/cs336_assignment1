import os
import argparse
import torch
import numpy as np
import cs336_basics.pre_norm_transformer_block as pre_norm_transformer_block
import cs336_basics.optim as optim
import cs336_basics.dataloader as dataloader


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
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--save_iters", type=int, default=1000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--eval_batch", type=int, default=10)
    # 路径
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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
    train_loop(model, optimizer, train_data, valid_data, args, device)


def train_loop(model, optimizer, train_data, valid_data, args, device):
    best_loss = float('inf')
    model.train()
    running_loss = 0
    for iter in range(args.max_iters):
        # 1.获取数据
        xb, yb = dataloader.data_loading(train_data, args.batch_size, args.context_length, device)
        # 2.前向传播
        logits = model(xb)
        # 计算loss
        curr_loss = optim.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        running_loss += curr_loss.item()
        # 3.后向传播
        optimizer.zero_grad()
        curr_loss.backward()
        optim.gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()
        # 4. 打印日志与验证
        if iter % args.eval_iters == 0 and iter > 0:
            val_loss = estimate_loss(model, valid_data, args, device)
            print(f"Iter {iter}: avg train loss {running_loss / args.eval_iters:.4f}, val loss {val_loss:.4f}")
            running_loss = 0
            if val_loss < best_loss:
                dataloader.save_checkpoint(model, optimizer, iter, f"{args.save_path}/ckpt_best.pt")
                print(f"Iter {iter} save best model!")
                best_loss = val_loss
        # 5.保存模型
        if iter % args.save_iters == 0 or iter == args.max_iters - 1:
            dataloader.save_checkpoint(model, optimizer, iter, f"{args.save_path}/ckpt_{iter}.pt")


@torch.no_grad()
def estimate_loss(model, valid_data, args, device):
    model.eval()
    losses = []
    for _ in range(args.eval_batch):
        xb, yb = dataloader.data_loading(valid_data, args.batch_size, args.context_length, device)
        logits = model(xb)
        loss = optim.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


if __name__ == '__main__':
    main()
