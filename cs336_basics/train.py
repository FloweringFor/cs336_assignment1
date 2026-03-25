import os
import argparse
import torch
import numpy as np
import cs336_basics.pre_norm_transformer_block as pre_norm_transformer_block
import cs336_basics.optim as optim
import cs336_basics.dataloader as dataloader
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
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--save_iters", type=int, default=1000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--eval_batch", type=int, default=10)
    # 预测超参数
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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

    special_tokens = ["<|endoftext|>"]
    tok = tokenizer.Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens)

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
    train_loop(model, optimizer, train_data, valid_data, args, device, tok)


def train_loop(model, optimizer, train_data, valid_data, args, device, tok):
    best_loss = float('inf')
    model.train()
    running_loss = 0
    text_idx = torch.tensor([tok.encode(args.text)], dtype=torch.long, device=device)
    for iter in range(args.max_iters):
        # 1. 获取数据
        xb, yb = dataloader.data_loading(train_data, args.batch_size, args.context_length, device)
        # 2.1 前向传播
        logits = model(xb)
        # 2.2 计算loss
        curr_loss = optim.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        running_loss += curr_loss.item()
        # 3. 后向传播
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
                gen_text_idx = generate_text(model, text_idx, args, eos_token_id=tok.encode("<|endoftext|>")[0])
                print("生成文字：", tok.decode(gen_text_idx[0].tolist()))
        # 5. 保存模型
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


@torch.no_grad()
def generate_text(model, idx, args, eos_token_id=None):
    """
    idx: (Batch, T) 的初始 token 序列
    max_new_tokens: 最多生成多少个新 token
    temperature: 温度系数 (0, inf)
    top_p: Nucleus sampling 阈值 (0, 1]
    eos_token_id: 停止符 ID
    context_length: 模型允许的最大上下文长度
    """
    model.eval()
    for _ in range(args.max_new_tokens):
        # 1. 截断上下文（如果超过了模型 context_length）
        idx_cond = idx[:, -args.context_length:]
        # 2. 前向传播获取最后一个时刻的 logits
        logits = model(idx_cond)  # (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :]  # 只取最后一个 token 的预测结果
        # 3. Temperature Scaling
        if args.temperature != 1.0:
            logits = logits / args.temperature
        # 4. Top-p (Nucleus) Sampling
        if args.top_p < 1.0:
            # a. 排序
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(pre_norm_transformer_block.softmax(sorted_logits, dim=-1), dim=-1)

            # b. 移除累加概率超过p的token
            sorted_indices_to_remove = cumulative_probs > args.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # c. 将被移除的词的 logits 设为负无穷
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # 5. 转化为概率分布并采样
        probs = pre_norm_transformer_block.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 6. 拼接结果
        idx = torch.cat((idx, idx_next), dim=-1)

        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break
    model.train()
    return idx


if __name__ == '__main__':
    main()
