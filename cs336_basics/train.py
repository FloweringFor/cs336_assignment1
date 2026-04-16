import os
import argparse
import torch
import numpy as np
import pre_norm_transformer_block
import optim
import dataloader
import tokenizer
from torch.amp import GradScaler, autocast


def main():
    parser = argparse.ArgumentParser()
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    # 训练超参数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--eval_iters", type=int, default=1000)
    parser.add_argument("--save_iters", type=int, default=1000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    parser.add_argument("--eval_batch", type=int, default=300)
    # 预测超参数
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--text", type=str,
                        default="The president told reporters at the White House that")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    # 路径
    parser.add_argument("--train_path", type=str,
                        default="/root/autodl-tmp/cs336_assignment1/datasets/owt_train.bin")
    parser.add_argument("--valid_path", type=str,
                        default="/root/autodl-tmp/cs336_assignment1/datasets/owt_valid.bin")
    parser.add_argument("--vocab_path", type=str,
                        default="/root/autodl-tmp/cs336_assignment1/datasets/owt_vocab.json")
    parser.add_argument("--merges_path", type=str,
                        default="/root/autodl-tmp/cs336_assignment1/datasets/owt_merges.txt")
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/cs336_assignment1/checkpoints/owt")

    parser.add_argument("--load_model", type=bool, default=True)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 核心：使用 np.memmap 加载大数据集 (Memory-efficient)
    # train_data = np.memmap(args.train_path, dtype=np.uint16, mode='r')
    # valid_data = np.memmap(args.valid_path, dtype=np.uint16, mode='r')
    train_data = np.fromfile(args.train_path, dtype=np.uint16)
    valid_data = np.fromfile(args.valid_path, dtype=np.uint16)

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

    # 参数对齐
    alpha_max = args.lr  # 1e-3
    alpha_min = args.lr * 0.01  # 通常降到最大值的 10%
    t_w = 2000  # Warmup 步数
    t_c = args.max_iters  # 整个 Cosine 退火结束的步数

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=alpha_max)

    start_iter = 0
    if args.load_model and os.path.exists(f"{args.save_path}/ckpt_best.pt"):
        checkpoint = f"{args.save_path}/ckpt_best.pt"
        start_iter = dataloader.load_checkpoint(checkpoint, model, optimizer)
    else:
        print("未发现 checkpoint 或未开启加载模式，将从头开始训练。")

    # 启动训练循环
    train_loop(model, optimizer, train_data, valid_data, start_iter, args, device, tok, alpha_max, alpha_min, t_w, t_c)


def train_loop(model, optimizer, train_data, valid_data, start_iter, args, device, tok, alpha_max, alpha_min, t_w, t_c):
    best_loss = float('inf')
    model.train()
    running_loss = 0
    text_idx = torch.tensor([tok.encode(args.text)], dtype=torch.long, device=device)

    # --- 核心配置：梯度累积与混合精度 ---
    # 物理 batch 为 16，累积 4 步后有效 batch 为 64
    accumulation_steps = 4
    scaler = GradScaler('cuda')

    # 预先清空梯度
    optimizer.zero_grad(set_to_none=True)

    for iter in range(start_iter, args.max_iters):

        # 1. 更新学习率
        current_lr = optim.learning_rate_schedule(iter, alpha_max, alpha_min, t_w, t_c)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 2. 获取数据
        xb, yb = dataloader.data_loading(train_data, args.batch_size, args.context_length, device)

        # 3. 前向传播 + AMP (3090 强烈建议 bfloat16)
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb)
            curr_loss = optim.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            # 缩放 Loss 以适配累积步数
            loss_to_backward = curr_loss / accumulation_steps

        # 4. 后向传播 (累加梯度而不清空)
        scaler.scale(loss_to_backward).backward()
        running_loss += curr_loss.item()

        # 5. 只有在达到累积步数时，才执行物理参数更新
        if (iter + 1) % accumulation_steps == 0:
            # 梯度裁剪：在 step 之前必须先 unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_l2_norm)

            # 执行 Step
            scaler.step(optimizer)
            scaler.update()

            # 更新完后彻底释放梯度张量显存
            optimizer.zero_grad(set_to_none=True)

        # 6. 打印日志与验证
        if iter % args.eval_iters == 0 and iter > start_iter:
            val_loss = estimate_loss(model, valid_data, args, device)
            print(f"Iter {iter}: avg train loss {running_loss / args.eval_iters:.4f}, "
                  f"val loss {val_loss:.4f}, current lr {current_lr:.2e}")
            running_loss = 0

            if val_loss < best_loss:
                dataloader.save_checkpoint(model, optimizer, iter, f"{args.save_path}/ckpt_best.pt")
                print(f"Iter {iter} save best model!")
                best_loss = val_loss
                gen_text_idx = generate_text(model, text_idx, args, eos_token_id=tok.encode("<|endoftext|>")[0])
                print("生成文字：", tok.decode(gen_text_idx[0].tolist()))

        # 7. 保存模型
        if iter % args.save_iters == 0 or iter == args.max_iters - 1:
            dataloader.save_checkpoint(model, optimizer, iter, f"{args.save_path}/ckpt_{iter}.pt")


@torch.no_grad()
def estimate_loss(model, valid_data, args, device):
    model.eval()
    losses = []
    # 验证时也启用混合精度，保证评估环境与训练一致
    with autocast('cuda', dtype=torch.bfloat16):
        for _ in range(args.eval_batch):
            xb, yb = dataloader.data_loading(valid_data, args.batch_size, args.context_length, device)
            logits = model(xb)
            loss = optim.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


@torch.no_grad()
def generate_text(model, idx, args, eos_token_id=None, repetition_penalty=1.2):
    """
    idx: (Batch, T) 的初始 token 序列
    repetition_penalty: 1.0 为无惩罚，建议 1.1-1.3 之间
    """
    model.eval()

    # 获取设备信息，确保计算在同一设备
    device = idx.device

    for _ in range(args.max_new_tokens):
        # 1. 截断上下文
        idx_cond = idx[:, -args.context_length:]

        # 2. 前向传播 (建议开启 AMP 推理，速度更快)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # (Batch, Vocab)

        # --- 核心新增：Repetition Penalty ---
        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                # 找到当前 Batch 这一行已经出现过的所有 token
                # set() 去重，提升查找效率
                yielded_tokens = set(idx[b].tolist())
                for token_id in yielded_tokens:
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

        # 3. Temperature Scaling
        if args.temperature != 1.0:
            logits = logits / args.temperature

        # 4. Top-p (Nucleus) Sampling
        if args.top_p < 1.0:
            # 排序并计算累积概率
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            # 使用标准 F.softmax 替代自定义模块
            cumulative_probs = torch.cumsum(pre_norm_transformer_block.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累加概率超过 p 的 token
            sorted_indices_to_remove = cumulative_probs > args.top_p
            # 保证至少保留一个 token (Shift logic)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # 将被移除的词的 logits 设为极小值
            # 注意：scatter 之前需要根据 sorted_indices 把 Mask 还原回原词表顺序
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # 5. 转化为概率分布并采样
        probs = pre_norm_transformer_block.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 6. 拼接结果
        idx = torch.cat((idx, idx_next), dim=-1)

        # 7. 判断是否结束
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break

    model.train()
    return idx


if __name__ == '__main__':
    main()
