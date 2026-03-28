import torch
import numpy as np
import train  # 假设你的训练脚本叫 train.py
import cs336_basics.tokenizer as tokenizer
import cs336_basics.pre_norm_transformer_block as pre_norm_transformer_block

# 1. 基础配置 (根据你 train.py 的参数对齐)
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "/root/autodl-tmp/cs336_assignment1/checkpoints/owt/ckpt_best.pt"
valid_path = "/root/autodl-tmp/cs336_assignment1/datasets/owt_valid.bin"

# 2. 加载模型与权重
print(f"--- 正在加载模型: {ckpt_path} ---")
checkpoint = torch.load(ckpt_path, map_location=device)


model = pre_norm_transformer_block.TransformerLM(
    vocab_size=50257, context_length=512, d_model=768,
    num_layers=12, num_heads=12, d_ff=3072, rope_theta=10000.0
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 计算验证集 Loss (复用你 train.py 里的函数)
print("\n正在计算最终 Validation Loss...")
valid_data = np.fromfile(valid_path, dtype=np.uint16)


# 模拟一个 args 对象供函数使用
class Args:
    eval_batch = 200  # 采样 200 个 batch 算平均值
    batch_size = 64
    context_length = 512
    temperature = 0.8
    top_p = 0.8
    max_new_tokens = 150


args = Args()
# v_loss = train.estimate_loss(model, valid_data, args, device)
# print(f"✅ 最终验证集 Loss: {v_loss:.4f}")

# 4. 生成文本测试
tok = tokenizer.Tokenizer.from_files(
    "/root/autodl-tmp/cs336_assignment1/datasets/owt_vocab.json",
    "/root/autodl-tmp/cs336_assignment1/datasets/owt_merges.txt",
    ["<|endoftext|>"]
)

prompts = ["The Earth is the third planet from the sun and it has",
           "If you drop an apple, it will fall to the ground because",
           "Once upon a time, in a world made of binary code, there was a",
           "he president told reporters at the White House that"]

print("\n--- 最终模型采样展示 ---")
for p in prompts:
    text_idx = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
    # 直接调用你 train.py 里的 generate_text
    gen_idx = train.generate_text(model, text_idx, args, eos_token_id=tok.encode("<|endoftext|>")[0])
    print(f"\nPrompt: {p}\nOutput: {tok.decode(gen_idx[0].tolist())}")
    print("-" * 30)
